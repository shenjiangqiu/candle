//! in this version, same as sink, but there are 2 different
//! 1. we store the tensor after ROPE
//! 2. for tensors in sink, we use the full tensor, for others, we use the MSB

use super::with_tracing::{linear_no_bias as linear, Linear};
use candle::{CpuStorage, CustomOp1, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use tracing::{debug, info};
pub static ENABLE_SAVE: AtomicBool = AtomicBool::new(false);
pub static SAVE_PATH: RwLock<String> = RwLock::new(String::new());

/// changes here: allow super long sequences
pub const MAX_SEQ_LEN: usize = 4096 * 10;
pub const MAX_WINDOW_SIZE: usize = 1024;
pub const MAX_SINK_SIZE: usize = 40;
#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
        }
    }
}

pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

#[derive(Clone)]
struct KVStore {
    sink_kv: Vec<Option<(Tensor, Tensor)>>,
}

impl KVStore {
    pub fn new(blocks: usize) -> Self {
        Self {
            sink_kv: vec![None; blocks],
        }
    }

    pub fn clear(&mut self) {
        self.sink_kv.iter_mut().for_each(|x| *x = None);
    }
    /// save the tensor into sink or window
    #[allow(unused)]
    pub fn save_tensor(
        &mut self,
        k: Tensor,
        v: Tensor,
        _index_pos: usize,

        block_idx: usize,
    ) -> Result<()> {
        self.save_sink(block_idx, &k, &v)?;
        Ok(())
    }
    /// get the next token's real tensors for attention.
    // Removed the unused method get_tensor_all

    /// return the tensor at 0..index_pos.
    /// ## note that index_pos is the next token of the current sequence. just like old_index+seq_len
    pub fn get_tensor_at(&self, block_idx: usize, index_pos: usize) -> (Tensor, Tensor) {
        debug!("[{block_idx}]get_tensor_at: index_pos: {index_pos}");
        let (result_k, result_v) = self.sink_kv[block_idx].clone().unwrap();
        if index_pos <= MAX_SINK_SIZE + MAX_WINDOW_SIZE {
            (
                result_k.narrow(1, 0, index_pos).unwrap(),
                result_v.narrow(1, 0, index_pos).unwrap(),
            )
        } else {
            let seq_len = result_k.dims3().unwrap().1;
            let sink_part_k = result_k.narrow(1, 0, MAX_SINK_SIZE).unwrap();
            // recent window
            // at least MAX_SINK_SIZE +1
            let winow_part_k = result_k
                .narrow(1, index_pos - MAX_WINDOW_SIZE, MAX_WINDOW_SIZE)
                .unwrap();
            let result_k = Tensor::cat(&[&sink_part_k, &winow_part_k], 1).unwrap();

            let sink_part_v = result_v.narrow(1, 0, MAX_SINK_SIZE).unwrap();
            // recent window
            let winow_part_v = result_v
                .narrow(1, seq_len - MAX_WINDOW_SIZE, MAX_WINDOW_SIZE)
                .unwrap();
            let result_v = Tensor::cat(&[&sink_part_v, &winow_part_v], 1).unwrap();

            (result_k, result_v)
        }
    }

    fn save_sink(&mut self, block_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let sink = &mut self.sink_kv[block_idx];

        if let Some(sink) = sink {
            sink.0 = Tensor::cat(&[&sink.0, k], 1)?.contiguous()?;
            sink.1 = Tensor::cat(&[&sink.1, v], 1)?.contiguous()?;
        } else {
            *sink = Some((k.clone(), v.clone()));
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct StreamCache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<KVStore>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl StreamCache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(KVStore::new(config.num_hidden_layers))),
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&self, t: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn clear_kv_cache(&self) {
        let mut kvs = self.kvs.lock().unwrap();
        kvs.clear();
        // let mut masks = self.masks.lock().unwrap();
        // masks.clear();
    }
}

struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}
#[allow(dead_code)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: StreamCache,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
#[allow(dead_code)]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _, seq_len, hidden_size) = x.dims4()?;
        let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let x1 = x.narrow(D::Minus1, 0, hidden_size / 2)?;
        let x2 = x.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        _sink_size: usize,
        _window_size: usize,
    ) -> Result<Tensor> {
        debug!("index_pos: {index_pos}, block_idx: {block_idx}");
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        // println!("attantion:q {:?}", q.dims());
        // println!("attantion:k {:?}", k.dims());
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let mut k = self.apply_rotary_emb(&k, index_pos)?;

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache.sink_kv[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2).unwrap().contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2).unwrap().contiguous()?;
            }
            cache.sink_kv[block_idx] = Some((k.clone(), v.clone()))
        }

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        // we cannot do attention in a round, because the window is dynamic, we should simulate the dynamic behavior
        // println!("start att");
        let att = if seq_len == 1 {
            // let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            // if the seq_len is 1 just do it with the parts
            let cache_len = k.dims4().unwrap().2;
            if cache_len <= MAX_WINDOW_SIZE + MAX_SINK_SIZE {
                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                att
            } else {
                let k = special_handle(&k, cache_len);
                let att_part = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                att_part
            }
        } else {
            // the first part is ok
            assert_eq!(index_pos, 0, "index_pos should be 0");
            let q_first_part = q.narrow(2, 0, MAX_WINDOW_SIZE + MAX_SINK_SIZE).unwrap();
            let mut att_first_part =
                (q_first_part.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            // println!("finished first part: {:?}", att_first_part.dims());
            for i in (MAX_WINDOW_SIZE + MAX_SINK_SIZE)..seq_len {
                let q_part = q.narrow(2, i, 1)?;
                let k = special_handle(&k, i + 1);
                let att_part = (q_part.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                att_first_part = Tensor::cat(&[att_first_part, att_part], 2)?;
            }
            att_first_part
        };
        // println!("end att");
        let att_len = att.dims4().unwrap().2;
        assert_eq!(att_len, seq_len, "att_len should be seq_len");

        let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;

        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        println!("finished att block {}", block_idx);
        Ok(y)
    }
    /// for those q > MAX_SINK_SIZE + MAX_WINDOW_SIZE, we should handle this one by one.
    #[allow(unused)]
    fn handle_q_one_by_one(
        &self,
        kvs: &KVStore,
        block_idx: usize,
        index_pos: usize,
        q_pos: usize,
        b_sz: usize,
        q: &Tensor,
        hidden_size: usize,
    ) -> Result<Tensor> {
        debug!("handle_q_one_by_one: index_pos: {index_pos}, q_pos:{q_pos}");
        let (cache_k, cache_v) = kvs.get_tensor_at(block_idx, index_pos);
        let reshaped_k = cache_k
            .reshape((
                b_sz,
                MAX_SINK_SIZE + MAX_WINDOW_SIZE,
                self.num_key_value_heads,
                self.head_dim,
            ))
            .expect(&format!("cannot reshape:{index_pos}"))
            .transpose(1, 2)?;
        let reshaped_v = cache_v
            .reshape((
                b_sz,
                MAX_SINK_SIZE + MAX_WINDOW_SIZE,
                self.num_key_value_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?;
        let reshaped_q = q
            .narrow(1, q_pos, 1)?
            .reshape((b_sz, 1, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let rotated_k = self.apply_rotary_emb(&reshaped_k, 0)?;
        // it should always be the max
        let rotated_q = self.apply_rotary_emb(&reshaped_q, MAX_SINK_SIZE + MAX_WINDOW_SIZE - 1)?;
        debug!("computing one by one: index_pos: {index_pos}, q_pos:{q_pos}");
        let y = self.compute_attention(rotated_k, reshaped_v, rotated_q, 1, b_sz, hidden_size)?;
        Ok(y)
    }

    fn compute_attention(
        &self,
        k: Tensor,
        v: Tensor,
        q: Tensor,
        seq_len: usize,
        b_sz: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        debug!("compute_attention: seq_len: {seq_len}");
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;
        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;

            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    fn load(vb: VarBuilder, cache: &StreamCache, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            cache: cache.clone(),
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
        })
    }
}
struct MsbOnly;
impl CustomOp1 for MsbOnly {
    fn name(&self) -> &'static str {
        "MSBOnly"
    }

    fn cpu_fwd(
        &self,
        storage: &candle::CpuStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        let mut data = storage.clone();
        use rayon::prelude::*;
        match data {
            CpuStorage::F32(ref mut data) => {
                data.par_iter_mut().for_each(|d| {
                    let bits = d.to_bits();
                    let masked = bits & 0xff80_0000;
                    *d = f32::from_bits(masked);
                });
            }
            _ => unimplemented!(),
        }
        Ok((data, layout.shape().clone()))
    }
}
#[allow(unused)]
fn msb_only(x: Tensor) -> Tensor {
    x.apply_op1(MsbOnly).unwrap()
}

/// return the tensor where 1-MAX_SINK_SIZE is full, and -MAX_SINK_SIZE..-1 is the MSB
fn special_handle(k: &Tensor, part_length: usize) -> Tensor {
    assert!(part_length > MAX_SINK_SIZE + MAX_WINDOW_SIZE);
    return k.clone();
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        sink_size: usize,
        window_size: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, index_pos, block_idx, sink_size, window_size)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cache: &StreamCache, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    sink_size: usize,
    window_size: usize,
}

impl Llama {
    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        info!("running forward of tensor: {:?}", x);
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, self.sink_size, self.window_size)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(
        vb: VarBuilder,
        cache: &StreamCache,
        cfg: &Config,
        sink_size: usize,
        window_size: usize,
    ) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cache, cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            sink_size,
            window_size,
        })
    }
}

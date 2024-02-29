//! in this version, same as sink, but there are 2 different
//! 1. we store the tensor after ROPE
//! 2. for tensors in sink, we use the full tensor, for others, we use the MSB

use super::with_tracing::{linear_no_bias as linear, Linear};
use candle::{CpuStorage, CustomOp1, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tracing::{debug, info};
pub static ENABLE_SAVE: AtomicBool = AtomicBool::new(false);
pub static SAVE_PATH: RwLock<String> = RwLock::new(String::new());

/// changes here: allow super long sequences
pub const MAX_SEQ_LEN: usize = 4096;
pub static MAX_WINDOW_SIZE: RwLock<usize> = RwLock::new(1024);
pub static MAX_SINK_SIZE: RwLock<usize> = RwLock::new(4);

pub static EVICT_THRESHOLD: RwLock<f32> = RwLock::new(0.05);
pub static RESTORE_THRESHOLD: RwLock<f32> = RwLock::new(0.1);
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
    /// k, k_msb, v, v_msb
    sink_kv: Vec<Option<(Tensor, Tensor, Tensor, Tensor)>>,
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
}
#[derive(Clone, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct MsbIndexItem {
    pub seq: usize,
}

pub type MsbItems = (usize, BTreeSet<MsbIndexItem>);
#[derive(Clone)]
pub struct StreamCache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<KVStore>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    // to implement dynamic MSB, we need a global mask, the dim is[blocks, headers, seq_len] if it's true, we use the full tensor, else we use the MSB
    // # need to implement
    // 1. apply the mask
    // 2. update the mask when it's out of the window
    // 3. resotre the mask if it's attention is large
    // pub global_mask: Arc<Mutex<Vec<Vec<[bool; MAX_SEQ_LEN]>>>>,
    pub current_msb_index: Arc<Mutex<MsbItems>>,
    pub current_mask: Arc<Mutex<(Tensor, Tensor)>>,
}

impl StreamCache {
    /// return batch,header,part_len,header_dim
    pub fn get_current_mask(
        &self,
        block_idx: usize,
        header_dim: usize,
        part_length: usize,
    ) -> (Tensor, Tensor) {
        // info!("start get current mask");
        // if block_idx
        let current_msb_index = self.current_msb_index.lock().unwrap();
        // only update the mask at block 0, else use the cached one
        if block_idx == 0 {
            // generate new mask

            let part_origin_k_mask: Vec<_> = (0..part_length)
                .into_iter()
                .map(|part_idx| {
                    if current_msb_index
                        .1
                        .contains(&MsbIndexItem { seq: part_idx })
                    {
                        vec![0.; header_dim]
                    } else {
                        vec![1.; header_dim]
                    }
                })
                .collect();
            // 32 headers
            let origin_k_mask: Vec<_> = (0..32)
                .into_iter()
                .map(move |_| part_origin_k_mask.clone())
                .collect();
            // for msb
            let part_msb_k_mask: Vec<_> = (0..part_length)
                .into_iter()
                .map(|part_idx| {
                    if current_msb_index
                        .1
                        .contains(&MsbIndexItem { seq: part_idx })
                    {
                        vec![1.; header_dim]
                    } else {
                        vec![0.; header_dim]
                    }
                })
                .collect();
            let msb_k_mask: Vec<_> = (0..32)
                .into_iter()
                .map(move |_| part_msb_k_mask.clone())
                .collect();
            // 32 headers

            let origin_k_mask: Vec<Vec<Vec<Vec<f32>>>> = vec![origin_k_mask];
            let msb_k_mask: Vec<Vec<Vec<Vec<f32>>>> = vec![msb_k_mask];
            let k_mask = Tensor::new(origin_k_mask, &Device::Cpu).unwrap();
            let msb_k_mask = Tensor::new(msb_k_mask, &Device::Cpu).unwrap();
            *self.current_mask.lock().unwrap() = (k_mask, msb_k_mask);
        }
        // info!("end getting mask");
        let result = self.current_mask.lock().unwrap().clone();
        // let dim1 = result.0.dims();
        // let dim2 = result.1.dims();
        // info!("end getting result {dim1:?} {dim2:?}");
        return result;
    }
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
            // global_mask: Arc::new(Mutex::new(vec![
            //     vec![
            //         [true; MAX_SEQ_LEN];
            //         config.num_attention_heads
            //     ];
            //     config.num_hidden_layers
            // ])),
            current_msb_index: Default::default(),
            current_mask: Arc::new(Mutex::new((
                Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
                Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            ))),
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

        *self.current_msb_index.lock().unwrap() = Default::default();
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
        // fix here, if the seq_len is not 1, will do that same as original
        let max_sink_size = *MAX_SINK_SIZE.read().unwrap();
        let max_window_size = *MAX_WINDOW_SIZE.read().unwrap();
        let evict_threshold = *EVICT_THRESHOLD.read().unwrap();
        let restore_threshold = *RESTORE_THRESHOLD.read().unwrap();

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
        let mut cache = self.cache.kvs.lock().unwrap();

        let mut k_msb = msb_only(k.to_dtype(DType::F32).unwrap())
            .to_dtype(DType::F16)
            .unwrap();
        let mut v_msb = msb_only(v.to_dtype(DType::F32).unwrap())
            .to_dtype(DType::F16)
            .unwrap();
        if let Some((cache_k, cache_k_msb, cache_v, cache_v_msb)) = &cache.sink_kv[block_idx] {
            k = Tensor::cat(&[cache_k, &k], 2).unwrap().contiguous()?;
            v = Tensor::cat(&[cache_v, &v], 2).unwrap().contiguous()?;
            k_msb = Tensor::cat(&[cache_k_msb, &k_msb], 2)
                .unwrap()
                .contiguous()?;
            v_msb = Tensor::cat(&[cache_v_msb, &v_msb], 2)
                .unwrap()
                .contiguous()?;
        }

        cache.sink_kv[block_idx] = Some((k.clone(), k_msb.clone(), v.clone(), v_msb.clone()));

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let k_msb = k_msb.to_dtype(DType::F32)?;
        let v_msb = v_msb.to_dtype(DType::F32)?;
        // we cannot do attention in a round, because the window is dynamic, we should simulate the dynamic behavior
        // println!("start att");
        let cache_len = k.dims4().unwrap().2;
        if seq_len != 1 {
            // info!("start the inital prompt block:{}", block_idx);
            // let start = Instant::now();
            // work as original
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            // println!("end att");
            let att_len = att.dims4().unwrap().2;
            assert_eq!(att_len, seq_len, "att_len should be seq_len");

            let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;

            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // after the attention score, we need to update the global mask

            let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
            let y = self.o_proj.forward(&y)?;
            // info!(
            //     "finished att block in batch mode {}, time: {:?}",
            //     block_idx,
            //     start.elapsed()
            // );
            Ok(y)
        } else {
            // for each generation stage, record current cache mask
            // let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            // if the seq_len is 1 just do it with the parts
            // info!("start the step att:{}", block_idx);
            // let start = Instant::now();
            if cache_len <= max_window_size + max_sink_size {
                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                // println!("end att");
                let att_len = att.dims4().unwrap().2;
                assert_eq!(att_len, seq_len, "att_len should be seq_len");

                let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;

                let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
                let att = candle_nn::ops::softmax(&att, D::Minus1)?;
                // after the attention score, we need to update the global mask

                let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
                let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
                let y = self.o_proj.forward(&y)?;
                // info!("finished att block {}", block_idx);
                Ok(y)
            } else {
                // let mut global_mask = self.cache.global_mask.lock().unwrap();
                let (full_mask, msb_mask) =
                    self.cache
                        .get_current_mask(block_idx, self.head_dim, cache_len);
                // apply the mask to k and v
                // the dim for k is (1, headers, seq_len, head_dim), the mask is (headers,seq_len)
                // let time = start.elapsed();
                // info!("finished shpecial handle: {time:?}");
                let full_part = k.mul(&full_mask).unwrap();
                let msb_part = k_msb.mul(&msb_mask).unwrap();
                let k = full_part.add(&msb_part).unwrap();

                let full_part = v.mul(&full_mask).unwrap();
                let msb_part = v_msb.mul(&msb_mask).unwrap();
                let v = full_part.add(&msb_part).unwrap();

                // let time = start.elapsed();
                // info!("finished QK: {time:?}");

                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;
                let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
                let att = candle_nn::ops::softmax(&att, D::Minus1)?;
                // let time = start.elapsed();
                // info!("finished QKV: {time:?}");
                {
                    // update the global mask
                    Self::update_global_mask(
                        block_idx,
                        // &mut global_mask[block_idx],
                        &mut self.cache.current_msb_index.lock().unwrap(),
                        index_pos,
                        seq_len,
                        &att,
                        max_window_size,
                        max_sink_size,
                        evict_threshold,
                        restore_threshold,
                    );
                }
                // after the attention score, we need to update the global mask
                let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
                let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
                let y = self.o_proj.forward(&y)?;
                // let time = start.elapsed();
                // info!("finshed step 1: {time:?}");
                // info!("finished att block {}", block_idx);
                Ok(y)
            }
        }
    }
    fn update_global_mask(
        block_idx: usize,
        // cache: &mut Vec<[bool; MAX_SEQ_LEN]>,
        current_msb_history: &mut MsbItems,
        index_pos: usize,
        seq_len: usize,
        att: &Tensor,
        max_window_size: usize,
        max_sink_size: usize,
        evict_threshold: f32,
        restore_threshold: f32,
    ) {
        if seq_len != 1 {
            panic!("seq_len should be 1");
        }
        let att_dim = att.dims4().unwrap();
        // info!("the att dim: {:?}", att_dim);
        assert_eq!(
            att_dim.3,
            index_pos + 1,
            "the att dim should be index_pos+1 {att_dim:?}"
        );
        // info!("att:{:?}",att.get_on_dim(1,0).unwrap().to_vec3::<f32>().unwrap());
        let tokenindex_to_be_evicted = index_pos - max_window_size;

        // if it's the frist block, put it into msb_history.
        if block_idx == 0 {
            current_msb_history.1.insert(MsbIndexItem {
                seq: tokenindex_to_be_evicted,
            });
        }
        // first the token that go out of the window should set to use MSB, if the att score for that token is large. we should restore the mask
        if current_msb_history.1.contains(&MsbIndexItem {
            seq: tokenindex_to_be_evicted,
        }) {
            for header in 0..32 {
                let atten_header = att.get_on_dim(1, header).unwrap();
                let score = atten_header
                    .get_on_dim(2, tokenindex_to_be_evicted)
                    .unwrap();
                let score: Vec<f32> = score.flatten_all().unwrap().to_vec1().unwrap();
                assert!(score.len() == 1);
                let score = score[0];
                // info!(
                //     "the score:{}:{}: {:?}",
                //     header, tokenindex_to_be_evicted, score
                // );
                // remove it if it's less than 0.05
                if score >= evict_threshold {
                    // this item should not be removed
                    // if any block any header have large score, remove from the msb history
                    current_msb_history.1.remove(&MsbIndexItem {
                        seq: tokenindex_to_be_evicted,
                    });
                }
            }
        }
        // need to restore the
        // restore the mask if the score is large
        for header in 0..32 {
            let atten_header = att.get_on_dim(1, header).unwrap();
            for out_window_index in (max_sink_size + 1)..tokenindex_to_be_evicted {
                let score = atten_header.get_on_dim(2, out_window_index).unwrap();
                let score: f32 = score.flatten_all().unwrap().to_vec1().unwrap()[0];
                if score > restore_threshold {
                    // if any rector triggerd, remove from msb history
                    current_msb_history.1.remove(&MsbIndexItem {
                        seq: out_window_index,
                    });
                }
            }
        }
        current_msb_history.0 = index_pos;
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
        let mut data: CpuStorage = storage.clone();
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
fn msb_only(x: Tensor) -> Tensor {
    x.apply_op1(MsbOnly).unwrap()
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
        let start = Instant::now();

        let mut x = self.wte.forward(x)?;

        let time = start.elapsed();
        info!("time for wte  {time:?}");
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let start = Instant::now();
            x = block.forward(&x, index_pos, block_idx, self.sink_size, self.window_size)?;
            let time = start.elapsed();
            info!("time for block {block_idx}: {time:?}");
        }
        let start = Instant::now();

        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        let time = start.elapsed();
        info!("time lm_head  {time:?}");
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tensor_get() {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &device).unwrap();
        println!("{:?}", a.get_on_dim(1, 2).unwrap());
        println!("{:?}", a.flatten(0, 1).unwrap().to_vec1::<u32>().unwrap());
    }
}

//! a stream sink version of llama

// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
// use the sink version of llama
use candle_transformers::models::llama_sink_mix_dynamic::{self as model, MsbIndexItem};

use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use itertools::Itertools;
use model::{Llama, LlamaConfig};
use polars::lazy::dsl::col;
use polars::prelude::*;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Mutex;
use tracing::info;

const EOS_TOKEN: &str = "</s>";
// const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The model size to use.
    #[arg(long, default_value = "v2")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}
fn limit_len(i: &Vec<(String, String)>, min: usize, max: usize) -> bool {
    let words = i
        .iter()
        .map(|(_r, c)| 1 + c.split_whitespace().collect_vec().len())
        .sum::<usize>();
    words >= min && words <= max
}
static CURRENT_MSB_HISTORY: Mutex<Option<Vec<(usize, BTreeSet<MsbIndexItem>)>>> =
    Mutex::new(Option::None);
static CURRENT_FILE_PATH: Mutex<Option<PathBuf>> = Mutex::new(Option::None);
fn main() -> Result<()> {
    ctrlc::set_handler(move || {
        info!("ctrl-c received");
        if let Some(history) = &CURRENT_MSB_HISTORY.lock().unwrap().deref() {
            let file_path = CURRENT_FILE_PATH.lock().unwrap();
            let file_path = file_path.as_ref().unwrap();
            info!("saving history into {file_path:?}");
            let writer =
                File::create(Path::new(&file_path).join(format!("cache_mask.bin"))).unwrap();
            bincode::serialize_into(writer, history).unwrap();
        }
        exit(0);
    })
    .expect("unable to set ctrl-c handler");
    use tokenizers::Tokenizer;
    // use tracing_chrome::ChromeLayerBuilder;
    // use tracing_subscriber::prelude::*;
    let args = Args::parse();
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
    info!(
        "run with {}:{}",
        model::MAX_SINK_SIZE.read().unwrap(),
        model::MAX_WINDOW_SIZE.read().unwrap()
    );
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, cache) = {
        let api = Api::new()?;
        let model_id = args.model_id.unwrap_or_else(|| match args.which {
            Which::V1 => "Narsil/amall-7b".to_string(),
            Which::V2 => "meta-llama/Llama-2-7b-chat-hf".to_string(),
            Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0".to_string(),
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        });
        println!("loading the model weights from {model_id}");
        let revision = args.revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filenames = match args.which {
            Which::V1 | Which::V2 | Which::Solar10_7B => {
                candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
            }
            Which::TinyLlama1_1BChat => vec![api.get("model.safetensors")?],
        };
        println!("building the model");
        let cache = model::StreamCache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (
            Llama::load(vb, &cache, &config, 4, 1024)?,
            // Llama::load(vb, &cache, &config)?,
            tokenizer_filename,
            cache,
        )
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
    println!("finished loading the model");
    // ENABLE_SAVE.store(true, std::sync::atomic::Ordering::SeqCst);
    println!("fetching the user prompt");

    for (i, prompt) in fetch_all_user_prompt()
        .unwrap()
        .into_iter()
        .filter(|i| limit_len(i, 1240, 2048))
        .take(20)
        .enumerate()
    {
        [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1)].into_iter().for_each(|(evict, restore)|
         {
            *model::EVICT_THRESHOLD.write().unwrap() = evict;
            *model::RESTORE_THRESHOLD.write().unwrap()=restore;
            let max_sink_size = *model::MAX_SINK_SIZE.read().unwrap();
            let max_window_size = *model::MAX_WINDOW_SIZE.read().unwrap();
            let evict_threshold = evict;
            let restore_threshold = restore;

            let test_folder = format!(
                "tests/llama_7b_chat_1240_logits_sink_mix_dynamic-sink[{}]-hist[{}]-evict[{}]-restor[{}]",
                max_sink_size,
                max_window_size,
                evict_threshold,
                restore_threshold,
            );

            // set path
            println!("\n\n-----------------------\ngenerating for prompt: {}", i);
            let file_path = format!("{test_folder}/test_{}", i);
            fs::create_dir_all(&file_path).unwrap();
            *CURRENT_FILE_PATH.lock().unwrap()=Some(Path::new(&file_path).to_owned());

            // let mut save_path = SAVE_PATH.write().unwrap();
            // *save_path = file_path.clone();
            // drop(save_path);
            // clear the cache
            let mut real_prompt = build_prompt(&prompt);
            cache.clear_kv_cache();
            let mut tokens = tokenizer
                .encode(real_prompt.as_str(), true)
                .map_err(E::msg).unwrap()
                .get_ids()
                .to_vec();

            println!("starting the inference loop");
            let mut logits_processor =
                LogitsProcessor::new(args.seed, args.temperature, args.top_p);
            let start_gen = std::time::Instant::now();
            let mut index_pos = 0;
            let mut token_generated = 0;
            print!("{real_prompt}");

            // first feed the prompt
            // first feed 0..MAX_SINK_SIZE + MAX_WINDOW_SIZE
            let ctxt = &tokens[0..(max_sink_size + max_window_size).min(tokens.len() - 1)];
            let input = Tensor::new(ctxt, &device).unwrap().unsqueeze(0).unwrap();
            let _logits = llama.forward(&input, 0).unwrap();

            index_pos += ctxt.len();
            let update_save_to_file = |i:usize,index_pos:usize| {
                let cache_mask = cache.current_msb_index.lock().unwrap();
                    let cache_mask = cache_mask.deref();
                    let mut current_msb_history = CURRENT_MSB_HISTORY.lock().unwrap();
                    let  history = current_msb_history.get_or_insert(vec![]);
                    history.push(cache_mask.clone());

                    let writer = File::create(
                        Path::new(&file_path).join(format!("cache_mask-{}-{}.bin", i, index_pos)),
                    )
                    .unwrap();
                    bincode::serialize_into(writer, cache_mask).unwrap();
            };
            // then feed the rest one by one
            for i in (max_sink_size + max_window_size)..(tokens.len() - 1) {
                // one token from the prompt
                let ctxt = &tokens[i..(i + 1)];
                let input = Tensor::new(ctxt, &device).unwrap().unsqueeze(0).unwrap();

                // save current cache mask
                {
                    update_save_to_file(i,index_pos);
                }

                let _logits = llama.forward(&input, index_pos).unwrap();
                index_pos += ctxt.len();
                info!("{}/{} tokens in prompt finished", i, (tokens.len() - 1 - 1));
            }

            // then feed the rest
            for _index in 0..100 {
                // save current cache mask
                {
                    update_save_to_file(tokens.len() - 1,index_pos);
                }
                // always fetch the last token inserted
                let ctxt = &tokens[(tokens.len() - 1)..];
                let input = Tensor::new(ctxt, &device).unwrap().unsqueeze(0).unwrap();
                let logits = llama.forward(&input, index_pos).unwrap();
                let logits = logits.squeeze(0).unwrap();
                // println!("logits shape: {:.unwrap()}", logits.shape());
                // println!("logits: {:.unwrap()}", logits.to_vec1::<f32>());
                let logits = if args.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        args.repeat_penalty,
                        &tokens[start_at..],
                    ).unwrap()
                };
                index_pos += ctxt.len();

                let next_token = logits_processor.sample(&logits).unwrap();
                token_generated += 1;
                tokens.push(next_token);

                // Extracting the last token as a string is complicated, here we just apply some simple
                // heuristics as it seems to work well enough for this example. See the following for more
                // details:
                // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
                if let Some(text) = tokenizer.id_to_token(next_token) {
                    let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                    real_prompt.extend(text.chars());

                    print!("{text}");
                    std::io::stdout().flush().unwrap();
                }
                if Some(next_token) == eos_token_id {
                    break;
                }
            }
            let dt = start_gen.elapsed();
            println!(
                "\n\n{} tokens generated ({} token/s)\n",
                token_generated,
                token_generated as f64 / dt.as_secs_f64(),
            );
            // save the generated text
            let cache = cache.global_mask.lock().unwrap();
            let blocks = cache.len();
            let headers = cache[0].len();
            let num_false = cache
                .iter()
                .flatten()
                .flatten()
                .map(|x| if *x { 0 } else { 1 })
                .sum::<usize>();
            let total_element = index_pos * blocks * headers;

            // fix bug here, each config should write to different file
            let mut f = File::create(
                Path::new(&file_path).join(format!("1-prompts-{}-{}.txt", evict, restore)),
            )
            .unwrap();
            let mut element = File::create(
                Path::new(&file_path).join(format!("1-ele-{}-{}.txt", evict, restore)),
            )
            .unwrap();
            f.write_all(real_prompt.as_bytes()).unwrap();
            element
                .write_all(
                    format!(
                        "total: {}, num_false: {}, ratio: {}\n",
                        total_element,
                        num_false,
                        (num_false as f32 / total_element as f32)
                    )
                    .as_bytes(),
                )
                .unwrap();



            }
        ) ;
    }

    Ok(())
}

fn build_prompt(prompt: &Vec<(String, String)>) -> String {
    let len = prompt.len();
    assert!(len >= 2);
    let all_but_last = prompt
        .iter()
        .take(len - 1)
        .map(|(role, data)| format!("{}: {}", role, data))
        .join("\n");
    format!("{}\n {} :", all_but_last, prompt[len - 1].0)
}

pub fn fetch_all_user_prompt() -> anyhow::Result<Vec<Vec<(String, String)>>> {
    let file_names = [
        // "train-00002-of-00006-1779b7cec9462180.parquet",
        // "train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
        // "train-00003-of-00006-2fa862bfed56af1f.parquet",
        "train-00000-of-00006-4feeb3f83346a0e9.parquet",
        // "train-00004-of-00006-18f4bdd50c103e71.parquet",
        // "train-00001-of-00006-4030672591c2f478.parquet",
    ];
    let path = Path::new("./dataset/d1");
    let test_data = path.join(file_names[0]);
    let df = ParquetReader::new(File::open(test_data)?).finish()?;
    let mask = df.column("language")?.str()?.equal("English");
    let c = df.filter(&mask)?;
    let c = c["conversation"].list()?;
    let all_c: Vec<_> = c
        .into_iter()
        .take(1000)
        .map(|s| {
            // s is a single conversation, might have multiple turns, each turn is role: text
            let s = s.unwrap();
            let talk = s.struct_().unwrap();
            let all_talks: Vec<_> = talk
                .into_iter()
                .map(|t| {
                    // each t is a trun, 0 is text, 1 is role
                    let talk_data = &t[0];
                    let talk_role = &t[1];
                    let talk_data = match talk_data {
                        AnyValue::String(s) => s,
                        _ => panic!(""),
                    };
                    let talk_role = match talk_role {
                        AnyValue::String(s) => s,
                        _ => panic!(""),
                    };
                    // println!("{:?} : {:?}", talk_role, talk_data,);
                    (talk_role.to_string(), talk_data.to_string())
                })
                .collect();
            assert!(all_talks.len() >= 1);
            assert_eq!(all_talks[0].0, "user");
            all_talks
            // println!("{:?}", talk);
        })
        .collect();
    Ok(all_c)
}
pub fn fetch_all_user_prompt_lazy() -> anyhow::Result<Vec<String>> {
    let file_names = [
        "train-00002-of-00006-1779b7cec9462180.parquet",
        "train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
        "train-00003-of-00006-2fa862bfed56af1f.parquet",
        "train-00000-of-00006-4feeb3f83346a0e9.parquet",
        "train-00004-of-00006-18f4bdd50c103e71.parquet",
        "train-00001-of-00006-4030672591c2f478.parquet",
    ];
    let path = Path::new("../dataset/");
    let df = LazyFrame::scan_parquet_files(
        file_names
            .iter()
            .map(|n| path.join(n))
            .collect::<Vec<_>>()
            .into(),
        Default::default(),
    )?;
    let c = df.filter(col("language").eq(lit("English")));
    let c = c.select([col("conversation")]);
    let c = c.limit(1000).fetch(2000)?;

    let c = c["conversation"].list().unwrap();
    let all_c: Vec<_> = c
        .into_iter()
        .take(1000)
        .map(|s| {
            // s is a single conversation, might have multiple turns, each turn is role: text
            let s = s.unwrap();
            let talk = s.struct_().unwrap();
            let all_talks: Vec<_> = talk
                .into_iter()
                .take(1)
                .map(|t| {
                    // each t is a trun, 0 is text, 1 is role
                    let talk_data = &t[0];
                    let talk_role = &t[1];
                    let (talk_data, talk_role) = match (talk_data, talk_role) {
                        (AnyValue::String(s), AnyValue::String(r)) => (s, r),
                        _ => panic!(""),
                    };
                    // println!("{:?} : {:?}", talk_role, talk_data,);
                    (talk_role.to_string(), talk_data.to_string())
                })
                .collect();
            assert_eq!(all_talks.len(), 1);
            assert_eq!(all_talks[0].0, "user");
            all_talks[0].1.clone()
            // println!("{:?}", talk);
        })
        .collect();
    Ok(all_c)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use super::*;
    use itertools::Itertools;
    use polars::{
        io::{parquet::ParquetReader, SerReader},
        lazy::frame::LazyFrame,
    };

    #[test]
    #[ignore]
    fn test_load() {
        let file_names = [
            "train-00002-of-00006-1779b7cec9462180.parquet",
            "train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
            "train-00003-of-00006-2fa862bfed56af1f.parquet",
            "train-00000-of-00006-4feeb3f83346a0e9.parquet",
            "train-00004-of-00006-18f4bdd50c103e71.parquet",
            "train-00001-of-00006-4030672591c2f478.parquet",
        ];
        let df = LazyFrame::scan_parquet_files(
            file_names
                .iter()
                .map(|n| Path::new("../dataset").join(n))
                .collect::<Vec<_>>()
                .into_boxed_slice()
                .into(),
            Default::default(),
        )
        .unwrap();

        println!("{:?}", df.collect().unwrap().get_column_names());
    }

    #[test]
    #[ignore]
    fn test_data_type() {
        let file_names = [
            "train-00002-of-00006-1779b7cec9462180.parquet",
            "train-00005-of-00006-fe1acc5d10a9f0e2.parquet",
            "train-00003-of-00006-2fa862bfed56af1f.parquet",
            "train-00000-of-00006-4feeb3f83346a0e9.parquet",
            "train-00004-of-00006-18f4bdd50c103e71.parquet",
            "train-00001-of-00006-4030672591c2f478.parquet",
        ];
        let path = Path::new("../dataset/");
        let test_data = path.join(file_names[0]);
        let df = ParquetReader::new(File::open(test_data).unwrap())
            .finish()
            .unwrap();
        let _cols = [
            "conversation_id",
            "model",
            "conversation",
            "turn",
            "language",
            "openai_moderation",
            "redacted",
        ];
        let c = df["conversation"].list().unwrap();
        c.into_iter().take(1).for_each(|s| {
            let s = s.unwrap();
            let talk = s.struct_().unwrap();
            talk.into_iter().for_each(|t| {
                let talk_data = &t[0];
                let talk_role = &t[1];
                let talk_data = match talk_data {
                    AnyValue::String(s) => s,
                    _ => panic!(""),
                };
                let talk_row = match talk_role {
                    AnyValue::String(s) => s,
                    _ => panic!(""),
                };
                println!("{:?} : {:?}", talk_row, talk_data,);
            });
            // println!("{:?}", talk);
        });
    }

    #[test]
    #[ignore]
    fn test_fetch() {
        let now = std::time::Instant::now();
        let result = fetch_all_user_prompt_lazy().unwrap();
        println!("lazy time:{:?}", now.elapsed());
        println!("{:?}", result.len());

        let now = std::time::Instant::now();

        let result = fetch_all_user_prompt().unwrap();
        println!("eager time:{:?}", now.elapsed());
        println!("{:?}", result.len());
    }

    #[test]
    #[ignore]
    fn test_chat_bot() -> anyhow::Result<()> {
        let data_path = "../dataset/chatbot/train-00000-of-00001-cced8514c7ed782a.parquet";
        let df = LazyFrame::scan_parquet(Path::new(data_path), Default::default()).unwrap();
        let df = df
            .select(&[col("conversation_a"), col("conversation_b")])
            .collect()
            .unwrap();
        let c = df["conversation_a"].list().unwrap();
        let all_cov_a = fetch_all_cov(c);
        let all_cov_b = fetch_all_cov(df["conversation_b"].list().unwrap());
        let iterleaved = all_cov_a
            .interleave(all_cov_b)
            .filter(|i| limit_len(i, 1240, 2048))
            .collect_vec();

        println!("{:?}", iterleaved.len());
        for p in iterleaved.into_iter().take(10) {
            println!("{:?}", p);
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_build_prompt() {
        let data_path = "../dataset/chatbot/train-00000-of-00001-cced8514c7ed782a.parquet";
        let df = LazyFrame::scan_parquet(Path::new(data_path), Default::default()).unwrap();
        let df = df
            .select(&[col("conversation_a"), col("conversation_b")])
            .collect()
            .unwrap();
        let c = df["conversation_a"].list().unwrap();
        let all_cov_a = fetch_all_cov(c);
        let all_cov_b = fetch_all_cov(df["conversation_b"].list().unwrap());
        let iterleaved = all_cov_a
            .interleave(all_cov_b)
            .filter(|i| limit_len(i, 1240, 2048))
            .collect_vec();
        for p in iterleaved.into_iter().take(10) {
            let prompt = build_prompt(&p);
            println!("{:?}", prompt);
        }
    }

    fn fetch_all_cov(
        c: &ChunkedArray<ListType>,
    ) -> impl Iterator<Item = Vec<(String, String)>> + '_ {
        let all_cov = c.into_iter().take(10).map(|s| {
            let s = s.unwrap();
            let talk = s.struct_().unwrap();
            let cov = talk
                .into_iter()
                .map(move |t| {
                    let talk_data = &t[0];
                    let talk_role = &t[1];
                    let talk_data = match talk_data {
                        AnyValue::String(s) => s.to_string(),
                        _ => panic!(""),
                    };
                    let talk_role = match talk_role {
                        AnyValue::String(s) => s.to_string(),
                        _ => panic!(""),
                    };
                    (talk_role, talk_data)
                })
                .collect::<Vec<_>>();
            cov
            // println!("{:?}", talk);
        });
        all_cov
    }
}

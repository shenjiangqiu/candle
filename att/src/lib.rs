use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    iter::Sum,
    ops::Add,
};

use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

use candle_transformers::models::llama_sink_mix_dynamic::MsbIndexItem;

pub fn init_logger() {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
pub fn init_logger_debug() {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::DEBUG.into())
                .from_env_lossy(),
        )
        .init();
}
/// ones , zeros, recalculate, others
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MySum(pub usize, pub usize, pub usize, pub usize);
impl Add for MySum {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        MySum(
            self.0 + rhs.0,
            self.1 + rhs.1,
            self.2 + rhs.2,
            self.3 + rhs.3,
        )
    }
}
impl Sum for MySum {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(MySum(0, 0, 0, 0), |a, b| a + b)
    }
}

impl From<(usize, usize, usize, usize)> for MySum {
    fn from((a, b, c, d): (usize, usize, usize, usize)) -> Self {
        MySum(a, b, c, d)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub evict: f32,
    pub restore: f32,
    pub cache_remain_time: usize,
    pub sink_size: usize,
    pub window_size: usize,
}
#[derive(Serialize, Deserialize)]
pub struct DynStat {
    pub total_restores: usize,
    pub total_evicts: usize,
    pub total_tests: usize,
    pub current_cache: VecDeque<BTreeSet<MsbIndexItem>>,
    pub hit_histogram: [usize; 32],
    pub miss_histogram: BTreeMap<usize, usize>,
    pub evict_num_history: Vec<(usize, usize)>,
    pub restore_count_histogram: BTreeMap<String, usize>,
    pub restore_seq_count_histogram: BTreeMap<String, usize>,
    pub restore_block_count_histogram: [usize; 32],
}

#[derive(Serialize, Deserialize)]
pub struct Stat {
    pub config: Config,
    pub dyn_state: DynStat,
}
impl Stat {
    pub fn new(
        evict: f32,
        restore: f32,
        cache_size: usize,
        sink_size: usize,
        window_size: usize,
    ) -> Self {
        Self {
            config: Config {
                evict,
                restore,
                cache_remain_time: cache_size,
                sink_size,
                window_size,
            },
            dyn_state: DynStat {
                total_restores: 0,
                total_evicts: 0,
                total_tests: 0,
                current_cache: Default::default(),
                hit_histogram: [0; 32],
                miss_histogram: Default::default(),
                // more
                evict_num_history: Default::default(),
                restore_count_histogram: Default::default(),
                restore_seq_count_histogram: Default::default(),
                restore_block_count_histogram: Default::default(),
            },
        }
    }
}

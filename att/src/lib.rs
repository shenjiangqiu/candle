use std::{iter::Sum, ops::Add};

use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

pub fn init_logger() {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
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

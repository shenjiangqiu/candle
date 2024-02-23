use std::{fs::File, io::BufReader};

use att::Stat;

fn main() {
    for cache_size in [4, 6, 8] {
        for (evict, restore) in [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1)] {
            let result_name = format!("result-{evict}-{restore}-{cache_size}.json");
            let reader = BufReader::new(File::open(&result_name).unwrap());
            let stat: Stat = serde_json::from_reader(reader).unwrap();
            let time_before_cache: usize = compute_time_before_cache(&stat);
            let time_after_cache: usize = compute_time_after_cache(&stat);
            println!("evict: {evict}, restore: {restore}, time_before_cache: {time_before_cache}, time_after_cache: {time_after_cache}");
        }
    }
}
fn compute_time_before_cache(stat: &Stat) -> usize {
    todo!()
}
fn compute_time_after_cache(stat: &Stat) -> usize {
    todo!()
}

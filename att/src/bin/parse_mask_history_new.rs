use std::{
    collections::{BTreeSet, VecDeque},
    fs::File,
    io::{BufReader, BufWriter},
    ops::AddAssign,
    path::Path,
};

use att::Stat;
use candle_transformers::models::llama_sink_mix_dynamic::{CurrentRoundEvictRestore, MsbIndexItem};

use itertools::Itertools;
fn main() {
    use rayon::prelude::*;
    let prompt_idx = 0;
    [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1)].into_par_iter().for_each(|(evict, restore)|
        {
            [(4, 1024), (4, 1), (40, 1024), (40, 1)].into_par_iter().for_each(|(max_sink_size, max_window_size)| {
                let evict_threshold = evict;
                let restore_threshold = restore;

                let _test_folder = format!(
                    "tests/0223llama_7b_chat_1240_logits_sink_mix_dynamic-sink[{}]-hist[{}]-evict[{}]-restor[{}]",
                    max_sink_size,
                    max_window_size,
                    evict_threshold,
                    restore_threshold,
                );
                // set path
                println!("\n\n-----------------------\ngenerating for prompt: {}", prompt_idx);
                let _file_path = format!("{_test_folder}/test_{}", prompt_idx);
                let _file_path = Path::new(&_file_path);
                let history_path = _file_path.join("history.bin");
                // let prompt_path = _file_path.join(format!("1-prompts-{}-{}.txt", evict, restore));
                let history_items:Vec<CurrentRoundEvictRestore> = bincode::deserialize_from(BufReader::new(File::open(&history_path).unwrap())).unwrap();
                [4, 6, 8].into_iter().for_each(|cache_size| {

                    let mut stat = Stat::new(evict, restore, cache_size,max_sink_size,max_window_size);
                    println!("start");
                    for (item, index_pos) in history_items.iter().zip(max_sink_size..) {
                        update_result(&mut stat,  &item, index_pos);
                    }
                    let result_name = format!("result-{evict}-{restore}-{cache_size}.json");
                    let writer = BufWriter::new(File::create(&result_name).unwrap());
                    serde_json::to_writer_pretty(writer, &stat).unwrap();
                });
        });
    });
}

fn update_result(stat: &mut Stat, current: &CurrentRoundEvictRestore, step: usize) {
    let evict_index = step - stat.config.window_size - 1;
    // no evict and stores yet
    if evict_index < stat.config.sink_size {
        println!("evict index: {evict_index}, continue");
        return;
    }

    let cache_size = stat.config.cache_remain_time;

    // update evict
    // println!("step :{step}, previouse: {previouse:?},current: {current:?}");
    let current_evicted = &current.evict;
    let current_restore = &current.restore;
    let current_evicted = current_evicted.len();
    println!("{current_evicted}/1024 items evicted in location: {evict_index}");
    stat.dyn_state.total_tests += 32 * 32;
    stat.dyn_state.total_evicts += current_evicted;
    stat.dyn_state
        .evict_num_history
        .push((evict_index, current_evicted));

    // update restore
    // the restored elements
    // note that the current_restore is already sorted by block id, so do not need to sort it again!!
    // apply the cache block-by-block
    debug_assert!({
        current_restore
            .clone()
            .into_iter()
            .map(|a| a.block_idx)
            .tuple_windows()
            .all(|(a, b)| a <= b)
    });
    // push a new slot into cache
    stat.dyn_state.current_cache.push_back(BTreeSet::new());
    // first count cache coverage
    // current block requests should be merged
    for item in current_restore {
        stat.dyn_state.total_restores += 1;
        let key = format!("seq-{}-gap-{}", item.seq, evict_index - item.seq);
        stat.dyn_state
            .restore_count_histogram
            .entry(key)
            .or_default()
            .add_assign(1);

        stat.dyn_state
            .restore_seq_count_histogram
            .entry(item.seq.to_string())
            .or_default()
            .add_assign(1);

        stat.dyn_state.restore_block_count_histogram[item.block_idx] += 1;

        if let Some(gap) = in_cache(&item, &stat.dyn_state.current_cache) {
            stat.dyn_state.hit_histogram[gap] += 1;
        } else {
            // miss
            let gap_between_added = evict_index - item.seq;
            *stat
                .dyn_state
                .miss_histogram
                .entry(gap_between_added)
                .or_default() += 1;
        }
        stat.dyn_state
            .current_cache
            .back_mut()
            .unwrap()
            .insert(item.clone());
        // get the restore distence
        if stat.dyn_state.current_cache.len() > cache_size {
            // println!("the cache len is {},pop frot", {
            //     stat.dyn_state.current_cache.len()
            // });
            stat.dyn_state.current_cache.pop_front().unwrap();
            // println!("the poped records: {}", p.len());
        }
    }
}

/// return the num of layers between current and cache.
fn in_cache(item: &MsbIndexItem, cache: &VecDeque<BTreeSet<MsbIndexItem>>) -> Option<usize> {
    // todo: make it parallel?
    use rayon::prelude::*;
    let seq = item.seq;
    let block_idx = item.block_idx;
    assert!(block_idx < 32, "{block_idx} is to large");
    let hit_cache = cache
        .par_iter()
        .flatten()
        .filter(|x| x.seq == seq && x.block_idx <= block_idx)
        .max_by_key(|x| x.block_idx);
    hit_cache.map(|hit| block_idx.checked_sub(hit.block_idx).unwrap())
}

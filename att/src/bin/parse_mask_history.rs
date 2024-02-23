use std::{
    collections::{BTreeSet, VecDeque},
    fs::File,
    io::{BufReader, BufWriter},
    ops::AddAssign,
    path::Path,
};

use att::Stat;
use candle_transformers::models::llama_sink_mix_dynamic::MsbIndexItem;
use itertools::Itertools;
type MsbItems = (usize, BTreeSet<MsbIndexItem>);
fn main() {
    use rayon::prelude::*;
    [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1)].into_iter().for_each(|(evict, restore)| {
        let base_name = format!("tests/0223llama_7b_chat_1240_logits_sink_mix_dynamic-sink[4]-hist[1024]-evict[{evict}]-restor[{restore}]/test_0");
            let base_name = Path::new(&base_name);
            let file_name = base_name.join(format!("cache_mask.bin"));
            let file = BufReader::new(
                File::open(&file_name).expect(&format!("fail to open file : {file_name:?}")),
            );
            println!("reading files");
            let msb_items: Vec<MsbItems> = bincode::deserialize_from(file).unwrap();
            [4, 6, 8].into_iter().for_each(|cache_size| {

        let mut stat = Stat::new(evict, restore, cache_size);            
        println!("start");
            let mut previouse = Option::None;

            for (item, index_pos) in msb_items.iter().zip(1028..) {
                update_result(&mut stat, &previouse, &item, index_pos);
                previouse = Some(item.clone());
            }
            let result_name = format!("result-{evict}-{restore}-{cache_size}.json");
            let writer = BufWriter::new(File::create(&result_name).unwrap());
            serde_json::to_writer_pretty(writer, &stat).unwrap();
        });
    });
}

fn update_result(stat: &mut Stat, previouse: &Option<MsbItems>, current: &MsbItems, step: usize) {
    let evict_index = step - 1024 - 1;
    // no evict and stores yet
    if evict_index < 4 {
        println!("evict index: {evict_index}, continue");
        return;
    }

    let cache_size = stat.config.cache_remain_time;

    // update evict
    // println!("step :{step}, previouse: {previouse:?},current: {current:?}");
    let current_evicted = current.1.iter().filter(|x| x.seq == evict_index).count();
    debug_assert_eq!(
        0,
        current
            .1
            .iter()
            .filter(|x| x.seq == evict_index + 1)
            .count(),
        "this location is still in the window, should not be evicted!"
    );

    println!("{current_evicted}/1024 items evicted in location: {evict_index}");
    stat.dyn_state.total_tests += 32 * 32;
    stat.dyn_state.total_evicts += current_evicted;
    stat.dyn_state
        .evict_num_history
        .push((evict_index, current_evicted));

    // update restore
    if let Some(previouse) = previouse {
        // the restored elements
        // note that the current_restore is already sorted by block id, so do not need to sort it again!!
        // apply the cache block-by-block
        let current_restored = previouse.1.difference(&current.1).cloned();
        debug_assert!({
            current_restored
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
        for item in current_restored {
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
                .insert(item);
            // get the restore distence
        }
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

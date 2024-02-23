use std::{fs::File, io::BufReader};

use candle_transformers::models::llama_sink_mix_dynamic_no_header_block::MsbItems;

fn main() {
    let file = BufReader::new(File::open("tests/llama_7b_chat_1240_logits_sink_mix_dynamic-sink[4]-hist[1024]-evict[0.01]-restor[0.02]/test_0/cache_mask-1301-1301.bin").unwrap());
    let msb_items: MsbItems = bincode::deserialize_from(file).unwrap();
    println!("{:?}", msb_items);
}

use std::{
    fs::read_dir,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use att::{init_logger, MySum};
use clap::Parser;
use rayon::prelude::*;

use tracing::{debug, info};
#[derive(Parser)]
struct Cli {
    path: PathBuf,
}

const FULL_CYCLE: usize = 16 * 16;
const MSB_CYCLE: usize = 16;
fn ones_cycle(ones: usize) -> usize {
    // the q plus k plus v
    //
    2 * ones * (4096 / 32) * FULL_CYCLE
}
fn zeros_cycle(zeros: usize) -> usize {
    // the q plus k plus v
    2 * zeros * (4096 / 32) * MSB_CYCLE
}
const SINGLE_FFN: usize = 4096 * 4096 * 4 * 2 * FULL_CYCLE;
const SINGLE_PROJECTION: usize = 4096 * 4096 * 3 * FULL_CYCLE;
// the recalculate cylce:(block_id, seq_id)
fn recalculate_cycle(recalculate: &[(usize, usize)]) -> usize {
    // assume have no information,just recalculate all
    recalculate
        .into_iter()
        .map(|(block, seq)| {
            let num_project_att = *block;
            let num_ffn = num_project_att - 1;
            let single_project_att = SINGLE_PROJECTION + 2 * 4096 * FULL_CYCLE * seq;
            let full_project_att = single_project_att * num_project_att;
            let full_ffn = SINGLE_FFN * num_ffn;
            let total = full_project_att + full_ffn;
            total
        })
        .sum()
}
fn other_cycle() -> usize {
    // other include the projecton and ffn, this is for a single token, single block, single header
    // so it should be amotized to 32 heads
    let other = SINGLE_FFN + SINGLE_PROJECTION;
    other / 32
}
fn calculate_total_cycle(
    ones: usize,
    zeros: usize,
    recalculate: &[(usize, usize)],
) -> (usize, usize, usize, usize) {
    (
        ones_cycle(ones),
        zeros_cycle(zeros),
        recalculate_cycle(recalculate),
        other_cycle(),
    )
}

fn main() {
    init_logger();
    let cli = Cli::parse();
    info!("Parsing history file: {:?}", cli.path);
    let files = read_dir(cli.path).unwrap();
    let first_number_regex = regex::Regex::new(r"^.*-(\d+)-").unwrap();

    let mut all_bins: Vec<_> = files
        .filter_map(|f| f.ok())
        .map(|f| f.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
        .map(|p| {
            let file_name = p.file_name().unwrap().to_str().unwrap();
            let captures = first_number_regex.captures(file_name).unwrap();
            let n = captures.get(1).unwrap().as_str().parse::<u32>().unwrap();
            (n, p)
        })
        .collect();
    info!("Found {} bin files", all_bins.len());
    all_bins.sort_by_key(|(n, _)| *n);
    //sort by the number in the filename

    let mut last_round: Option<Vec<Vec<Vec<bool>>>> = Option::None;

    // println!("{:?}", all_bins);
    let all_len = all_bins.len();
    let cycle: Vec<_> = all_bins
        .iter()
        .enumerate()
        .map(|(i, (n, p))| {
            info!("Processing {}/{}: {:?}", i, all_len, p);
            let reader = std::fs::File::open(p).unwrap();
            let reader = BufReader::new(reader);
            let data: Vec<Vec<Vec<bool>>> = bincode::deserialize_from(reader).unwrap();
            let sum = data
                .par_iter()
                .enumerate()
                .map(|(block_id, block)| {
                    // the mask for last roun
                    block
                        .into_par_iter()
                        .enumerate()
                        .map(|(header_id, header)| {
                            let span = tracing::debug_span!("block", block_id, header_id);
                            let _enter = span.enter();
                            let out_slice = &header[4..(*n as usize - 1023)];
                            // let out = formate_bools(out_slice);
                            // println!("{block_id:02}-{header_id:02}:{}", out);
                            let total_ones = out_slice.iter().filter(|b| **b).count();
                            let total_zeros = out_slice.len() - total_ones;
                            let total_ones = total_ones + 4 + 1023;
                            let total_recalculate = if let Some(last_round) = &last_round {
                                let last_round = &last_round[block_id][header_id];
                                check_last_round(&last_round[4..(*n as usize - 1024)], out_slice)
                            } else {
                                vec![]
                            };
                            let total_recalculate: Vec<_> = total_recalculate
                                .into_iter()
                                .map(|i| (block_id, i + 4))
                                .collect();
                            calculate_total_cycle(total_ones, total_zeros, &total_recalculate)
                                .into()
                        })
                        .sum::<MySum>()
                })
                .sum::<MySum>();
            last_round = Some(data);
            (i, sum)
        })
        .collect();
    let file = std::fs::File::create("cycle.json").unwrap();
    let file = BufWriter::new(file);
    serde_json::to_writer(file, &cycle).unwrap();
}
fn check_last_round(last_round: &[bool], header: &[bool]) -> Vec<usize> {
    let mut total_recalculate = vec![];
    for (_i, (last, current)) in last_round.iter().zip(header.iter()).enumerate() {
        // when last if false and current is true, means the value is recalculated
        if (*last == false) && (*last != *current) {
            debug!("{} is recalculated", _i);
            // let inserted = recalcualte.insert((block_id, header_id, _i));
            // assert!(inserted);
            total_recalculate.push(_i)
        }
    }
    total_recalculate
}

use att::init_logger;
use bit_set::BitSet;
use candle_transformers::models::llama::{AttRecord, SingleLocationRecord, SingleTimeAttRecord};
use clap::Parser;
use serde::Serialize;
use std::{
    collections::{BTreeSet, VecDeque},
    fmt::Debug,
    io::BufReader,
    path::{Path, PathBuf},
};
use tracing::{debug, info};

#[derive(Parser)]
struct Cli {
    pub file_dir: PathBuf,
    pub report_dir: PathBuf,
}

fn run_analyzer_and_save_report<A: Analyzer>(analyzer: A, record: &Vec<Vec<SingleLocationRecord>>, report_path: &Path) {
    let report_file_path = report_path.join(format!("report_{}.json", analyzer.name()));
    let report = run_analyzer(analyzer, &record);
    let writer = std::fs::File::create(report_file_path).unwrap();
    let writer = std::io::BufWriter::new(writer);
    serde_json::to_writer_pretty(writer, &report).unwrap();
}
#[allow(unused_macros)]
macro_rules! run_all {
    ($($analyzer:expr),* ;$record:expr;$path:expr; $func:ident) => {{
        $(
            $func($analyzer,&$record,&$path);
        )*
    }};
}
fn main() {
    init_logger();
    let cli = Cli::parse();
    let report_dir = cli.report_dir;
    (0..20).into_iter().for_each(|i| {
        let report_path = report_dir.join(format!("test_{}", i));
        std::fs::create_dir_all(&report_path).unwrap();
        info!("parsing {}", i);
        let file_path = cli.file_dir.join(format!("test_{}", i)).join("att_history.bin");
        let reader = std::fs::File::open(&file_path).unwrap();
        let reader = BufReader::new(reader);
        let record: AttRecord = bincode::deserialize_from(reader).unwrap();
        info!("analyzing {}", file_path.display());
        let record = record.all_records;
        let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        thread_pool.scope(|scope| {
            let report_path = &report_path;
            let record = &record;
            for (sink, window) in [(4, 1024), (4, 1)] {
                for (evict, restore) in [(0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 0.16)] {
                    for (ave, cur, rst) in [
                        (Some(evict), Some(evict), Some(restore)),
                        (Some(evict), None, Some(restore)),
                        (None, Some(evict), Some(restore)),
                        (Some(evict), Some(evict), None),
                        (Some(evict), None, None),
                        (None, Some(evict), None),
                    ] {
                        for cache_size in [4, 8, 16] {
                            scope.spawn(move |_| {
                                run_analyzer_and_save_report(
                                    AverageWindowAnalyzer::new(window, sink, ave, cur, rst, cache_size),
                                    &record,
                                    &report_path,
                                );
                            });
                        }
                    }
                }
            }
        });
    });
}
fn run_analyzer<A: Analyzer>(analyzer: A, record: &Vec<Vec<SingleLocationRecord>>) -> A::Report {
    analyzer.analyze(record)
}
pub trait Analyzer {
    type Report: Serialize + Debug;
    fn name(&self) -> String;
    fn analyze(self, record: &Vec<Vec<SingleLocationRecord>>) -> Self::Report;
}

struct TestAnalyzer;
impl Analyzer for TestAnalyzer {
    type Report = String;
    fn name(&self) -> String {
        "TestAnalyzer".to_string()
    }
    fn analyze(self, record: &Vec<Vec<SingleLocationRecord>>) -> Self::Report {
        let records = record[0][0].time_line.len();
        format!(" records: {records}")
    }
}

#[derive(Serialize, Debug, Default)]
struct AverageWindowReport {
    block_headers: Vec<Vec<SingleHeadReport>>,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
struct SingleHeadReport {
    tested: usize,
    evicted: usize,
    restored: usize,
    restored_missed: usize,
    restored_hit: usize,
    total_hit_distance: usize,
}

impl SingleHeadReport {
    pub fn update_evict(&mut self) {
        self.evicted += 1;
    }
    pub fn update_tested(&mut self) {
        self.tested += 1;
    }
    pub fn update_restored(&mut self, hit_distance: Option<usize>) {
        self.restored += 1;
        if let Some(hit_distance) = hit_distance {
            self.restored_hit += 1;
            self.total_hit_distance += hit_distance;
        } else {
            self.restored_missed += 1;
        }
    }
}
struct AverageWindowAnalyzer {
    window_size: usize,
    sink_size: usize,
    ave_evict_threshold: Option<f32>,
    curr_evict_threshold: Option<f32>,
    restore_threshold: Option<f32>,
    cache_size: usize,
}
fn average_att(record: &SingleLocationRecord) -> SingleLocationRecord {
    let mut accumulated =
        record
            .time_line
            .iter()
            .enumerate()
            .fold(SingleLocationRecord::default(), |mut acc, (index_pos, it)| {
                let mut it = it.att[0..=index_pos].to_vec();

                if let Some(last) = acc.time_line.last() {
                    it.iter_mut().zip(last.att.iter()).for_each(|(i, l)| {
                        *i += l;
                    });
                }
                acc.time_line.push(SingleTimeAttRecord { att: it });
                acc
            });
    accumulated
        .time_line
        .iter_mut()
        .enumerate()
        .for_each(|(index_pos, time_record)| {
            time_record
                .att
                .iter_mut()
                .enumerate()
                .take(index_pos + 1)
                .for_each(|(seq_id, att)| {
                    let total_history = index_pos + 1 - seq_id;
                    *att /= total_history as f32;
                });
        });

    accumulated
}

impl AverageWindowAnalyzer {
    pub fn new(
        window_size: usize,
        sink_size: usize,
        ave_evict_threshold: Option<f32>,
        curr_evict_threshold: Option<f32>,
        restore_threshold: Option<f32>,
        cache_size: usize,
    ) -> Self {
        Self {
            window_size,
            sink_size,
            ave_evict_threshold,
            curr_evict_threshold,
            restore_threshold,
            cache_size,
        }
    }
    // the input is the original att record, the output is the average att record across timeline

    fn analysis_single_head(
        &self,
        block_id: usize,
        head_id: usize,
        seq_id: usize,
        current_record: &SingleLocationRecord,
        average_record: &SingleLocationRecord,
        current_restore_cache: &mut VecDeque<BTreeSet<(usize, usize, usize)>>,
        report: &mut SingleHeadReport,
        current_kv_msb: &mut BitSet,
    ) {
        debug!(
            "analyzing block_id: {}, head_id: {}, seq_id: {}",
            block_id, head_id, seq_id
        );
        debug!("current_record: {:?}", current_record);
        debug!("average_record: {:?}", average_record);
        debug!("current_kv_msb: {:?}", current_kv_msb);
        let sink_size = self.sink_size;
        let window_size = self.window_size;
        assert!(seq_id >= sink_size + window_size);
        let ave_evict_threshold = self.ave_evict_threshold;
        let curr_evict_threshold = self.curr_evict_threshold;
        let restore_threshold = self.restore_threshold;

        let average_att = &average_record.time_line[seq_id].att;
        let current_att = &current_record.time_line[seq_id].att;

        let idx_to_evict = seq_id - window_size;
        let average_score = average_att[idx_to_evict];
        let current_score = current_att[idx_to_evict];

        if !(ave_evict_threshold.is_some_and(|th| average_score > th)
            || curr_evict_threshold.is_some_and(|th| current_score > th))
        {
            // evict it
            current_kv_msb.insert(idx_to_evict);
            report.update_evict();
        }
        report.update_tested();

        // restore
        if let Some(restore_threshold) = restore_threshold {
            for restore_idx in sink_size..idx_to_evict {
                if current_kv_msb.contains(restore_idx) {
                    let restore_score = current_att[restore_idx];

                    if restore_score > restore_threshold {
                        // remove from the current kv msb
                        current_kv_msb.remove(restore_idx);
                        // need to restore, test the cache first
                        let hit_block_id_max = current_restore_cache
                            .iter()
                            .flatten()
                            .filter_map(
                                |(block_id, _, seq_id)| {
                                    if *seq_id == restore_idx {
                                        Some(block_id)
                                    } else {
                                        None
                                    }
                                },
                            )
                            .max();
                        let hit_distance =
                            hit_block_id_max.map(|hit_id| if block_id <= *hit_id { 0 } else { block_id - *hit_id });
                        report.update_restored(hit_distance);
                        // restore it!, save it to cache
                        current_restore_cache
                            .back_mut()
                            .unwrap()
                            .insert((block_id, head_id, restore_idx));
                    }
                }
            }
        }
    }
}
use itertools::*;
impl Analyzer for AverageWindowAnalyzer {
    type Report = AverageWindowReport;
    fn name(&self) -> String {
        format!(
            "AverageWindowAnalyzer-wd@{}-sk@{}-cache@{}-aveth@{:?}-currth@{:?}-rest@{:?}",
            self.window_size,
            self.sink_size,
            self.cache_size,
            self.ave_evict_threshold,
            self.curr_evict_threshold,
            self.restore_threshold,
        )
    }
    fn analyze(self, record: &Vec<Vec<SingleLocationRecord>>) -> Self::Report {
        let mut current_restore_cache = VecDeque::new();
        let seq_len = record[0][0].time_line.len();
        let mut all_reports = vec![vec![SingleHeadReport::default(); 32]; 32];
        let mut all_current_kv_msb = vec![vec![BitSet::with_capacity(4096); 32]; 32];

        // translate record into average_record
        let ave_record = record
            .iter()
            .map(|it| it.iter().map(|it| average_att(it)).collect_vec())
            .collect_vec();
        debug!("ave_record: {:?}", ave_record);
        let start_sim_seq_id: usize = self.sink_size + self.window_size;
        for seq_id in start_sim_seq_id..seq_len {
            current_restore_cache.push_back(Default::default());
            for (block_id, (record, ave_record, block_report, block_kv_msb)) in izip!(
                record.iter(),
                ave_record.iter(),
                all_reports.iter_mut(),
                all_current_kv_msb.iter_mut()
            )
            .enumerate()
            {
                for (head_id, (record, ave_record, header_report, header_kv_msb)) in
                    izip!(record, ave_record, block_report.iter_mut(), block_kv_msb.iter_mut()).enumerate()
                {
                    self.analysis_single_head(
                        block_id,
                        head_id,
                        seq_id,
                        record,
                        ave_record,
                        &mut current_restore_cache,
                        header_report,
                        header_kv_msb,
                    );
                }
            }
            if current_restore_cache.len() > self.cache_size {
                current_restore_cache.pop_front();
            }
        }

        AverageWindowReport {
            block_headers: all_reports,
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use att::init_logger_debug;

    use super::*;
    #[test]
    fn test_average_att() {
        let record = SingleLocationRecord {
            time_line: vec![
                SingleTimeAttRecord {
                    att: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                },
                SingleTimeAttRecord {
                    att: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                },
                SingleTimeAttRecord {
                    att: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                },
                SingleTimeAttRecord {
                    att: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                },
                SingleTimeAttRecord {
                    att: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                },
            ],
        };
        let result = average_att(&record);
        assert_eq!(result.time_line[0].att, vec![1.0,]);
        assert_eq!(result.time_line[1].att, vec![1.0, 2.0,]);
        assert_eq!(result.time_line[2].att, vec![1.0, 2.0, 3.]);
        assert_eq!(result.time_line[3].att, vec![1.0, 2.0, 3.0, 4.]);
        assert_eq!(result.time_line[4].att, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_analyzer() {
        // init_logger_debug();
        let analyzer = AverageWindowAnalyzer::new(2, 2, Some(0.1), Some(0.1), Some(0.2), 4);
        let record = vec![
            vec![
                SingleLocationRecord {
                    time_line: vec![
                        vec![0.05,].into(),
                        vec![0.05, 0.05].into(),
                        vec![0.05, 0.05, 0.05].into(),
                        vec![0.05, 0.05, 0.05, 0.05].into(),
                        vec![0.05, 0.05, 0.05, 0.05, 0.35].into(), // evit here
                        vec![0.05, 0.05, 0.05, 0.05, 0.35, 0.05].into(),
                        vec![0.05, 0.05, 0.25, 0.25, 0.05, 0.05, 0.05].into(), // restore here. and keep here
                    ],
                };
                32
            ];
            32
        ];
        let report = analyzer.analyze(&record);

        // the first case, miss the two 0.25
        let first_case = SingleHeadReport {
            tested: 3,
            evicted: 2,
            restored: 2,
            restored_missed: 2,
            restored_hit: 0,
            total_hit_distance: 0,
        };
        // the second case, restore and hit to the first head with distance 0
        let second_case = SingleHeadReport {
            tested: 3,
            evicted: 2,
            restored: 2,
            restored_missed: 0,
            restored_hit: 2,
            total_hit_distance: 0,
        };
        // the third case, the first head of other blocks, restore hit to previouse block with distance 1
        let third_case = SingleHeadReport {
            tested: 3,
            evicted: 2,
            restored: 2,
            restored_missed: 0,
            restored_hit: 2,
            total_hit_distance: 2,
        };
        
        assert_eq!(report.block_headers[0][0], first_case);
        report.block_headers[0].iter().skip(1).for_each(|item| {
            assert_eq!(item, &second_case);
        });

        report.block_headers.iter().skip(1).for_each(|item| {
            assert_eq!(&item[0], &third_case);
            item.iter().skip(1).for_each(|item| {
                assert_eq!(item, &second_case);
            });
        });
    }
}

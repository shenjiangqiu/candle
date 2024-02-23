use std::{collections::BTreeMap, fs, io::BufWriter, path::PathBuf};

use candle::*;
use clap::Parser;
use itertools::Itertools;
use regex::Regex;
#[derive(Parser)]
struct Cli {
    path: PathBuf,
}
fn main() {
    // (token, block) -> tensor
    let cli = Cli::parse();
    let path = cli.path;
    let mut tensors = BTreeMap::new();
    let files_in_path = fs::read_dir(&path).unwrap();
    let names = files_in_path.map(|f| f.unwrap().file_name().to_string_lossy().to_string());
    let regex = Regex::new(r"^att-(\d+)-(\d+)$").unwrap();
    let min_index = names
        .filter(|name| regex.is_match(name))
        .map(|name| {
            let captures = regex.captures(&name).unwrap();
            let token = captures.get(1).unwrap().as_str().parse::<usize>().unwrap();
            token
        })
        .filter(|x| *x != 0)
        .minmax();
    let (min, max) = min_index.into_option().unwrap();
    println!("min: {}, max: {}", min, max);
    println!("Loading tensors...");
    for block in 0..32 {
        for token in min..=max {
            println!("Loading token: {}, block: {}", token, block);
            let tensor_name = format!("att-{}-{}", token, block);
            let file = path.join(&tensor_name);
            let tensor = safetensors::load(&file, &Device::Cpu)
                .expect(format!("Failed to load {:?}", &file).as_str());
            let tensor = tensor.get(&tensor_name).unwrap();
            tensors.insert((token, block), tensor.clone());
        }
    }
    let mut block_header_tensor_map = BTreeMap::new();
    for block in 0..32 {
        for header in 0..32 {
            println!("Writing token: {}, block: {}", header, block);

            let mut all_tensors = vec![];
            for token in min..=max {
                let tensor = tensors.get(&(token, block)).unwrap();
                let t = tensor.get_on_dim(1, header).unwrap();
                let t = t.flatten_all().unwrap();
                let t: Vec<f32> = t.to_vec1().unwrap();
                all_tensors.push(t);
            }
            block_header_tensor_map.insert(format!("{}-{}", block, header), all_tensors);
        }
    }
    let result_file = format!("att-result.txt");
    let result_path = path.join(&result_file);
    let result_file = std::fs::File::create(result_path).unwrap();
    let result_file_buffer = BufWriter::new(result_file);
    serde_json::to_writer_pretty(result_file_buffer, &block_header_tensor_map).unwrap();
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, time::Instant};

    use candle::{Device, Tensor};

    #[test]
    fn test() {
        let d = Device::Cpu;
        let t = Tensor::new(&[[[2., 3.], [5., 6.]], [[2., 3.], [5., 6.]]], &d).unwrap();
        let t = t.narrow(1, 1, 1).unwrap();
        let t = Tensor::cat(&[&t, &t], 1).unwrap();
        println!("{:?}", t.layout());
    }

    #[test]
    fn test_broadcast() {
        // let value = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap();
        // let mask: Tensor = Tensor::new(&[1., 0.], &Device::Cpu).unwrap();
    }

    #[test]
    fn test_perf() {
        let now = Instant::now();
        for _i in 0..100000000 {
            let a = black_box(0.1);
            let b = black_box(10.);
            let c = a * b;
            black_box(c);
        }
        let elapsed = now.elapsed();
        println!("time: {}", elapsed.as_millis());

        let now = Instant::now();
        for _i in 0..100000000 {
            let a = black_box(1);
            let b = black_box(1);
            let c = a + b;
            black_box(c);
        }
        let elapsed = now.elapsed();
        println!("time: {}", elapsed.as_millis());
    }
}

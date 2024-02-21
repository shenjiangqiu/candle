use att::MySum;

fn main() {
    let cycles: Vec<(usize, MySum)> =
        serde_json::from_reader(std::fs::File::open("cycle.json").unwrap()).unwrap();
    println!("index: ones, zeros, recalculate, others");
    for (index_pos, cycle) in cycles {
        let MySum(ones, zeros, recalculate, others) = cycle;
        println!(
            "{} {} {} {} {}",
            index_pos, ones, zeros, recalculate, others
        );
    }
}

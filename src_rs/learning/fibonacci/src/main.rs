fn main() {
    let num = 1;
    println!("Calculating");
    println!("{}th fibonacci number is {}", num, nth_fibonacci(num));
}

fn nth_fibonacci(in_idx: u32) -> u64 {
    if in_idx == 0 || in_idx == 1 {
        return 1;
    }

    let mut count_minus_1: u64 = 1;
    let mut count_minus_2: u64 = 1;

    for _ in 2..in_idx + 1 {
        let count: u64 = count_minus_1 + count_minus_2;
        count_minus_2 = count_minus_1;
        count_minus_1 = count;
    }
    return count_minus_1;
}

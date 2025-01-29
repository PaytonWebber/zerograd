use std::time::Instant;
use tensor::Tensor;

fn main() {
    let size = 1024;
    let a = Tensor::ones(vec![size, size]);
    let b = Tensor::ones(vec![size, size]);

    let start = Instant::now();
    let _ = a.matmul(&b);
    let duration = start.elapsed();
    println!("Matrix multiplication took: {:?}", duration);
}

use tensor::{Tensor, TensorF32, TensorF64};

fn main() {
    // Test f32 tensors
    let a = TensorF32::ones(vec![2, 2]);
    let b = TensorF32::zeros(vec![2, 2]);
    println!("f32 tensor:\n{}", &a + &b);

    // Test f64 tensors
    let a_f64 = TensorF64::ones(vec![2, 2]);
    let b_f64 = TensorF64::zeros(vec![2, 2]);
    println!("f64 tensor:\n{}", &a_f64 + &b_f64);

    // Test with explicit generic syntax
    let a_explicit: Tensor<f32> = Tensor::ones(vec![2, 2]);
    let b_explicit: Tensor<f32> = Tensor::new(vec![2, 2], vec![2.5, 3.0, 1.5, 4.0]).unwrap();
    println!("Explicit f32 tensor:\n{}", &a_explicit + &b_explicit);
}

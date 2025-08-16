pub mod core;

pub use core::{Numeric, Tensor};

pub type TensorF32 = Tensor<f32>;
pub type TensorF64 = Tensor<f64>;

use super::broadcast::{compute_broadcast_shape_and_strides, is_broadcastable};
use crate::core::{unravel_index, TensorError};
use crate::Tensor;

impl Tensor {
    fn binary_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();
        if !is_broadcastable(self_shape, other_shape) {
            return Err(TensorError::BroadcastError(
                "Shapes are not compatible for the operation".to_string(),
            ));
        }

        if self_shape == other_shape {
            let result_data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| op(*a, *b))
                .collect();
            return Tensor::new(self_shape.clone(), result_data);
        }

        let (bc_shape, self_bc_strides, other_bc_strides) =
            compute_broadcast_shape_and_strides(self_shape, other_shape);

        let self_data = self.data();
        let other_data = other.data();
        let result_size: usize = bc_shape.iter().product();
        let mut result_data: Vec<f32> = Vec::with_capacity(result_size);

        for i in 0..result_size {
            let multi_idx = unravel_index(i, &bc_shape);
            let mut self_offset = 0;
            let mut other_offset = 0;

            for (dim_i, &stride) in self_bc_strides.iter().enumerate() {
                self_offset += multi_idx[dim_i] * stride;
            }
            for (dim_i, &stride) in other_bc_strides.iter().enumerate() {
                other_offset += multi_idx[dim_i] * stride;
            }
            result_data.push(op(self_data[self_offset], other_data[other_offset]));
        }
        Tensor::new(bc_shape, result_data)
    }

    fn binary_op_inplace<F>(&mut self, other: &Tensor, op: F)
    where
        F: Fn(&mut f32, f32),
    {
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape != other_shape {
            panic!("Shapes not compatible for in-place operation");
        }

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, &b)| {
                op(a, b);
            });
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.binary_op(other, |a, b| a + b)
    }

    pub fn add_inplace(&mut self, other: &Tensor) {
        self.binary_op_inplace(other, |a, b| *a += b);
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.binary_op(other, |a, b| a - b)
    }

    pub fn sub_inplace(&mut self, other: &Tensor) {
        self.binary_op_inplace(other, |a, b| *a -= b);
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.binary_op(other, |a, b| a * b)
    }

    pub fn mul_inplace(&mut self, other: &Tensor) {
        self.binary_op_inplace(other, |a, b| *a *= b);
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.binary_op(other, |a, b| a / b)
    }

    pub fn div_inplace(&mut self, other: &Tensor) {
        self.binary_op_inplace(other, |a, b| *a /= b);
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let lhs_shape: &Vec<usize> = self.shape();
        let rhs_shape: &Vec<usize> = other.shape();
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            return Err(TensorError::BroadcastError(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let (rows_left, cols_left) = (lhs_shape[0], lhs_shape[1]);
        let (rows_right, cols_right) = (rhs_shape[0], rhs_shape[1]);
        if cols_left != rows_right {
            return Err(TensorError::BroadcastError(
                "Incompatible shapes for matrix multiplication".to_string(),
            ));
        }

        let lhs_data: &Vec<f32> = self.data();
        let rhs_data: &Vec<f32> = other.data();
        let mut result_data: Vec<f32> = vec![0.0_f32; rows_left * cols_right];
        for i in 0..rows_left {
            for k in 0..cols_left {
                for j in 0..cols_right {
                    result_data[i * cols_right + j] +=
                        lhs_data[i * cols_left + k] * rhs_data[k * cols_right + j];
                }
            }
        }
        Ok(Tensor::new(vec![rows_left, cols_right], result_data).unwrap())
    }
}

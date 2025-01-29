use crate::core::{unravel_index, TensorError};
use crate::Tensor;

impl Tensor {
    pub fn sum(&self) -> Tensor {
        let sum: f32 = self.data().iter().sum();
        Tensor::new(vec![1], vec![sum]).unwrap()
    }

    pub fn sum_dim(&self, dim: usize) -> Result<Tensor, TensorError> {
        let self_data = self.data();
        let self_shape = self.shape();
        let self_strides = self.strides();
        if self_shape.len() < dim {
            return Err(TensorError::IndexError(
                "Dimension out of range for the tensor".to_string(),
            ));
        }

        let mut result_shape = self_shape.clone();
        let dim_size = result_shape.remove(dim);

        let result_size: usize = result_shape.iter().product();
        let mut result_data = vec![0.0_f32; result_size];

        for i in 0..result_size {
            let result_multi_idx = unravel_index(i, &result_shape);
            let mut full_multi_idx: Vec<usize> = Vec::with_capacity(self_shape.len());
            let mut j = 0;
            for k in 0..self_shape.len() {
                if k == dim {
                    full_multi_idx.push(0);
                } else {
                    full_multi_idx.push(result_multi_idx[j]);
                    j += 1;
                }
            }
            let mut sum = 0.0_f32;
            for k in 0..dim_size {
                full_multi_idx[dim] = k;
                let mut offset = 0;
                for (dim_i, &stride) in self_strides.iter().enumerate() {
                    offset += full_multi_idx[dim_i] * stride;
                }
                sum += self_data[offset];
            }
            result_data[i] = sum;
        }
        Tensor::new(result_shape, result_data)
    }

    pub fn mean(&self) -> Tensor {
        let sum: Tensor = self.sum();
        &sum / self.shape().iter().product::<usize>() as f32
    }

    pub fn mean_dim(&self, dim: usize) -> Result<Tensor, TensorError> {
        if self.shape().len() < dim {
            return Err(TensorError::IndexError(
                "Dimension out of range for the tensor".to_string(),
            ));
        }
        let sum: Tensor = self.sum_dim(dim).unwrap();
        Ok(&sum / self.shape()[dim] as f32)
    }
}

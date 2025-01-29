mod errors;
mod ops;
mod traits;
mod utils;

use errors::TensorError;
use utils::{calculate_strides, unravel_index};

pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let length: usize = shape.iter().product();
        if data.len() != length {
            return Err(TensorError::CreationError(
                "Data does not fit within shape".to_string(),
            ));
        }
        let strides: Vec<usize> = calculate_strides(&shape);
        Ok(Tensor {
            shape: shape.to_vec(),
            strides,
            data,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = calculate_strides(&shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![0.0; num_elements],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = calculate_strides(&shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![1.0; num_elements],
        }
    }

    pub fn get(&self, indices: Vec<usize>) -> Option<&f32> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut idx: usize = 0;
        for (i, &dim) in indices.iter().enumerate() {
            if dim >= self.shape[i] {
                return None;
            }
            idx += dim * self.strides[i];
        }
        self.data.get(idx)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn strides(&self) -> &Vec<usize> {
        &self.strides
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }
}

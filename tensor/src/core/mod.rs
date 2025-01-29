mod errors;
mod ops;
mod traits;
mod utils;

use errors::TensorError;
use utils::calculate_strides;

pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new<S>(shape: S, data: Vec<f32>) -> Result<Self, TensorError>
    where
        S: Into<Vec<usize>>,
    {
        let shape_vec = shape.into();
        let length: usize = shape_vec.iter().product();
        if data.len() != length {
            return Err(TensorError::CreationError(
                "Data does not fit within shape".to_string(),
            ));
        }
        let strides: Vec<usize> = calculate_strides(&shape_vec);
        Ok(Tensor {
            shape: shape_vec,
            strides,
            data,
        })
    }

    pub fn zeros<S>(shape: S) -> Self
    where
        S: Into<Vec<usize>>,
    {
        let shape_vec = shape.into();
        let num_elements: usize = shape_vec.iter().product();
        let strides: Vec<usize> = calculate_strides(&shape_vec);
        Tensor {
            shape: shape_vec,
            strides,
            data: vec![0.0; num_elements],
        }
    }

    pub fn ones<S>(shape: S) -> Self
    where
        S: Into<Vec<usize>>,
    {
        let shape_vec = shape.into();
        let num_elements: usize = shape_vec.iter().product();
        let strides: Vec<usize> = calculate_strides(&shape_vec);
        Tensor {
            shape: shape_vec,
            strides,
            data: vec![1.0; num_elements],
        }
    }

    pub fn get(&self, indices: &[usize]) -> Option<&f32> {
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

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }
}

use crate::core::TensorError;
use crate::Tensor;

impl Tensor {
    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<(), TensorError> {
        let new_length: usize = shape.iter().product();
        let current_length: usize = self.shape.iter().product();
        if new_length != current_length {
            return Err(TensorError::CreationError(
                "The new shape does not align with the size of the data.".to_string(),
            ));
        }
        self.strides = Self::calculate_strides(&shape);
        self.shape = shape.to_vec();
        Ok(())
    }

    pub fn permute(&mut self, order: Vec<usize>) -> Result<(), TensorError> {
        if order.len() != self.shape.len() {
            return Err(TensorError::CreationError(
                "The permutation does not align with the current shape.".to_string(),
            ));
        }

        let mut sorted_order: Vec<usize> = order.to_vec();
        sorted_order.sort();
        if sorted_order != (0..self.shape.len()).collect::<Vec<_>>() {
            return Err(TensorError::CreationError(
                "Index out of range for shape.".to_string(),
            ));
        }

        let new_shape: Vec<usize> = order.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<usize> = order.iter().map(|&i| self.strides[i]).collect();

        self.shape = new_shape;
        self.strides = new_strides;
        Ok(())
    }

    pub fn flatten(&mut self) {
        self.shape = vec![self.shape.iter().product()];
        self.strides = vec![1];
    }

    pub fn transpose(&mut self) -> Result<(), TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::CreationError(
                "transpose only supports 2D tensors currently.".to_string(),
            ));
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let mut new_data = vec![0.0_f32; self.data.len()];
        for i in 0..m {
            for j in 0..n {
                let old_idx = i * n + j;
                let new_idx = j * m + i;
                new_data[new_idx] = self.data[old_idx];
            }
        }
        self.data = new_data;
        self.shape = vec![n, m];
        self.strides = vec![m, 1];
        Ok(())
    }
}

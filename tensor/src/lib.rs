use std::ops::{Add, Index};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self, &'static str> {
        let length: usize = shape.iter().product();
        if data.len() != length {
            return Err("Data does not fit within shape");
        }
        let strides: Vec<usize> = Self::calculate_strides(shape);
        Ok(Tensor {
            shape: shape.to_vec(),
            strides,
            data,
        })
    }

    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let length: usize = shape.len();
        let mut strides = vec![1; length];
        strides.iter_mut().enumerate().for_each(|(i, stride)| {
            // stride[i] = (shape[i+1]*shape[i+2]*...*shape[N-1])
            *stride = shape.iter().take(length).skip(i + 1).product();
        });
        strides
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = Self::calculate_strides(shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![0.0; num_elements],
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = Self::calculate_strides(shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![1.0; num_elements],
        }
    }

    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), &'static str> {
        let new_length: usize = shape.iter().product();
        let current_length: usize = self.shape.iter().product();
        if new_length != current_length {
            return Err("The new shape does not align with the size of the data.");
        }
        self.strides = Self::calculate_strides(shape);
        self.shape = shape.to_vec();
        Ok(())
    }

    pub fn permute(&mut self, order: &[usize]) -> Result<(), &'static str> {
        if order.len() != self.shape.len() {
            return Err("The permutation does not align with the current shape.");
        }

        let mut sorted_order: Vec<usize> = order.to_vec();
        sorted_order.sort();
        if sorted_order != (0..self.shape.len()).collect::<Vec<_>>() {
            return Err("Index out of range for shape.");
        }

        let new_shape: Vec<usize> = order.iter().map(|i| self.shape[*i]).collect();
        self.reshape(&new_shape)
    }

    pub fn flatten(&mut self) {
        self.shape = vec![self.shape.iter().product()];
        self.strides = vec![1];
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

    fn get(&self, indices: &[usize]) -> Option<&f32> {
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

    pub fn add(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.shape != other.shape {
            return Err("Shapes of the tensors do not match for addition.");
        }

        let result_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::new(&self.shape, result_data)
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;
    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get(indices).expect("Index out of bounds")
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        match Tensor::add(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for addition."),
        }
    }
}

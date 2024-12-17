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
        self.shape = shape.to_vec();
        Ok(())
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
}

pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self, &'static str> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err("Data length does not match shape");
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            data,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: vec![0.0; num_elements],
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let num_elements: usize = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: vec![1.0; num_elements],
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }
}

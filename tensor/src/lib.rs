use core::{f32, fmt};
use std::{
    ops::{Add, Index},
    usize,
};

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

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let shape_a: Vec<usize> = self.shape().to_vec();
        let shape_b: Vec<usize> = other.shape().to_vec();

        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err("matmul currently only supports 2D tensors.");
        }

        let m: usize = shape_a[0];
        let n: usize = shape_a[1];
        let n_b: usize = shape_b[0]; // should be equal to n
        let p: usize = shape_b[1];

        if n != n_b {
            return Err("Dimension mismatch for matmul: A.cols must equal B.rows");
        }

        let mut c: Tensor = Tensor::zeros(&[m, p]);
        let a_data = self.data();
        let b_data = other.data();
        let c_data = c.data_mut();

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0_f32;
                for k in 0..n {
                    // Compute the indices:
                    let a_val = a_data[i * n + k];
                    let b_val = b_data[k * p + j];
                    sum += a_val * b_val;
                }
                c_data[i * p + j] = sum;
            }
        }

        Ok(c)
    }
}

fn calculate_data_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| idx * strides[i])
        .sum()
}

fn print_tensor_recursive(
    f: &mut fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    current_index: &mut [usize],
    dim: usize,
    ndims: usize,
) -> fmt::Result {
    if ndims == 0 {
        // 0-D tensor (scalar)
        if let Some(value) = data.first() {
            return write!(f, "{:.4}", value);
        } else {
            return write!(f, "");
        }
    }

    if dim == ndims - 1 {
        // Last dimension: print elements in a row
        write!(f, "[")?;
        for i in 0..shape[dim] {
            current_index[dim] = i;
            let idx = calculate_data_index(current_index, strides);
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", data[idx])?;
        }
        write!(f, "]")?;
    } else {
        // Not the last dimension
        write!(f, "[")?;
        for i in 0..shape[dim] {
            current_index[dim] = i;

            if i > 0 {
                // Subsequent slices/rows
                if dim == 0 {
                    // Top-level dimension
                    if ndims >= 3 {
                        // For 3D or more: blank line between top-level slices
                        write!(f, "\n\n")?;
                        // 7 spaces indentation
                        for _ in 0..7 {
                            write!(f, " ")?;
                        }
                    } else if ndims == 2 {
                        // For 2D: no blank line, just newline + 8 spaces
                        writeln!(f)?;
                        for _ in 0..8 {
                            write!(f, " ")?;
                        }
                    }
                } else {
                    // Inner dimension (dim > 0)
                    // newline + 8 spaces
                    writeln!(f)?;
                    for _ in 0..8 {
                        write!(f, " ")?;
                    }
                }
            }

            print_tensor_recursive(f, data, shape, strides, current_index, dim + 1, ndims)?;
        }
        write!(f, "]")?;
    }

    Ok(())
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.is_empty() {
            // 0-D tensor
            if let Some(value) = self.data.first() {
                return write!(f, "tensor({:.4})", value);
            } else {
                return write!(f, "tensor()");
            }
        }

        write!(f, "tensor(")?;
        let mut current_index = vec![0; self.shape.len()];
        print_tensor_recursive(
            f,
            &self.data,
            &self.shape,
            &self.strides,
            &mut current_index,
            0,
            self.shape.len(),
        )?;
        write!(f, ")")?;
        Ok(())
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

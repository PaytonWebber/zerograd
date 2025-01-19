use core::{f32, fmt};
use std::ops::{Add, Div, Index, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, &'static str> {
        let length: usize = shape.iter().product();
        if data.len() != length {
            return Err("Data does not fit within shape");
        }
        let strides: Vec<usize> = Self::calculate_strides(&shape);
        Ok(Tensor {
            shape: shape.to_vec(),
            strides,
            data,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = Self::calculate_strides(&shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![0.0; num_elements],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let strides: Vec<usize> = Self::calculate_strides(&shape);
        Tensor {
            shape: shape.to_vec(),
            strides,
            data: vec![1.0; num_elements],
        }
    }

    fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
        let length: usize = shape.len();
        let mut strides = vec![1; length];
        strides.iter_mut().enumerate().for_each(|(i, stride)| {
            // stride[i] = (shape[i+1]*shape[i+2]*...*shape[N-1])
            *stride = shape.iter().take(length).skip(i + 1).product();
        });
        strides
    }

    /* MOVEMENT OPS */

    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<(), &'static str> {
        let new_length: usize = shape.iter().product();
        let current_length: usize = self.shape.iter().product();
        if new_length != current_length {
            return Err("The new shape does not align with the size of the data.");
        }
        self.strides = Self::calculate_strides(&shape);
        self.shape = shape.to_vec();
        Ok(())
    }

    pub fn permute(&mut self, order: Vec<usize>) -> Result<(), &'static str> {
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

    pub fn transpose(&mut self) -> Result<(), &'static str> {
        if self.shape.len() != 2 {
            return Err("transpose only supports 2D tensors currently.");
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

    /* REDUCTION OPS */

    pub fn sum(&self) -> Tensor {
        let sum: f32 = self.data().iter().sum();
        Tensor::new(vec![1], vec![sum]).unwrap()
    }

    pub fn sum_dim(&self, dim: usize) -> Result<Tensor, &'static str> {
        let self_data = self.data();
        let self_shape = self.shape();
        let self_strides = self.strides();
        if self_shape.len() < dim {
            return Err("Dimension out of range for the tensor");
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

    pub fn mean_dim(&self, dim: usize) -> Result<Tensor, &'static str> {
        if self.shape().len() < dim {
            return Err("Dimension out of range for the tensor");
        }
        let sum: Tensor = self.sum_dim(dim).unwrap();
        Ok(&sum / self.shape()[dim] as f32)
    }

    /* UNARY OPS */

    pub fn exp(&self) -> Tensor {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x.exp()).collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }

    pub fn log(&self) -> Tensor {
        let result_data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| {
                if x == 0.0 {
                    f32::NEG_INFINITY // log(0) -> -inf
                } else if x < 0.0 {
                    f32::NAN // log of negative numbers is undefined
                } else {
                    x.ln()
                }
            })
            .collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }

    /* BINARY OPS */

    pub fn add(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        if !is_broadcastable(self_shape, other_shape) {
            return Err("The tensor shapes are not compatible for addition.");
        }

        if self_shape == other_shape {
            let result_data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
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
            for (dim_i, &stride) in self_bc_strides.iter().enumerate() {
                self_offset += multi_idx[dim_i] * stride;
            }

            let mut other_offset = 0;
            for (dim_i, &stride) in other_bc_strides.iter().enumerate() {
                other_offset += multi_idx[dim_i] * stride;
            }

            let val = self_data[self_offset] + other_data[other_offset];
            result_data.push(val);
        }
        Tensor::new(bc_shape, result_data)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        if !is_broadcastable(self_shape, other_shape) {
            return Err("The tensor shapes are not compatible for subtraction.");
        }

        if self_shape == other_shape {
            let result_data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
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
            for (dim_i, &stride) in self_bc_strides.iter().enumerate() {
                self_offset += multi_idx[dim_i] * stride;
            }

            let mut other_offset = 0;
            for (dim_i, &stride) in other_bc_strides.iter().enumerate() {
                other_offset += multi_idx[dim_i] * stride;
            }

            let val = self_data[self_offset] - other_data[other_offset];
            result_data.push(val);
        }
        Tensor::new(bc_shape, result_data)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        if !is_broadcastable(self_shape, other_shape) {
            return Err("The tensor shapes are not compatible for multiplication.");
        }

        if self_shape == other_shape {
            let result_data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(self_shape.clone(), result_data);
        }

        let (bc_shape, self_bc_strides, other_bc_strides) =
            compute_broadcast_shape_and_strides(self_shape, other_shape);

        let result_size = bc_shape.iter().product();
        let mut result_data: Vec<f32> = Vec::with_capacity(result_size);
        let self_data = self.data();
        let other_data = other.data();

        for i in 0..result_size {
            let multi_idx = unravel_index(i, &bc_shape);

            let mut self_offset = 0;
            for (dim_i, &stride) in self_bc_strides.iter().enumerate() {
                self_offset += multi_idx[dim_i] * stride;
            }

            let mut other_offset = 0;
            for (dim_i, &stride) in other_bc_strides.iter().enumerate() {
                other_offset += multi_idx[dim_i] * stride;
            }

            let val = self_data[self_offset] * other_data[other_offset];
            result_data.push(val);
        }
        Tensor::new(bc_shape, result_data)
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let self_shape = self.shape();
        let other_shape = other.shape();
        if !is_broadcastable(self_shape, other_shape) {
            return Err("The tensor shapes are not compatible for division.");
        }

        if self_shape == other_shape {
            let result_data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a / b)
                .collect();
            return Tensor::new(self_shape.clone(), result_data);
        }

        let (bc_shape, self_bc_strides, other_bc_strides) =
            compute_broadcast_shape_and_strides(self_shape, other_shape);

        let result_size = bc_shape.iter().product();
        let mut result_data: Vec<f32> = Vec::with_capacity(result_size);
        let self_data = self.data();
        let other_data = other.data();

        for i in 0..result_size {
            let multi_idx = unravel_index(i, &bc_shape);

            let mut self_offset = 0;
            for (dim_i, &stride) in self_bc_strides.iter().enumerate() {
                self_offset += multi_idx[dim_i] * stride;
            }

            let mut other_offset = 0;
            for (dim_i, &stride) in other_bc_strides.iter().enumerate() {
                other_offset += multi_idx[dim_i] * stride;
            }

            let val = self_data[self_offset] / other_data[other_offset];
            result_data.push(val);
        }
        Tensor::new(bc_shape, result_data)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        let lhs_shape: &Vec<usize> = self.shape();
        let rhs_shape: &Vec<usize> = other.shape();
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            return Err("matmul requires 2D tensors");
        }

        let (rows_left, cols_left) = (lhs_shape[0], lhs_shape[1]);
        let (rows_right, cols_right) = (rhs_shape[0], rhs_shape[1]);
        if cols_left != rows_right {
            return Err("Incompatible shapes for matrix multiplication");
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

    /* GETTERS */

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

fn unravel_index(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0; ndim];
    for j in (0..ndim).rev() {
        let dim_size = shape[j];
        coords[j] = i % dim_size;
        i /= dim_size;
    }
    coords
}

pub fn is_broadcastable(a: &Vec<usize>, b: &Vec<usize>) -> bool {
    // This is based on NumPy's rules: https://numpy.org/doc/stable/user/basics.broadcasting.html
    for (i, j) in a.into_iter().rev().zip(b.into_iter().rev()) {
        if *i == 1 || *j == 1 {
            continue;
        }
        if *i != *j {
            return false;
        }
    }
    true
}

pub fn compute_broadcast_shape_and_strides(
    a_shape: &Vec<usize>,
    b_shape: &Vec<usize>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let ndims = a_shape.len().max(b_shape.len());
    let mut a_bc_strides = vec![1; ndims];
    let mut b_bc_strides = vec![1; ndims];
    let mut bc_shape = vec![0; ndims];
    let mut a_dims = vec![0; ndims];
    let mut b_dims = vec![0; ndims];

    let a_shape_rev: Vec<usize> = a_shape.iter().copied().rev().collect();
    let b_shape_rev: Vec<usize> = b_shape.iter().copied().rev().collect();
    for i in 0..ndims {
        let dim_a = a_shape_rev.get(i).copied().unwrap_or(1);
        let dim_b = b_shape_rev.get(i).copied().unwrap_or(1);
        a_dims[ndims - i - 1] = dim_a;
        b_dims[ndims - i - 1] = dim_b;
        if i == 0 {
            if dim_a != dim_b {
                a_bc_strides[ndims - i - 1] = match dim_a {
                    1 => 0,
                    _ => a_dims[ndims - i..].into_iter().product(),
                };
                b_bc_strides[ndims - i - 1] = match dim_b {
                    1 => 0,
                    _ => b_dims[ndims - i..].into_iter().product(),
                };
            }
        } else {
            if dim_a != dim_b {
                a_bc_strides[ndims - i - 1] = match dim_a {
                    1 => 0,
                    _ => a_dims[ndims - i..].into_iter().product(),
                };
                b_bc_strides[ndims - i - 1] = match dim_b {
                    1 => 0,
                    _ => b_dims[ndims - i..].into_iter().product(),
                };
            } else {
                a_bc_strides[ndims - i - 1] = a_dims[ndims - i..].into_iter().product();
                b_bc_strides[ndims - i - 1] = b_dims[ndims - i..].into_iter().product();
            }
        }
        bc_shape[ndims - i - 1] = dim_a.max(dim_b);
    }
    (bc_shape, a_bc_strides, b_bc_strides)
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
        if let Some(value) = data.first() {
            return write!(f, "{:.4}", value);
        } else {
            return write!(f, "");
        }
    }

    if dim == ndims - 1 {
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
        write!(f, "[")?;
        for i in 0..shape[dim] {
            current_index[dim] = i;

            if i > 0 {
                if dim == 0 {
                    if ndims >= 3 {
                        write!(f, "\n\n")?;
                        for _ in 0..7 {
                            write!(f, " ")?;
                        }
                    } else if ndims == 2 {
                        writeln!(f)?;
                        for _ in 0..8 {
                            write!(f, " ")?;
                        }
                    }
                } else {
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

/* TRAIT IMPLEMENTATIONS */

impl Index<Vec<usize>> for Tensor {
    type Output = f32;
    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        self.get(indices).expect("Index out of bounds")
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        match Tensor::add(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for addition."),
        }
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x + rhs).collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let result_data: Vec<f32> = rhs.data().iter().map(|&x| x + self).collect();
        Tensor::new(rhs.shape().clone(), result_data).unwrap()
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        match Tensor::sub(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for subtraction."),
        }
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x - rhs).collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let result_data: Vec<f32> = rhs.data().iter().map(|&x| x - self).collect();
        Tensor::new(rhs.shape().clone(), result_data).unwrap()
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        match Tensor::mul(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for multiplication."),
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x * rhs).collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let result_data: Vec<f32> = rhs.data().iter().map(|&x| x * self).collect();
        Tensor::new(rhs.shape().clone(), result_data).unwrap()
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        match Tensor::div(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for division."),
        }
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x / rhs).collect();
        Tensor::new(self.shape().clone(), result_data).unwrap()
    }
}

mod errors;
mod ops;
mod traits;
mod utils;

use errors::TensorError;
use std::fmt;
use utils::calculate_strides;

pub trait Numeric:
    Copy
    + fmt::Display
    + fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + PartialEq
    + PartialOrd
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn neg_infinity() -> Self;
    fn nan() -> Self;
    fn is_sign_negative(self) -> bool;
    fn from_usize(val: usize) -> Self;
}

impl Numeric for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }
    fn from_usize(val: usize) -> Self {
        val as f32
    }
}

impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }
    fn from_usize(val: usize) -> Self {
        val as f64
    }
}

pub struct Tensor<T: Numeric = f32> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<T>,
}

impl<T: Numeric> Tensor<T> {
    pub fn new<S>(shape: S, data: Vec<T>) -> Result<Self, TensorError>
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
            data: vec![T::zero(); num_elements],
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
            data: vec![T::one(); num_elements],
        }
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
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

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
}

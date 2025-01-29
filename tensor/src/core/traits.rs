use super::Tensor;
use crate::tensor::utils::print_tensor_recursive;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

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

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace addition")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a += b;
            });
    }
}

impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, rhs: f32) {
        self.data.iter_mut().for_each(|a| {
            *a += rhs;
        });
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

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a -= b;
            });
    }
}

impl SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, rhs: f32) {
        self.data.iter_mut().for_each(|a| {
            *a -= rhs;
        });
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

impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: &Tensor) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a *= b;
            });
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, rhs: f32) {
        self.data.iter_mut().for_each(|a| {
            *a *= rhs;
        });
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

impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, rhs: &Tensor) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a /= b;
            });
    }
}

impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, rhs: f32) {
        self.data.iter_mut().for_each(|a| {
            *a /= rhs;
        });
    }
}

use crate::core::utils::print_tensor_recursive;
use crate::core::Numeric;
use crate::Tensor;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

impl<T: Numeric> fmt::Display for Tensor<T> {
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

impl<T: Numeric> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get(indices).expect("Index out of bounds")
    }
}

impl<T: Numeric> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        match Tensor::add(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for addition."),
        }
    }
}

impl<T: Numeric> Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        let result_data: Vec<T> = self.data().iter().map(|&x| x + rhs).collect();
        Tensor::new(self.shape(), result_data).unwrap()
    }
}


impl<T: Numeric> AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace addition")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a += *b;
            });
    }
}

impl<T: Numeric> AddAssign<T> for Tensor<T> {
    fn add_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| {
            *a += rhs;
        });
    }
}

impl<T: Numeric> Sub<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        match Tensor::sub(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for subtraction."),
        }
    }
}

impl<T: Numeric> Sub<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let result_data: Vec<T> = self.data().iter().map(|&x| x - rhs).collect();
        Tensor::new(self.shape(), result_data).unwrap()
    }
}


impl<T: Numeric> SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a -= *b;
            });
    }
}

impl<T: Numeric> SubAssign<T> for Tensor<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| {
            *a -= rhs;
        });
    }
}

impl<T: Numeric> Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        match Tensor::mul(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for multiplication."),
        }
    }
}

impl<T: Numeric> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let result_data: Vec<T> = self.data().iter().map(|&x| x * rhs).collect();
        Tensor::new(self.shape(), result_data).unwrap()
    }
}


impl<T: Numeric> MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, rhs: &Tensor<T>) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a *= *b;
            });
    }
}

impl<T: Numeric> MulAssign<T> for Tensor<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| {
            *a *= rhs;
        });
    }
}

impl<T: Numeric> Div<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: &Tensor<T>) -> Self::Output {
        match Tensor::div(&self, &rhs) {
            Ok(result) => result,
            Err(_) => panic!("Shapes of the tensors do not match for division."),
        }
    }
}

impl<T: Numeric> Div<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        let result_data: Vec<T> = self.data().iter().map(|&x| x / rhs).collect();
        Tensor::new(self.shape(), result_data).unwrap()
    }
}

impl<T: Numeric> DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, rhs: &Tensor<T>) {
        if *self.shape() != *rhs.shape() {
            panic!("The tensor shape not compatible for inplace subtraction")
        }
        self.data
            .iter_mut()
            .zip(rhs.data().iter())
            .for_each(|(a, b)| {
                *a /= *b;
            });
    }
}

impl<T: Numeric> DivAssign<T> for Tensor<T> {
    fn div_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a| {
            *a /= rhs;
        });
    }
}

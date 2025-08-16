use crate::{Tensor, core::Numeric};

impl<T: Numeric> Tensor<T> {
    fn unary_op<F>(&self, op: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let result_data: Vec<T> = self.data().iter().map(|x| op(*x)).collect();
        Tensor::new(self.shape(), result_data).unwrap()
    }

    pub fn exp(&self) -> Tensor<T> {
        self.unary_op(|x| x.exp())
    }

    pub fn log(&self) -> Tensor<T> {
        self.unary_op(|x| {
            if x == T::zero() {
                T::neg_infinity()
            } else if x.is_sign_negative() {
                T::nan()
            } else {
                x.ln()
            }
        })
    }

    pub fn relu(&self) -> Tensor<T> {
        self.unary_op(|x| if x > T::zero() { x } else { T::zero() })
    }
}

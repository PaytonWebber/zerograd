use crate::Tensor;

impl Tensor {
    fn unary_op<F>(&self, op: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let result_data: Vec<f32> = self.data().iter().map(|x| op(*x)).collect();
        return Tensor::new(self.shape(), result_data).unwrap();
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(|x| x.exp())
    }

    pub fn log(&self) -> Tensor {
        self.unary_op(|x| {
            if x == 0.0 {
                f32::NEG_INFINITY // log(0) -> -inf
            } else if x < 0.0 {
                f32::NAN // log of negative numbers is undefined
            } else {
                x.ln()
            }
        })
    }

    pub fn relu(&self) -> Tensor {
        self.unary_op(|x| if x > 0.0_f32 { x } else { 0.0_f32 })
    }
}

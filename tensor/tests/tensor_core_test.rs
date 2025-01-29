use tensor::Tensor;

/* CREATION OPS */

#[test]
fn create_tensor_from_data() {
    let shape = vec![3, 4, 3];
    let strides = vec![12, 3, 1];
    let length: usize = shape.iter().product();
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let expected_data: Vec<f32> = data.to_vec();
    let a = Tensor::new(shape.clone(), data).unwrap();

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_zeros_tensor() {
    let shape = vec![4, 2];
    let strides = vec![2, 1];
    let length: usize = shape.iter().product();
    let expected_data = vec![0.0; length];
    let a = Tensor::zeros(shape.clone());

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_ones_tensor() {
    let shape = vec![1, 9, 2, 5];
    let strides = vec![90, 10, 5, 1];
    let length: usize = shape.iter().product();
    let expected_data = vec![1.0; length];
    let a = Tensor::ones(shape.clone());

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

/* MOVEMENT OPS */

#[test]
fn reshape_tensor_valid_shape() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(original_shape);

    let new_shape = vec![2, 2, 2];
    let new_strides = vec![4, 2, 1];
    a.reshape(new_shape.clone()).unwrap();

    assert_eq!(new_shape, *a.shape());
    assert_eq!(new_strides, *a.strides());
}

#[test]
fn reshape_tensor_invalid_shape() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(original_shape);

    let new_shape = vec![7, 6];
    if a.reshape(new_shape).is_ok() {
        panic!("The new shape should've been invalid.");
    }
}

#[test]
fn permute_tensor_valid_order() {
    let original_shape = vec![1, 4, 2];
    let mut a = Tensor::ones(original_shape);

    let permutation = vec![1, 2, 0];
    let new_strides = vec![2, 1, 8];
    let new_shape = vec![4, 2, 1];
    a.permute(permutation).unwrap();

    assert_eq!(new_shape, *a.shape());
    assert_eq!(new_strides, *a.strides());
}

#[test]
fn permute_tensor_index_out_of_range() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(original_shape);

    let permutation = vec![4, 2, 1]; // invalid since original_shape.len() = 2
    if a.permute(permutation).is_ok() {
        panic!("The permutation should've been invalid.");
    }
}

#[test]
fn permute_tensor_invalid_order() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(original_shape);

    let permutation = vec![2, 0, 1]; // not a proper permutation of [0, 1]
    if a.permute(permutation).is_ok() {
        panic!("The permutation should've been invalid.");
    }
}

#[test]
fn flatten_tensor() {
    let length: usize = 42;
    let expected_data = vec![1.0_f32; length];
    let mut a = Tensor::ones(vec![7, 6]);

    a.flatten();
    let elem: f32 = a[&[22]];
    assert_eq!(vec![length], *a.shape());
    assert_eq!(vec![1], *a.strides());
    assert_eq!(expected_data, *a.data());
    assert_eq!(elem, 1.0_f32);
}

#[test]
fn transpose_tensor() {
    // Create a 2D tensor:
    // A = [ [1, 2, 3],
    //       [4, 5, 6] ]
    let shape = vec![2, 3];
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut a = Tensor::new(shape, data).unwrap();

    // Transpose A:
    // A^T should be:
    // [ [1, 4],
    //   [2, 5],
    //   [3, 6] ]
    a.transpose().unwrap();
    assert_eq!(*a.shape(), vec![3, 2]);
    assert_eq!(*a.strides(), vec![2, 1]);

    // Check values:
    // A^T[0, 0] = 1, A^T[0, 1] = 4
    // A^T[1, 0] = 2, A^T[1, 1] = 5
    // A^T[2, 0] = 3, A^T[2, 1] = 6
    assert_eq!(a[&[0, 0]], 1.0);
    assert_eq!(a[&[1, 0]], 2.0);
    assert_eq!(a[&[2, 0]], 3.0);
    assert_eq!(a[&[0, 1]], 4.0);
    assert_eq!(a[&[1, 1]], 5.0);
    assert_eq!(a[&[2, 1]], 6.0);
}

/* REDUCTION OPS */

#[test]
fn tensor_sum() {
    let shape = vec![5];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum();
    assert_eq!(vec![15.0_f32], *result.data());

    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum();
    assert_eq!(vec![21.0_f32], *result.data());

    let shape = vec![2, 2, 3];
    let data: Vec<f32> = vec![1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum();
    assert_eq!(vec![36.0_f32], *result.data());
}

#[test]
fn tensor_sum_dim() {
    let shape = vec![5];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum_dim(0).unwrap();
    assert_eq!(vec![15.0_f32], *result.data());

    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum_dim(0).unwrap();
    assert_eq!(vec![5.0_f32, 7.0_f32, 9.0_f32], *result.data());

    let result = a.sum_dim(1).unwrap();
    assert_eq!(vec![6.0_f32, 15.0_f32], *result.data());

    let shape = vec![2, 2, 3];
    let data: Vec<f32> = vec![1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.sum_dim(0).unwrap();
    let expected: Vec<f32> = vec![3.0, 6.0, 7.0, 5.0, 7.0, 8.0];
    assert_eq!(expected, *result.data());
    assert_eq!(vec![2, 3], *result.shape());

    let result = a.sum_dim(1).unwrap();
    let expected: Vec<f32> = vec![5.0, 8.0, 9.0, 3.0, 5.0, 6.0];
    assert_eq!(expected, *result.data());
    assert_eq!(vec![2, 3], *result.shape());

    let result = a.sum_dim(2).unwrap();
    let expected: Vec<f32> = vec![7.0, 15.0, 9.0, 5.0];
    assert_eq!(expected, *result.data());
    assert_eq!(vec![2, 2], *result.shape());
}

#[test]
fn tensor_mean() {
    let shape = vec![5];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.mean();
    assert_eq!(vec![3.0_f32], *result.data());

    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.mean();
    assert_eq!(vec![3.5_f32], *result.data());

    let shape = vec![2, 2, 3];
    let data: Vec<f32> = vec![1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.mean();
    assert_eq!(vec![3.0_f32], *result.data());
}

#[test]
fn tensor_mean_dim() {
    let shape = vec![5];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.mean_dim(0).unwrap();
    assert_eq!(vec![3.0_f32], *result.data());

    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.mean_dim(0).unwrap();
    assert_eq!(vec![2.5_f32, 3.5_f32, 4.5_f32], *result.data());

    let result = a.mean_dim(1).unwrap();
    assert_eq!(vec![2.0_f32, 5.0_f32], *result.data());
}

/* UNARY OPS */

#[test]
fn tensor_exp() {
    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.exp();
    let expected_result: Vec<f32> = vec![
        2.7182817, 7.389056, 20.085537, 54.59815, 148.41316, 403.4288,
    ];
    assert_eq!(expected_result, *result.data());
}

#[test]
fn tensor_log() {
    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.log();
    let expected_result: Vec<f32> = vec![0.0, 0.6931472, 1.0986123, 1.3862944, 1.609438, 1.7917595];
    assert_eq!(expected_result, *result.data());

    let shape = vec![2];
    let data: Vec<f32> = vec![1.0, 0.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.log();
    let expected_result: Vec<f32> = vec![0.0, f32::NEG_INFINITY];
    assert_eq!(expected_result, *result.data());
}

#[test]
fn tensor_relu() {
    let shape = vec![2, 3];
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    let result = a.relu();
    let expected_data: Vec<f32> = vec![1.0, 0.0, 3.0, 4.0, 0.0, 6.0];
    assert_eq!(expected_data, *result.data());
}

/* BINARY OPS */

#[test]
fn tensor_addition_method() {
    let shape = vec![4, 2];
    let a = Tensor::ones(shape.clone());
    let b = Tensor::ones(shape);
    let result = a.add(&b).unwrap();

    let expected_data = vec![2.0_f32; 8];
    assert_eq!(expected_data, *result.data());
}

#[test]
fn tensor_broadcasted_addition_method() {
    let a_shape = vec![4, 3];
    let b_shape = vec![3];
    let a_data = vec![
        0_f32, 0_f32, 0_f32, 10_f32, 10_f32, 10_f32, 20_f32, 20_f32, 20_f32, 30_f32, 30_f32, 30_f32,
    ];
    let b_data = vec![1_f32, 2_f32, 3_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.add(&b_tensor).unwrap();
    let expected_data = vec![
        1_f32, 2_f32, 3_f32, 11_f32, 12_f32, 13_f32, 21_f32, 22_f32, 23_f32, 31_f32, 32_f32, 33_f32,
    ];

    assert_eq!(vec![4, 3], *c.shape());
    assert_eq!(expected_data, *c.data());
}

#[test]
fn tensor_addition_operator() {
    let shape = vec![4, 2];
    let a = Tensor::ones(shape.clone());
    let b = Tensor::ones(shape);
    let result = &a + &b;

    let expected_data = vec![2.0_f32; 8];
    assert_eq!(expected_data, *result.data());

    let result = &a + 2.0_f32;
    let expected_data = vec![3.0_f32; 8];
    assert_eq!(expected_data, *result.data());

    let result = 2.0 + &a;
    assert_eq!(expected_data, *result.data());
}

#[test]
fn tensor_broadcasted_addition_operator() {
    let a_shape = vec![4, 3];
    let b_shape = vec![3];
    let a_data = vec![
        0_f32, 0_f32, 0_f32, 10_f32, 10_f32, 10_f32, 20_f32, 20_f32, 20_f32, 30_f32, 30_f32, 30_f32,
    ];
    let b_data = vec![1_f32, 2_f32, 3_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = &a_tensor + &b_tensor;
    let expected_data = vec![
        1_f32, 2_f32, 3_f32, 11_f32, 12_f32, 13_f32, 21_f32, 22_f32, 23_f32, 31_f32, 32_f32, 33_f32,
    ];

    assert_eq!(vec![4, 3], *c.shape());
    assert_eq!(expected_data, *c.data());
}

#[test]
fn tensor_add_inplace_method() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::ones(b_shape);
    a.add_inplace(&b);

    let expected_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_add_inplace_operator() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();

    a += &b;
    let expected_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    assert_eq!(expected_data, *a.data());

    a += 4.0;
    let expected_data: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_subtraction_method() {
    let shape = vec![4, 2];
    let a = Tensor::ones(shape.clone());
    let b = Tensor::ones(shape);
    let result = a.sub(&b).unwrap();

    let expected_data = vec![0.0_f32; 8];
    assert_eq!(expected_data, *result.data());
}

#[test]
fn tensor_broadcasted_subtraction_method() {
    let a_shape = vec![4, 3];
    let b_shape = vec![3];
    let a_data = vec![
        0_f32, 0_f32, 0_f32, 10_f32, 10_f32, 10_f32, 20_f32, 20_f32, 20_f32, 30_f32, 30_f32, 30_f32,
    ];
    let b_data = vec![1_f32, 2_f32, 3_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.sub(&b_tensor).unwrap();
    let expected_data = vec![
        -1_f32, -2_f32, -3_f32, 9_f32, 8_f32, 7_f32, 19_f32, 18_f32, 17_f32, 29_f32, 28_f32, 27_f32,
    ];

    assert_eq!(vec![4, 3], *c.shape());
    assert_eq!(expected_data, *c.data());
}

#[test]
fn tensor_subtraction_operator() {
    let shape = vec![4, 2];
    let a = Tensor::ones(shape.clone());
    let b = Tensor::ones(shape);

    let result = &a - &b;
    let expected_data = vec![0.0_f32; 8];
    assert_eq!(expected_data, *result.data());

    let result = &a - 2.0_f32;
    let expected_data = vec![-1.0_f32; 8];
    assert_eq!(expected_data, *result.data());

    let result = 2.0 - &a;
    assert_eq!(expected_data, *result.data());
}

#[test]
fn tensor_broadcasted_subtraction_operator() {
    let a_shape = vec![4, 3];
    let b_shape = vec![3];
    let a_data = vec![
        0_f32, 0_f32, 0_f32, 10_f32, 10_f32, 10_f32, 20_f32, 20_f32, 20_f32, 30_f32, 30_f32, 30_f32,
    ];
    let b_data = vec![1_f32, 2_f32, 3_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = &a_tensor - &b_tensor;
    let expected_data = vec![
        -1_f32, -2_f32, -3_f32, 9_f32, 8_f32, 7_f32, 19_f32, 18_f32, 17_f32, 29_f32, 28_f32, 27_f32,
    ];

    assert_eq!(vec![4, 3], *c.shape());
    assert_eq!(expected_data, *c.data());
}

#[test]
fn tensor_sub_inplace_method() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::ones(b_shape);
    a.sub_inplace(&b);

    let expected_data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_sub_inplace_operator() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();

    a -= &b;
    let expected_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(expected_data, *a.data());

    a -= 2.0;
    let expected_data: Vec<f32> = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_mul_method() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![3_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.mul(&b_tensor).unwrap();
    let expected = vec![3_f32, 4_f32, 3_f32];
    assert_eq!(expected, *c.data());

    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32, 2_f32, 2_f32, 1_f32];
    let b_data = vec![2_f32, 4_f32, 6_f32, 1_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.mul(&b_tensor).unwrap();
    let expected = vec![2_f32, 8_f32, 18_f32, 2_f32, 4_f32, 1_f32];
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_mul_operator() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![3_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = &a_tensor * &b_tensor;
    let expected = vec![3_f32, 4_f32, 3_f32];
    assert_eq!(expected, *c.data());

    let c = &a_tensor * 2.0;
    let expected = vec![2_f32, 4_f32, 6_f32];
    assert_eq!(expected, *c.data());

    let c = 2.0 * &a_tensor;
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_broadcasted_mul_method() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![2_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.mul(&b_tensor).unwrap();
    let expected = vec![2_f32, 4_f32, 6_f32];
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_mul_inplace_method() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();
    a.mul_inplace(&b);

    let expected_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_mul_inplace_operator() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();

    a *= &b;
    let expected_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    assert_eq!(expected_data, *a.data());

    a *= 2.0;
    let expected_data: Vec<f32> = vec![4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_div_method() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![3_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.div(&b_tensor).unwrap();
    let expected = vec![(1_f32 / 3_f32), 1_f32, 3_f32];
    assert_eq!(expected, *c.data());

    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32, 2_f32, 2_f32, 1_f32];
    let b_data = vec![2_f32, 4_f32, 6_f32, 1_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.div(&b_tensor).unwrap();
    let expected = vec![0.5_f32, 0.5_f32, 0.5_f32, 2_f32, 1_f32, 1_f32];
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_div_operator() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1, 3];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![3_f32, 2_f32, 1_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = &a_tensor / &b_tensor;
    let expected = vec![(1_f32 / 3_f32), 1_f32, 3_f32];
    assert_eq!(expected, *c.data());

    let c = &a_tensor / 2.0;
    let expected = vec![(1_f32 / 2_f32), 1_f32, 3_f32 / 2.0];
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_broadcasted_div_method() {
    let a_shape = vec![1, 3];
    let b_shape = vec![1];
    let a_data = vec![1_f32, 2_f32, 3_f32];
    let b_data = vec![2_f32];

    let a_tensor = Tensor::new(a_shape, a_data).unwrap();
    let b_tensor = Tensor::new(b_shape, b_data).unwrap();

    let c = a_tensor.div(&b_tensor).unwrap();
    let expected = vec![0.5_f32, 1_f32, 3_f32 / 2_f32];
    assert_eq!(expected, *c.data());
}

#[test]
fn tensor_div_inplace_method() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();
    a.div_inplace(&b);

    let expected_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_div_inplace_operator() {
    let a_shape = vec![2, 3];
    let b_shape = vec![2, 3];
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut a = Tensor::new(a_shape, a_data).unwrap();
    let b = Tensor::new(b_shape, b_data).unwrap();

    a /= &b;
    let expected_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    assert_eq!(expected_data, *a.data());

    a /= 0.5;
    let expected_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(expected_data, *a.data());
}

#[test]
fn tensor_matmul() {
    // A is 2x3:
    // [1, 2, 3]
    // [4, 5, 6]
    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(vec![2, 3], a_data).unwrap();

    // B is 3x2:
    // [7,  8]
    // [9, 10]
    // [11,12]
    let b_data = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b = Tensor::new(vec![3, 2], b_data).unwrap();

    // Expected C = A * B should be 2x2:
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    let expected_data = vec![58.0_f32, 64.0, 139.0, 154.0];

    let c = a.matmul(&b).expect("Matrix multiplication failed");
    println!("{}", c);

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(*c.data(), expected_data);
}

/* EXTRA FUNCTIONS */

#[test]
fn tensor_display_1d() {
    let a = Tensor::new(vec![3], vec![0.0, 1.0, 2.0]).unwrap();
    assert_eq!(format!("{}", a), "tensor([0.0000, 1.0000, 2.0000])");
}

#[test]
fn tensor_display_2d() {
    let t = Tensor::new(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let expected = "tensor([[0.0000, 1.0000, 2.0000]\n        [3.0000, 4.0000, 5.0000]])";
    assert_eq!(format!("{}", t), expected);
}

#[test]
fn tensor_display_3d() {
    let t = Tensor::new(vec![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let expected = "tensor([[[0.0000, 1.0000]\n        [2.0000, 3.0000]]\n\n       [[4.0000, 5.0000]\n        [6.0000, 7.0000]]])";
    assert_eq!(format!("{}", t), expected);
}

#[test]
fn get_element_with_index() {
    let shape = vec![2, 3];
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::new(shape, data).unwrap();

    assert_eq!(a[&[0, 0]], 1.0);
    assert_eq!(a[&[0, 1]], 2.0);
    assert_eq!(a[&[0, 2]], 3.0);
    assert_eq!(a[&[1, 0]], 4.0);
    assert_eq!(a[&[1, 1]], 5.0);
    assert_eq!(a[&[1, 2]], 6.0);
}

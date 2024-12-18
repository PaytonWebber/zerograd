use tensor::Tensor;

#[test]
fn create_tensor_from_data() {
    let shape = vec![3, 4, 3];
    let strides = vec![12, 3, 1];
    let length: usize = shape.iter().product();
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let expected_data: Vec<f32> = data.to_vec();
    let a = Tensor::new(&shape, data).unwrap();

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
    let a = Tensor::zeros(&shape);

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
    let a = Tensor::ones(&shape);

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn reshape_tensor_valid_shape() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(&original_shape);

    let new_shape = vec![2, 2, 2];
    let new_strides = vec![4, 2, 1];
    a.reshape(&new_shape).unwrap();

    assert_eq!(new_shape, *a.shape());
    assert_eq!(new_strides, *a.strides());
}

#[test]
fn reshape_tensor_invalid_shape() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(&original_shape);

    let new_shape = vec![7, 6];
    if a.reshape(&new_shape).is_ok() {
        panic!("The new shape should've been invalid.");
    }
}

#[test]
fn permute_tensor_valid_order() {
    let original_shape = vec![1, 4, 2];
    let mut a = Tensor::ones(&original_shape);

    let permutation = vec![1, 2, 0];
    let new_strides = vec![2, 1, 8];
    let new_shape = vec![4, 2, 1];
    a.permute(&permutation).unwrap();

    assert_eq!(new_shape, *a.shape());
    assert_eq!(new_strides, *a.strides());
}

#[test]
fn permute_tensor_index_out_of_range() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(&original_shape);

    let permutation = vec![4, 2, 1]; // invalid since original_shape.len() = 2
    if a.permute(&permutation).is_ok() {
        panic!("The permutation should've been invalid.");
    }
}

#[test]
fn permute_tensor_invalid_order() {
    let original_shape = vec![4, 2];
    let mut a = Tensor::ones(&original_shape);

    let permutation = vec![2, 0, 1]; // not a proper permutation of [0, 1]
    if a.permute(&permutation).is_ok() {
        panic!("The permutation should've been invalid.");
    }
}

#[test]
fn flatten_tensor() {
    let length: usize = 42;
    let expected_data = vec![1.0_f32; length];
    let mut a = Tensor::ones(&[7, 6]);

    a.flatten();
    let elem: f32 = a[&[22]];
    assert_eq!(vec![length], *a.shape());
    assert_eq!(vec![1], *a.strides());
    assert_eq!(expected_data, *a.data());
    assert_eq!(elem, 1.0_f32);
}

#[test]
fn get_element_with_index() {
    let length: usize = 24;
    let shape = vec![3, 2, 4];
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let a = Tensor::new(&shape, data).unwrap();

    let elem: f32 = a[&[1, 0, 3]];
    assert_eq!(elem, 21.0_f32);
}

#[test]
fn tensor_addition_method() {
    let shape = vec![4, 2];
    let a = Tensor::ones(&shape);
    let b = Tensor::ones(&shape);
    let result = a.add(&b).unwrap();

    let expected_data = vec![2.0_f32; 8];
    assert_eq!(expected_data, *result.data());
}

#[test]
fn tensor_addition_operator() {
    let shape = vec![4, 2];
    let a = Tensor::ones(&shape);
    let b = Tensor::ones(&shape);
    let result = a + b;

    let expected_data = vec![2.0_f32; 8];
    assert_eq!(expected_data, *result.data());
}

#[test]
fn test_display_1d() {
    let a = Tensor::new(&[3], vec![0.0, 1.0, 2.0]).unwrap();
    assert_eq!(format!("{}", a), "tensor([0.0000, 1.0000, 2.0000])");
}

#[test]
fn test_display_2d() {
    let t = Tensor::new(&[2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let expected = "tensor([[0.0000, 1.0000, 2.0000]\n        [3.0000, 4.0000, 5.0000]])";
    assert_eq!(format!("{}", t), expected);
}

#[test]
fn test_display_3d() {
    let t = Tensor::new(&[2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let expected = "tensor([[[0.0000, 1.0000]\n        [2.0000, 3.0000]]\n\n       [[4.0000, 5.0000]\n        [6.0000, 7.0000]]])";
    assert_eq!(format!("{}", t), expected);
}

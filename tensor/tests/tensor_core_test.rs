use tensor::Tensor;

#[test]
fn create_tensor_from_data() {
    let shape: Vec<usize> = vec![3, 4, 3];
    let strides: Vec<usize> = vec![12, 3, 1];
    let length: usize = shape.iter().product();
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let expected_data: Vec<f32> = data.to_vec().clone();
    let a: Tensor = Tensor::new(&shape, data).unwrap();

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_zeros_tensor() {
    let shape: Vec<usize> = vec![4, 2];
    let strides: Vec<usize> = vec![2, 1];
    let length: usize = shape.iter().product();
    let expected_data: Vec<f32> = vec![0.0; length];
    let a: Tensor = Tensor::zeros(&shape);

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_ones_tensor() {
    let shape: Vec<usize> = vec![1, 9, 2, 5];
    let strides: Vec<usize> = vec![90, 10, 5, 1];
    let length: usize = shape.iter().product();
    let expected_data: Vec<f32> = vec![1.0; length];
    let a: Tensor = Tensor::ones(&shape);

    assert_eq!(shape, *a.shape());
    assert_eq!(strides, *a.strides());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn reshape_tensor_valid_shape() {
    let original_shape: Vec<usize> = vec![4, 2];
    let mut a: Tensor = Tensor::ones(&original_shape);

    let new_shape: Vec<usize> = vec![2, 2, 2];
    a.reshape(&new_shape).unwrap();

    assert_eq!(new_shape, *a.shape());
}

#[test]
fn reshape_tensor_invalid_shape() {
    let original_shape: Vec<usize> = vec![4, 2];
    let mut a: Tensor = Tensor::ones(&original_shape);

    let new_shape: Vec<usize> = vec![7, 6];
    if let Ok(()) = a.reshape(&new_shape) {
        panic!("The new shape should've been invalid.")
    }
}

#[test]
fn get_element_with_index() {
    let length: usize = 24;
    let shape: Vec<usize> = vec![3, 2, 4];
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let a: Tensor = Tensor::new(&shape, data).unwrap();

    let elem: f32 = a[&[1, 0, 3]];
    assert_eq!(elem, 21.0_f32);
}

use tensor::Tensor;

#[test]
fn create_tensor_from_data() {
    let shape: Vec<usize> = vec![3, 4, 3];
    let length: usize = shape.iter().product();
    let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();
    let expected_data: Vec<f32> = data.to_vec().clone();
    let a: Tensor = Tensor::new(&shape, data).unwrap();

    assert_eq!(shape, *a.shape());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_zeros_tensor() {
    let shape: Vec<usize> = vec![4, 2];
    let length: usize = shape.iter().product();
    let expected_data: Vec<f32> = vec![0.0; length];
    let a: Tensor = Tensor::zeros(&shape);

    assert_eq!(shape, *a.shape());
    assert_eq!(expected_data, *a.data());
}

#[test]
fn create_ones_tensor() {
    let shape: Vec<usize> = vec![1, 9, 2, 5];
    let length: usize = shape.iter().product();
    let expected_data: Vec<f32> = vec![1.0; length];
    let a: Tensor = Tensor::ones(&shape);

    assert_eq!(shape, *a.shape());
    assert_eq!(expected_data, *a.data());
}

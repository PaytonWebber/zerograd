# tensor

**tensor** is a subcrate of the [zerograd] project that I am currently working on. This crate provides the core **tensor** data structure and operations (e.g., broadcasting, reshaping, arithmetic ops) that power will eventually be used with an automatic-differentiation engine.

---

## Table of Contents
1. [Features](#features)  
2. [Usage](#usage)  
3. [Examples](#examples)  

---

## Features

- **Flexible Tensor Data Structure**: Supports arbitrary-rank tensors with shape/stride-based internal representation in row-major order.  
- **Broadcasting**: Elementwise ops automatically broadcast based on [NumPy's rules](https://numpy.org/doc/stable/user/basics.broadcasting.html).  
- **Binary Ops**: Addition, subtraction, multiplication, division (both in-place and out-of-place), and matmul.  
- **Reduction Ops**: Sum, mean, and dimension-based reductions for partial aggregation.  
- **Movement Ops**: `reshape`, `permute`, `flatten`, and `transpose`.
- **Unary Ops**: Exponential, logarithm, ReLU.  
- **Trait Integrations**: Implements Rust’s `Add`, `Sub`, `Mul`, `Div`, and their in-place variants to make the interface more user-friendly.  
- **Educational Focus**: Written in a clear, beginner-friendly style, making it easy to learn how tensors work under the hood.

---

## Usage

This subcrate provides a core `Tensor` type along with standard tensor operations like creation, reshaping, indexing, arithmetic (with and without broadcasting), reductions, and more. The primary entry point is the `Tensor` struct:

```rust
use tensor::Tensor;
```

### 1. Creating Tensors

1. **From Existing Data**  
   ```rust
   // shape is [3, 4, 3]
   let shape = vec![3, 4, 3];
   let length = shape.iter().product();
   let data: Vec<f32> = (0..length).map(|v| v as f32 + 10.0).collect();

   // Create a Tensor, ensuring data.len() matches the product of shape
   let a = Tensor::new(shape.clone(), data).unwrap();
   ```
2. **Zeros and Ones**  
   ```rust
   // 4x2 filled with zeros
   let zeros_tensor = Tensor::zeros(vec![4, 2]);

   // 1x9x2x5 filled with ones
   let ones_tensor = Tensor::ones(vec![1, 9, 2, 5]);
   ```

### 2. Indexing and Accessing Data

- **Indexing** via `Index<Vec<usize>>`:  
  ```rust
  let t = Tensor::ones(vec![2, 3]);
  // t[vec![row, col]]
  assert_eq!(t[vec![0, 2]], 1.0_f32);
  ```
- **Check `.shape()` and `.strides()`**:  
  ```rust
  println!("Shape = {:?}", t.shape());   // e.g. [2, 3]
  println!("Strides = {:?}", t.strides()); // e.g. [3, 1]
  ```
- **Access raw data**:
  ```rust
  let data_ref: &Vec<f32> = t.data();
  // or get mutable reference with t.data_mut()
  ```

### 3. Movement Ops: Reshaping, Permuting, Flattening, Transposing

1. **Reshape**  
   ```rust
   let mut a = Tensor::ones(vec![4, 2]);
   a.reshape(vec![2, 2, 2]).unwrap(); 
   // shape is now [2, 2, 2]
   ```
2. **Permute** (change dimension ordering)  
   ```rust
   let mut a = Tensor::ones(vec![1, 4, 2]);
   // reorder dimensions to [1->4, 4->2, 2->1]
   a.permute(vec![1, 2, 0]).unwrap();  
   // shape is now [4, 2, 1]
   // strides updated accordingly
   ```
3. **Flatten**  
   ```rust
   let mut a = Tensor::ones(vec![7, 6]);
   a.flatten();
   // shape becomes [42], strides is [1]
   ```
4. **Transpose**  
   ```rust
   let mut a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
   a.transpose().unwrap();
   // shape is now [3, 2], data is reordered
   ```

### 4. Arithmetic Operations

All arithmetic methods come in two flavors:

- **Out-of-place**: returns a new `Tensor` (e.g. `add`, `sub`, `mul`, `div`).  
- **In-place**: modifies the existing tensor (e.g. `add_inplace`, `sub_inplace`, etc.).

They also have matching **trait operators**:

- `&a + &b` or `a.add(&b)`
- `a += &b` or `a.add_inplace(&b)`
- Same for `-`, `*`, `/`.

#### a) Simple Elementwise

```rust
let a = Tensor::ones(vec![2, 3]);
let b = Tensor::ones(vec![2, 3]);

// Out-of-place
let c = a.add(&b).unwrap();  // or &a + &b
// c has all 2.0 values

// In-place
let mut d = Tensor::zeros(vec![2, 3]);
d.add_inplace(&c);
```

#### b) Broadcasting

Broadcasting works as follows:

```rust
let a = Tensor::new(vec![4, 3], 
                    vec![0.,0.,0., 10.,10.,10., 20.,20.,20., 30.,30.,30.]).unwrap();
let b = Tensor::new(vec![3], vec![1., 2., 3.]).unwrap();

let c = a.add(&b).unwrap(); // shape is still [4, 3]
```

#### c) Scalar Operations

```rust
let a = Tensor::ones(vec![2, 2]);
let b = &a + 2.0;    // shape [2,2], adds 2.0 to each element
let c = 2.0 + &a;    // same result, uses "impl Add<&Tensor> for f32"
let d = &a * 3.0;
let e = &a / 0.5;
```

#### d) Matrix Multiplication

```rust
let a = Tensor::new(vec![2, 3], vec![1.,2.,3., 4.,5.,6.]).unwrap();
let b = Tensor::new(vec![3, 2], vec![7.,8., 9.,10., 11.,12.]).unwrap();
let c = a.matmul(&b).unwrap();
// shape [2, 2], data [58.0, 64.0, 139.0, 154.0]
```

### 5. Reduction Ops

1. **Sum**  
   ```rust
   let a = Tensor::new(vec![2, 3], vec![1.,2.,3., 4.,5.,6.]).unwrap();
   let s = a.sum();
   // s.shape() == [1], s.data() == [21.0]
   ```
2. **Sum Along a Dimension**  
   ```rust
   let reduced = a.sum_dim(1).unwrap();
   // For shape [2, 3], sum_dim(1) -> shape [2], data is sum across columns
   // e.g. [6., 15.]
   ```
3. **Mean** and **Mean Along Dimension**  
   ```rust
   let m = a.mean();          // overall mean
   let m_dim = a.mean_dim(0); // per-row or per-col mean
   ```

### 6. Unary Ops

- `exp()`: Elementwise exponent.  
- `log()`: Elementwise natural log (handles `0.0 -> -inf`, negative -> `NaN`).  
- `relu()`: ReLU activation `max(0,x)`.

```rust
let a = Tensor::new(vec![1, 3], vec![1.0, -2.0, 3.0]).unwrap();
let b = a.relu();
assert_eq!(*b.data(), vec![1.0, 0.0, 3.0]);
```

### 7. Display and Debug

Tensors have a custom `fmt::Display` so you can see them in a human-readable format:

```rust
let a = Tensor::new(vec![2,3], vec![0., 1., 2., 3., 4., 5.]).unwrap();
println!("{}", a);

// tensor([[0.0000, 1.0000, 2.0000]
//         [3.0000, 4.0000, 5.0000]])
```
For more examples, see the [`tests` folder](tests).

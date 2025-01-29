pub fn is_broadcastable(a_shape: &[usize], b_shape: &[usize]) -> bool {
    // This is based on NumPy's rules: https://numpy.org/doc/stable/user/basics.broadcasting.html
    for (i, j) in a_shape.iter().rev().zip(b_shape.iter().rev()) {
        if *i == 1 || *j == 1 {
            continue;
        }
        if *i != *j {
            return false;
        }
    }
    true
}

pub fn compute_broadcast_shape_and_strides(
    a_shape: &[usize],
    b_shape: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let ndims = a_shape.len().max(b_shape.len());
    let mut a_bc_strides = vec![1; ndims];
    let mut b_bc_strides = vec![1; ndims];
    let mut bc_shape = vec![0; ndims];
    let mut a_dims = vec![0; ndims];
    let mut b_dims = vec![0; ndims];

    let a_shape_rev: Vec<usize> = a_shape.iter().copied().rev().collect();
    let b_shape_rev: Vec<usize> = b_shape.iter().copied().rev().collect();
    for i in 0..ndims {
        let dim_a = a_shape_rev.get(i).copied().unwrap_or(1);
        let dim_b = b_shape_rev.get(i).copied().unwrap_or(1);
        a_dims[ndims - i - 1] = dim_a;
        b_dims[ndims - i - 1] = dim_b;
        if i == 0 {
            if dim_a != dim_b {
                a_bc_strides[ndims - i - 1] = match dim_a {
                    1 => 0,
                    _ => a_dims[ndims - i..].iter().product(),
                };
                b_bc_strides[ndims - i - 1] = match dim_b {
                    1 => 0,
                    _ => b_dims[ndims - i..].iter().product(),
                };
            }
        } else {
            if dim_a != dim_b {
                a_bc_strides[ndims - i - 1] = match dim_a {
                    1 => 0,
                    _ => a_dims[ndims - i..].iter().product(),
                };
                b_bc_strides[ndims - i - 1] = match dim_b {
                    1 => 0,
                    _ => b_dims[ndims - i..].iter().product(),
                };
            } else {
                a_bc_strides[ndims - i - 1] = a_dims[ndims - i..].iter().product();
                b_bc_strides[ndims - i - 1] = b_dims[ndims - i..].iter().product();
            }
        }
        bc_shape[ndims - i - 1] = dim_a.max(dim_b);
    }
    (bc_shape, a_bc_strides, b_bc_strides)
}

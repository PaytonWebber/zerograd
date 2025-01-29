use std::fmt;

pub fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
    let length: usize = shape.len();
    let mut strides = vec![1; length];
    strides.iter_mut().enumerate().for_each(|(i, stride)| {
        // stride[i] = (shape[i+1]*shape[i+2]*...*shape[N-1])
        *stride = shape.iter().take(length).skip(i + 1).product();
    });
    strides
}

pub fn unravel_index(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0; ndim];
    for j in (0..ndim).rev() {
        let dim_size = shape[j];
        coords[j] = i % dim_size;
        i /= dim_size;
    }
    coords
}

fn calculate_data_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| idx * strides[i])
        .sum()
}

pub fn print_tensor_recursive(
    f: &mut fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    current_index: &mut [usize],
    dim: usize,
    ndims: usize,
) -> fmt::Result {
    if ndims == 0 {
        if let Some(value) = data.first() {
            return write!(f, "{:.4}", value);
        } else {
            return write!(f, "");
        }
    }

    if dim == ndims - 1 {
        write!(f, "[")?;
        for i in 0..shape[dim] {
            current_index[dim] = i;
            let idx = calculate_data_index(current_index, strides);
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", data[idx])?;
        }
        write!(f, "]")?;
    } else {
        write!(f, "[")?;
        for i in 0..shape[dim] {
            current_index[dim] = i;

            if i > 0 {
                if dim == 0 {
                    if ndims >= 3 {
                        write!(f, "\n\n")?;
                        for _ in 0..7 {
                            write!(f, " ")?;
                        }
                    } else if ndims == 2 {
                        writeln!(f)?;
                        for _ in 0..8 {
                            write!(f, " ")?;
                        }
                    }
                } else {
                    writeln!(f)?;
                    for _ in 0..8 {
                        write!(f, " ")?;
                    }
                }
            }

            print_tensor_recursive(f, data, shape, strides, current_index, dim + 1, ndims)?;
        }
        write!(f, "]")?;
    }

    Ok(())
}

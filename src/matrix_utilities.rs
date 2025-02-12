use std::marker::PhantomData;
use std::ops::MulAssign;
use std::sync::Arc;
use crate::matrix::Matrix;
use crate::number::Number;

pub struct MatrixUtilities<T: Number + MulAssign + Clone> {
    _marker: PhantomData<T>,
}

impl<T: MulAssign + Clone + Number> MatrixUtilities<T> {
    /// Appends a row to a given `Matrix`
    ///
    ///
    /// ### Parameters
    /// - `matrix` - The `Matrix` needed to add a row
    /// - `row`: A `&[i64]` slice
    ///
    /// ### Returns
    /// - An updated `Matrix` object that adds the given `row` 
    ///   to the given `Matrix`
    pub fn append(mut matrix: Matrix<T>, row: &[T]) -> Matrix<T> {
        matrix.mat.push(Arc::from(row));
        matrix.rows = matrix.mat.len();
        matrix.cols = row.len();

        matrix
    }

    /// Appends multiple rows to a given `Matrix`
    ///
    /// ### Parameters
    /// - `matrix`: The `Matrix` required to add additional `rows`
    /// - `rows`: A slice of `&[i64]` slices representing a series of `rows` to add to a given `Matrix`
    ///
    /// ### Returns
    /// - An updated `Matrix` object that adds all arrays to this
    ///   `Matrix`
    pub fn append_multiple(mut matrix: Matrix<T>, rows: &[&[T]]) -> Matrix<T> {
        for &row in rows {
            matrix.mat.push(Arc::from(row));
        }
        matrix.rows = matrix.mat.len();
        if !rows.is_empty() {
            matrix.cols = rows[0].len();
        }

        matrix
    }


    /// Multiplies a given `Matrix` by a given scalar `constant`
    ///
    /// ### Parameters
    /// - `matrix`: The given `Matrix` to be multiplied by a scalar constant
    /// - `constant`: The given scalar constant to multiply the given `Matrix` by
    ///
    /// ### Returns
    /// - A new `Matrix` that contains the matrix after multiplying
    ///   by a scalar constant
    pub fn multiply_by_scalar(mut matrix: Matrix<T>, constant: T) -> Matrix<T> {
        for row in &mut matrix.mat {
            for elem in Arc::make_mut(row) {
                *elem *= constant;
            }
        }
        
        matrix
    }
}

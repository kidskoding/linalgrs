extern crate num;

use crate::number::Number;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

/// A struct representing that of a `Matrix` in linear algebra. This example models a `Matrix`
/// by building one as a computational representation along with including
/// core matrix operations, such as shape and determinant
///
/// In Linear Algebra, a `Matrix` is a rectangular array of numbers,
/// symbols, or expressions, arranged in rows and columns
///
/// Matrices are used to represent and solve systems of linear equations, perform
/// linear transformations, and more
#[derive(Clone, Debug)]
pub struct Matrix<T: Number + PartialEq> {
    /// Represents a vector of `Arc` atomic reference counting `[T]` arrays,
    /// where each represents a row in the `Matrix`
    pub mat: Vec<Arc<[T]>>,

    /// Stores the number of rows in the matrix
    pub rows: usize,

    /// Stores the number of columns in the matrix
    pub cols: usize,
}

impl<T: PartialEq + Number + num::One> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.mat == other.mat
    }
}

/// A macro to create a `Matrix` from a 2D array.
///
/// This macro allows you to create a `Matrix` instance by specifying its elements
/// in a 2D array format. Each inner array represents a row in the matrix.
///
/// ### Parameters
/// - `[$([$elem:expr),* $(,)?]),* $(,)?`: A 2D array where each inner array represents a row.
///
/// ### Returns
/// - A `Matrix` instance containing the specified elements.
#[macro_export]
macro_rules! matrix {
    ($([$($elem:expr),* $(,)?]),* $(,)?) => {
        {
            let mut rows = Vec::new();
            $(
                let row = vec![$($elem),*];
                rows.push(Arc::from(row.as_slice()));
            )*

            Matrix {
                mat: rows.clone(),
                rows: rows.len(),
                cols: if rows.len() > 0 { rows[0].len() } else { 0 },
            }
        }
    };
}

impl<T: Number + num::One> Default for Matrix<T> {
    /// Creates a default representation of this `Matrix`
    ///
    /// ### Returns
    /// - A default constructed `Matrix` object
    fn default() -> Self {
        Matrix {
            mat: vec![],
            rows: 0,
            cols: 0,
        }
    }
}

impl<T: Number + num::One> Display for Matrix<T> {
    /// Writes a `Matrix` as a pretty-printable string
    ///
    /// ### Returns
    /// - Unit result of the write operation
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in &self.mat {
            let mut curr_line = String::new();
            curr_line.push('|');

            for num in i.into_iter() {
                curr_line.push(' ');
                curr_line.push_str(&format!("{}", num));
            }
            
            curr_line.push_str(" |");
            writeln!(f, "{}", curr_line)?;
        }

        Ok(())
    }
}

impl<T: Number + num::One> Matrix<T> {
    /// Compute the shape of this `Matrix`
    ///
    /// The shape of a matrix is defined by the number of rows and
    /// columns it contains and is typically represented as a tuple pair -
    /// `(rows, columns)`
    ///
    /// ### Returns
    /// - A tuple of two positive integers - `(usize, usize)` - representing
    ///   the rows and columns of the matrix
    pub fn shape(&mut self) -> (usize, usize) {
        self.rows = self.mat.len();
        self.cols = if self.rows > 0 { self.mat[0].len() } else { 0 };

        (self.rows, self.cols)
    }

    /// Get a sub-matrix of this `Matrix`
    ///
    /// ### Parameters
    /// - `row_range` - A `Range<usize>` indicating the range of
    ///    rows to extract from this `Matrix`
    /// - `col_range` - A `Range<usize>` indicating the range of
    ///    columns to extract from this `Matrix`
    ///
    /// ### Returns
    /// - A `Result` containing whether this `Matrix` could be extracted
    ///   into a sub-matrix or not
    ///     - An `Ok` variant containing the new sub-matrix as a `Matrix` instance
    ///     - An `Err` with a custom `String` error message if either or
    ///       both provided ranges were out of bounds
    pub fn sub_matrix(
        &mut self,
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) -> Result<Matrix<T>, String> {
        if row_range.end > self.rows || col_range.end > self.cols {
            return Err("Range out of bounds!".to_string());
        }

        let mut new_mat = Vec::new();
        for i in row_range.clone() {
            let new_row: Arc<[T]> = Arc::from(&self.mat[i][col_range.clone()]);
            new_mat.push(new_row);
        }

        Ok(Matrix {
            mat: new_mat,
            rows: row_range.len(),
            cols: col_range.len(),
        })
    }
}

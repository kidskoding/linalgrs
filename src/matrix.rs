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
pub struct Matrix<T: Number + num::One> {
    /// Represents a vector of `Arc` atomic reference counting `[T]` arrays,
    /// where each represents a row in the `Matrix`
    pub mat: Vec<Arc<[T]>>,

    /// Stores the number of rows in the matrix
    pub rows: usize,

    /// Stores the number of columns in the matrix
    pub cols: usize,
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

    /// Compute the determinant of this `Matrix`
    ///
    /// - In a `Matrix` with a shape of `(1, 1)`, a `Matrix`'s determinant is
    /// simply that number itself
    ///
    /// - In a `Matrix` with a shape of `(2, 2)`, a `Matrix`'s determinant is
    /// equal to `ad - bc`, which is the difference
    /// between the left diagonal product and the right diagonal product
    ///
    /// - Any other `Matrix` bigger than a `(2, 2)` (i.e. `(3, 3)`, `(4, 4)`, etc.) utilizes the
    /// [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) approach.
    /// The [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) approach involves expanding
    /// the determinant along a row or column breaking it down into smaller sub-matrices until reaching
    /// 2x2 matrices, where the determinant can directly be calculated using the formula `ad - bc`
    ///
    /// ### Returns
    /// - A `Result` determining whether the determinant could be calculated
    ///     - An `Err` if the `Matrix`'s shape could not be calculated
    ///     - An `Ok` with the determinant value, if this `Matrix`'s
    ///       shape is `(2, 2)` - 2 rows and 2 columns
    pub fn determinant(&mut self) -> Option<T> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return None;
        }

        match rows {
            1 => Some(self.mat[0][0]),
            2 => {
                let ad = self.mat[0][0] * self.mat[1][1];
                let bc = self.mat[0][1] * self.mat[1][0];

                Some(ad - bc)
            }
            _ => Some(self.laplace_expansion_det_helper()),
        }
    }
    fn laplace_expansion_det_helper(&mut self) -> T {
        let (rows, cols) = self.shape();

        if rows == 1 {
            return self.mat[0][0];
        }
        if rows == 2 {
            let ad = self.mat[0][0] * self.mat[1][1];
            let bc = self.mat[0][1] * self.mat[1][0];
            return ad - bc;
        }

        let mut det = T::default();

        for col in 0..cols {
            let mut sub_matrix = self.create_laplace_expansion_submatrix(col);

            let sign = if col % 2 == 0 {
                T::default() + num::One::one()
            } else {
                T::default() - num::One::one()
            };

            det += sign * self.mat[0][col] * sub_matrix.laplace_expansion_det_helper();
        }

        det
    }
    fn create_laplace_expansion_submatrix(&mut self, exclude_col: usize) -> Matrix<T> {
        let (rows, cols) = self.shape();
        let mut new_matrix = Vec::new();

        for i in 1..rows {
            let filtered_row: Vec<T> = self.mat[i]
                .iter()
                .enumerate()
                .filter_map(|(j, &val)| if j != exclude_col { Some(val) } else { None })
                .collect();

            new_matrix.push(Arc::from(filtered_row.as_slice()));
        }

        Matrix {
            mat: new_matrix,
            rows: rows - 1,
            cols: cols - 1,
        }
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

    /// Computes the transpose of this `Matrix`
    ///
    /// The transpose of a `Matrix` is the resulting matrix where the columns are
    /// formed from the corresponding rows of the original matrix
    ///
    /// ### Returns
    /// - A `Matrix` instance containing the transposed matrix
    pub fn transpose(&mut self) -> Matrix<T> {
        let mut transposed_mat: Vec<Vec<T>> = vec![vec![T::default(); self.rows]; self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_mat[j][i] = self.mat[i][j];
            }
        }

        let transposed_mat: Vec<Arc<[T]>> = transposed_mat
            .into_iter()
            .map(|row| Arc::from(row.into_boxed_slice()))
            .collect();

        Matrix {
            mat: transposed_mat,
            rows: self.cols,
            cols: self.rows,
        }
    }
}
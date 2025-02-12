use std::ops::Range;
use std::sync::Arc;
use crate::number::Number;

/// A struct representing that of a `Matrix` in linear algebra
/// 
/// Contains a computational representation of a `Matrix`, along with
/// core matrix operations, such as shape and determinant
///
/// In Linear Algebra, a `Matrix` is a rectangular array of numbers,
/// symbols, or expressions, arranged in rows and columns
/// 
/// Matrices are used to represent and solve systems of linear equations, perform
/// linear transformations, and more
#[derive(Clone)]
pub struct Matrix<T: Number + Clone> {
    /// Represents a vector of `Arc` atomic reference counted `[i64]` arrays, 
    /// where each represents a row in the matrix
    pub mat: Vec<Arc<[T]>>,
    
    /// Stores the number of rows in the matrix
    pub rows: usize,
    
    /// Stores the number of columns in the matrix
    pub cols: usize,
}

impl<T: Number> Matrix<T> {
    /// Creates a new instance of this `Matrix`
    /// 
    /// ### Returns
    /// - A newly constructed `Matrix` object
    pub fn new() -> Matrix<T> {
        Matrix {
            mat: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }
    
    /// Compute the shape of this `Matrix`
    /// 
    /// ### Returns
    /// - A tuple of two positive integers - `(usize, usize)` - representing
    /// the rows and columns of the matrix
    pub fn shape(&mut self) -> (usize, usize) {
        self.rows = self.mat.len();
        self.cols = if self.rows > 0 {
            self.mat[0].len()
        } else {
            0
        };
        
        (self.rows, self.cols)
    }
    
    /// Compute the determinant of this `Matrix`. 
    /// 
    /// In a `Matrix` with a shape of
    /// `(2, 2)`, a `Matrix`'s determinant is equal to `ad - bc`, which is the difference
    /// between the left diagonal product and the right diagonal product
    /// 
    /// ### Returns
    /// - A `Result` determining whether the determinant could be calculated
    ///     - An `Err` if the `Matrix`'s shape could not be calculated 
    ///     - An `Ok` with the determinant value, if this `Matrix`'s 
    ///     shape is `(2, 2)` - 2 rows and 2 columns
    pub fn determinant(&mut self) -> Result<T, String> {
        if self.shape() == (2, 2) {
            let first = 0;
            let last = self.mat.len() - 1;
            
            let ad = self.mat[first][first] * self.mat[last][last];
            let bc = self.mat[first][last] * self.mat[last][first];
            
            return Ok(ad - bc);
        }
        
        Err("Unable to compute determinant. The shape must be (2, 2)"
            .to_string())
    }


    /// Get a sub-matrix of this `Matrix`
    ///
    /// ### Parameters
    /// - `row_range` - A `Range<usize>` indicating the range of 
    /// rows to extract from this `Matrix`
    /// - `col_range` - A `Range<usize>` indicating the range of
    /// columns to extract from this `Matrix`
    ///
    /// ### Returns
    /// - A `Result` containing whether this `Matrix` could be extracted 
    /// into a sub-matrix or not
    ///     - An `Ok` with the new sub-matrix
    ///     - An `Err` with a custom `String` error message if either or
    ///     both provided ranges were out of bounds
    pub fn sub_matrix(
        &mut self,
        row_range: Range<usize>,
        col_range: Range<usize>
    ) -> Result<Matrix<T>, String> {
        if &row_range.end > &self.rows || &col_range.end > &self.cols {
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
            cols: col_range.len()
        })
    }
}
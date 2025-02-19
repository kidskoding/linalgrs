extern crate num;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Neg;
use std::sync::Arc;
use crate::matrix::Matrix;
use crate::number::Number;

/// `MatrixUtilities` is a utility struct designed to perform
///  various algorithms or operations for `Matrix` instances, including
///  adding, subtracting, multiplying, and computing the row and reduced row
///  echelon form of `Matrix` instances
pub struct MatrixUtilities<T: Number> {
    _marker: PhantomData<T>,
}

impl<T: Number + Neg<Output = T> + num::One> MatrixUtilities<T> {
    /// Appends a `row` to a given `Matrix`, returning a updated `Matrix` instance with the newly
    /// appended row
    ///
    /// ### Parameters
    /// - `matrix` - The `Matrix` needed to add a `row`
    /// - `row`: An `&[i64]` slice
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

    /// Appends multiple `rows` to a given `Matrix`, returning a updated `Matrix` instance
    /// with the newly appended rows
    ///
    /// ### Parameters
    /// - `matrix`: The `Matrix` needed to add additional `rows`
    /// - `rows`: A slice of `&[i64]` slices representing a series of rows to add to a given `Matrix`
    ///
    /// ### Returns
    /// - An updated `Matrix` object that adds all `rows` to this
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
    
    /// Computes the row echelon form for the given `matrix` and returns the result as an updated 
    /// `Matrix` instance
    /// 
    /// ### Parameters
    /// - `matrix`: The `Matrix` needed to compute the row echelon form
    /// 
    /// ### Returns
    /// - A `Matrix` instance containing the given `matrix` in row echelon form
    pub fn row_echelon_form(mut matrix: Matrix<T>) -> Matrix<T> {
        let rows = matrix.rows;
        let cols = matrix.cols;
        
        for i in 0..rows {
            let pivot = matrix.mat[i][i];
            if pivot != T::default() {
                for c in 0..cols {
                    let row = Arc::make_mut(&mut matrix.mat[i]);
                    row[c] = row[c] / pivot;
                    if row[c] == -T::default() {
                        row[c] = T::default();
                    }
                }
            }
            
            let pivot_row = Arc::clone(&matrix.mat[i]);
            
            for j in (i + 1)..rows {
                let scale_factor = matrix.mat[j][i];
                let (_, lower) = matrix.mat.split_at_mut(j);
                let row_j = Arc::make_mut(&mut lower[0]);
                
                for c in 0..cols {
                    row_j[c] = row_j[c] - scale_factor * pivot_row[c];
                    if row_j[c] == -T::default() {
                        row_j[c] = T::default();
                    }
                }
            }
        }
        
        matrix
    }
    
    /// Computes the reduced row echelon form (RREF) for the given `matrix` and returns the result
    /// as a updated `Matrix` instance
    /// 
    /// ### Parameters
    /// - `matrix`: The `Matrix` needed to compute the reduced row echelon form
    /// 
    /// ### Returns
    /// - A `Matrix` instance containing the given `matrix` in reduced row echelon form
    pub fn rref(mut matrix: Matrix<T>) -> Matrix<T> {
        let rows = matrix.rows;
        let cols = matrix.cols;
        
        for i in 0..rows {
            let pivot = matrix.mat[i][i];
            if pivot != T::default() {
                for c in 0..cols {
                    let row = Arc::make_mut(&mut matrix.mat[i]);
                    row[c] = row[c] / pivot;
                }
            }
            
            let pivot_row = Arc::clone(&matrix.mat[i]);
            
            for j in (i + 1)..rows {
                let factor = matrix.mat[j][i];
                let pivot_row_clone = Arc::clone(&pivot_row);
                let (_, lower) = matrix.mat.split_at_mut(j);
                let row_j = Arc::make_mut(&mut lower[0]);
                
                for c in 0..cols {
                    row_j[c] = row_j[c] - factor * pivot_row_clone[c];
                }
            }
        }
        
        for i in (0..rows).rev() {
            for j in (0..i).rev() {
                let factor = matrix.mat[j][i];
                let pivot_row_clone = Arc::clone(&matrix.mat[i]);
                let (_, lower) = matrix.mat.split_at_mut(j);
                let row_j = Arc::make_mut(&mut lower[0]);
                
                for c in 0..cols {
                    row_j[c] = row_j[c] - factor * pivot_row_clone[c];
                }
            }
        }
        
        matrix
    }
    
    /// Performs the [Gaussian Elimination](https://en.wikipedia.org/wiki/Gaussian_elimination)
    /// technique on a given `matrix` to solve for its system of equations' missing variables 
    /// (e.g. x, y, and z)
    /// 
    /// ### Parameters
    /// - `matrix`: The `Matrix` to perform Gaussian Elimination on
    /// 
    /// ### Returns
    /// - A `Result` based on whether the matrix had a solution
    ///     - An `Err` with an enclosed `String` representing the error state of solving the `matrix`
    ///       using Gaussian Elimination (i.e. no solution or infinitely many solutions)
    ///     - An `Ok` enclosed with a `HashMap` containing each variable name 
    ///       mapped to a value with its solution
    pub fn gaussian_elimination(mut matrix: Matrix<T>) -> Result<HashMap<char, T>, String> {
        matrix = MatrixUtilities::rref(matrix);
        let mut pivot_vars = HashMap::new();
        
        for i in 0..matrix.rows {
            let pivot = matrix.mat[i][i];
            
            if pivot != T::default() {
                pivot_vars.insert(('a' as u8 + i as u8) as char, matrix.mat[i][matrix.cols - 1]);
            } else if matrix.mat[i][matrix.cols - 1] != T::default() {
                return Err("No solution exists for the given matrix.".to_string());
            }
        }

        for i in 0..matrix.rows {
            if matrix.mat[i].iter().all(|&x| x == T::default()) {
                return Err("Infinitely many solutions exist for the given matrix.".to_string());
            }
        }
        
        Ok(pivot_vars)
    }

    /// Adds two `Matrix` instances together and returns a new `Matrix` representing
    /// their sum
    ///
    /// ### Parameters
    /// - `a`: One `Matrix` operand addend
    /// - 'b': Another 'Matrix' operand addend
    ///
    /// ### Returns
    /// - A `Result` based on whether the two matrices were added or not 
    ///     - An `Err` if the two matrices are different shapes
    ///     - An `Ok` wrapped inside a `Matrix` instance that represents the sum
    ///       of the two matrices `a` and `b`
    pub fn add(mut a: Matrix<T>, mut b: Matrix<T>) -> Result<Matrix<T>, String> {
        if a.shape() != b.shape() {
            return Err("Cannot add the two matrices because 
                their shapes are unequal!".to_string())
        }
       
        let mut result = Vec::new();

        for r in 0..a.rows {
            let mut new_row = Vec::new();
            for c in 0..a.cols {
                new_row.push(a.mat[r][c] + b.mat[r][c]);
            }
            result.push(Arc::from(new_row.as_slice()));
        }

        Ok(Matrix {
            mat: result,
            rows: a.rows,
            cols: a.cols,
        })
    }

    /// Subtracts two `Matrix` instances together and returns a new `Matrix` representing
    /// their difference
    ///
    /// ### Parameters
    /// - `a`: A `Matrix` instance that will be one of the operands
    /// - 'b': Another 'Matrix' instance that will be the second operand to subtract from
    ///
    /// ### Returns
    /// - An `Result` based on whether the two matrices were added 
    ///   - An `Err` value when the two matrices have different shapes
    ///   - An `Ok` value wrapped with a `Matrix` instance that represents the difference
    ///     of the two matrices `a` and `b`
    pub fn subtract(mut a: Matrix<T>, mut b: Matrix<T>) -> Result<Matrix<T>, String> {
        if a.shape() != b.shape() {
            return Err("Cannot add the two matrices because 
                their shapes are unequal!".to_string())
        }
       
        let mut result = Vec::new();

        for r in 0..a.rows {
            let mut new_row = Vec::new();
            for c in 0..a.cols {
                new_row.push(a.mat[r][c] - b.mat[r][c]);
            }
            result.push(Arc::from(new_row.as_slice()));
        }

        Ok(Matrix {
            mat: result,
            rows: a.rows,
            cols: a.cols,
        })
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

    /// Multiplies two `Matrix` instances together and returns their product as a
    /// new `Matrix` object
    ///
    /// ### Parameters
    /// - `a`: One `Matrix` operand to be multiplied
    /// - 'b': Another `Matrix` operand to be multiplied
    ///
    /// ### Returns
    /// - A `Result` based on whether the two matrices were multiplied
    ///     - An `Err` if the columns of `Matrix` a does not equal the rows of `Matrix` b
    ///     - An `Ok` wrapped inside a `Matrix` object that represents the product between two
    ///       matrices
    pub fn multiply(a: Matrix<T>, b: Matrix<T>) -> Result<Matrix<T>, String> {    
        if a.cols != b.rows {
            return Err("The columns of matrix a do not 
                equal the rows of matrix b!".to_string());
        }

        let mut new_mat = vec![];
        for r in 0..a.rows {
            let mut new_row = vec![];
            for c in 0..b.cols {
                let mut sum = T::default();
                for k in 0..a.cols {
                    sum += a.mat[r][k] * b.mat[k][c];
                }
                new_row.push(sum);
            }
            new_mat.push(Arc::from(new_row.as_slice()));
        }

        Ok(Matrix {
            mat: new_mat.clone(),
            rows: new_mat.clone().len(),
            cols: new_mat[0].clone().len(),
        })
    }

    /// Gets the dot product of two matrices `a` and `b`
    /// 
    /// ### Parameters
    /// - `a`: One of the `Matrix` instance operands
    /// - `b`: Another `Matrix` instance operand
    ///
    /// ### Returns
    /// - A `Result` based on whether there is a 
    ///   valid dot product for matrices `a` and `b`
    ///     - An `Err` value if the columns of `Matrix` a` do not equal the 
    ///       rows of `Matrix` b`
    ///     - An `Ok` wrapped in a T generic value, representing the 
    ///       dot product
    pub fn dot(a: Matrix<T>, b: Matrix<T>) -> Result<T, String> {
        if a.cols != b.rows {
            return Err("Cannot get the dot product: The number of columns in A \
                must match the number of rows in B.".to_string());
        }
        if !(a.rows == 1 && b.cols == 1) {
            return Err("Dot product is only valid for a 
                row vector and a column vector.".to_string());
        }

        let mut sum = T::default();
        for i in 0..a.cols {
            sum += a.mat[0][i] * b.mat[i][0];
        }

        Ok(sum)
    }
}

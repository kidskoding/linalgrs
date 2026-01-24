use std::{collections::HashSet, marker::PhantomData, ops::Neg, sync::Arc};

use crate::{matrix::Matrix, matrix_utilities::MatrixUtilities, number::Number, vector::Vector};

pub struct VectorUtilities<T> {
    _marker: PhantomData<T>,
}

impl<T: Number + Neg<Output = T>> VectorUtilities<T> {
    /// Checks if a set of vectors are linearly independent
    ///
    /// A set of vectors are linearly independent if and when none of the vectors  
    /// in such set can be written as a linear combination of the others
    ///
    /// Mathematically speaking, a set of vectors are linearly independent 
    /// if and when
    ///
    /// $$
    ///     c_1\hat{v}_1 + c_2\hat{v}_2 + \dots + c_k\hat{v}_k = 0
    /// $$
    ///
    /// where $c_1 = c_2 = c_k = 0$
    ///
    /// ### Parameters
    /// - `vectors`: A reference to a `HashSet` of `Vector`s whose rank is to be calculated in
    /// order to determine linear indepedence
    ///
    /// ### Returns
    /// - `true` - If the set of vectors are linearly independent
    /// - `false` - If the set of vectors are not linearly independent
    pub fn is_linear_independent(vectors: &HashSet<Vector<T>>) -> bool {
        let n = vectors.len();

        let mut matrix_rows = Vec::new();
        let mut cols = 0;

        for v in vectors {
            cols = v.len;
            matrix_rows.push(Arc::clone(&v.data))
        }

        let mut mat = Matrix {
            rows: n,
            cols,
            mat: matrix_rows,
        };

        let rank = MatrixUtilities::rank(&mut mat);
        rank == n
    } 
} 

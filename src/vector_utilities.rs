use std::{collections::HashSet, marker::PhantomData, ops::Neg, sync::Arc};

use color_eyre::eyre::eyre;

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
    pub fn is_linear_independent(vectors: &HashSet<Vector<T>>) -> color_eyre::Result<bool> {
        if vectors.is_empty() {
            return Ok(true)
        }

        let mut iter = vectors.iter();
        let first_vector = iter.next()
            .unwrap();

        for v in iter {
            if v.len != first_vector.len {
                return Err(
                    eyre!("Vectors have different dimensions; They belong to different vector spaces! Linear independence is undefined!")
                );
            }
        }

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
        Ok(rank == n)
    }

    pub fn inner_product(u: &Vector<T>, v: &Vector<T>) -> color_eyre::Result<T> {
         if u.len != v.len {
             return Err(eyre!("Vectors have different dimensions; They belong to different vector spaces! The inner product is undefined!"));
         }

         let mut sum = T::default();
         for i in 0..u.len {
             sum += u.data[i] * v.data[i];
         }

         Ok(sum)
    }
} 

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
    /// - An `Ok` result, returning `true` if the set of vectors are linearly
    /// independent and `false` if the set of vectors are not linearly independent (linearly
    /// dependent)
    /// - An `Err` result, stating that the vectors belong to varying vector spaces! (typically different dimension!) true` - If the set of vectors are linearly independent
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

    /// Computes the inner product for two vectors, `u` and `v` 
    ///
    /// The inner product is a method used to multiply two vectors that exist 
    /// within the same vector space, resulting in a scalar value 
    /// that defines the alignment and magnitude 
    /// amongst the two vectors (supposedly `u` and `v`) in vector space 
    ///
    /// This value can either be > 0 (the vectors overlap), < 0 (vectors point 
    /// in opposite directions), or 0 (perpendicular)
    ///
    /// Mathematically speaking, the inner product (denoted as <u, v>) of two vectors `u` and `v`,
    /// where $u = (u_1, u_2, \dots, u_n)$ and $v = (v_1, v_2, \dots, v_n)$ is 
    /// defined as 
    /// 
    /// $$
    ///     <u, v> = u_1v_1 + u_2v_2 + \dots + u_nv_n
    /// $$
    ///
    /// ### Parameters
    /// - `u`: A reference to a `Vector` that acts as the first input to calculate the inner product 
    /// - `v`: A reference to a `Vector` that acts as the second input to calculate the inner product
    ///
    /// ### Returns
    /// - An `Ok` variant of a Result, returning a type `T` bounded by trait `Number` representing the scalar value of the inner product between vectors `u` and `v`
    /// - An `Err` variant of a Result if vectors `u` and `v` do not belong to the same dimension
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

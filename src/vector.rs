use std::sync::Arc;

use crate::number::Number;

/// A struct representing a `Vector` in linear algebra 
/// 
/// In Linear Algebra, a `Vector` is a quantity that has both magnitude and direction,
/// typically represented as a 1D array of numbers. This implementation uses an `Arc<[T]>`
/// to allow for efficient sharing and thread-safety
///
/// Vectors are fundamental in representing coordinates, velocities, and 
/// as the building blocks for matrix operations
#[derive(Clone, Debug)]
pub struct Vector<T: Number + PartialEq> {
    /// Represents an `Arc` atomic reference counting array,
    /// storing the elements of the `Vector`
    pub data: Arc<[T]>,

    /// Stores the number of elements (magnitude/dimension) in the vector
    pub len: usize
}

impl <T: Number + PartialEq> Vector<T> {
    /// Creates a new `Vector` from a standard type `Vec<T>`.
    ///
    /// This constructor converts the input vector into a boxed slice and then
    /// into an `Arc` to ensure the data is immutable and shareable.
    ///
    /// ### Parameters
    /// - `elements`: A `Vec<T>` containing the elements to be stored in the vector.
    ///
    /// ### Returns
    /// - A new `Vector` instance containing the provided elements.
    pub fn new(elements: Vec<T>) -> Self {
        let len = elements.len();
        
        Self {
            data: Arc::from(elements.into_boxed_slice()),
            len
        }
    }
}

/// A macro to create a `Vector` from a list of elements.
///
/// This macro allows you to create a `Vector` instance by specifying its elements
/// in a comma-separated list.
///
/// ### Parameters
/// - `$($elem:expr),*`: A comma-separated list of expressions to be elements of the vector.
///
/// ### Returns
/// - A `Vector` instance containing the specified elements.
#[macro_export]
macro_rules! vector {
    ($(elem:expr),* $(,)?) => {
        let mut temp_vec = Vec::new();
        $(
            temp_vec.push($elem)
        )*

        Vector::new(temp_vec)
    };
}

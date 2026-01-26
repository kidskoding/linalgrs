use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::{fmt::Display, sync::Arc};

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
    pub fn new(elements: &[T]) -> Self {
        let len = elements.len();
        
        Self {
            data: Arc::from(elements),
            len
        }
    }
}

impl<T: Number + PartialEq + Hash> Hash for Vector<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl <T: Number + PartialEq> Eq for Vector<T> {}

impl <T: Number + num::One> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data == other.data
    } 
}

impl <T: Number + num::One> Default for Vector<T> {
    fn default() -> Self {
        Vector {
            data: Arc::new([]), 
            len: 0
        } 
    }
} 

impl<T: Number + Display> Display for Vector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut curr_line = String::new();
        curr_line.push('|');

        for num in self.data.iter() {
            curr_line.push(' ');
            curr_line.push_str(&format!("{}", num));
        }

        curr_line.push_str(" |");
        write!(f, "{}", curr_line)
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
    [$($elem:expr),* $(,)?] => {
        $crate::vector::Vector::new(&[$($elem),*])
    };
}

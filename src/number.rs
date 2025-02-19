use std::ops::{Add, Sub, Mul, Div, 
               AddAssign, SubAssign, MulAssign, DivAssign};

/// A Number trait to restrict a `Matrix`'s `T` generic to only signed numeric types
pub trait Number: 
    Add<Output = Self>
    + Sub<Output = Self> 
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Copy 
    + Default 
    + PartialEq {}

impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for i128 {}
impl Number for f32 {}
impl Number for f64 {}
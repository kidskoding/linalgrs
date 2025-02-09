use std::ops::{Add, Sub, Mul, MulAssign};

/// A Number trait to restrict a `Matrix`'s `T` generic to only numeric types
pub trait Number: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + MulAssign + Copy {}

impl Number for u8 {}
impl Number for u16 {}
impl Number for u32 {}
impl Number for u64 {}
impl Number for u128 {}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for i128 {}
impl Number for f32 {}
impl Number for f64 {}
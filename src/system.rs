use std::ops::{AddAssign, Div, DivAssign, Mul, SubAssign};
use std::sync::Arc;
use crate::matrix::Matrix;
use crate::number::Number;

pub struct System<T: Number + AddAssign
    + SubAssign + Mul<Output = T>
    + Div<Output = T> + DivAssign 
    + Copy + Default + PartialOrd> {
    
    pub coefficients: Matrix<T>,
    pub variables: Vec<T>,
    pub constants: Vec<T>,
}
impl<T: Number 
    + Clone + Default 
    + SubAssign + DivAssign + Div<Output = T>
    + AddAssign + PartialOrd> System<T> {
    pub fn gaussian_elimination(system: &mut System<T>) -> Result<Vec<T>, String> {
        let n = system.coefficients.rows;
        if n != system.coefficients.rows || system.coefficients.cols != 1 {
            return Err("The number of rows in this coefficient matrix must equal \
                the number of rows in the constants matrix, and the constants matrix \
                must have exactly one column".to_string())
        }
        
        let mut augmented = system.coefficients.clone();
        for i in 0..n {
            let mut row = Vec::from(&*system.coefficients.mat[i]);
            row.push(system.constants[i]);
            augmented.mat[i] = Arc::from(row.as_slice());
        }
        augmented.cols += 1;
        
        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented.mat[k][i] > augmented.mat[max_row][i] {
                    max_row = k;
                }
            }

            augmented.mat.swap(i, max_row);

            for k in (i + 1)..n {
                let factor = augmented.mat[k][i] / augmented.mat[i][i];
                for j in i..augmented.cols {
                    augmented.mat[k][j] -= factor * augmented.mat[i][j];
                }
            }
        }

        let mut solution = vec![T::default(); n];
        for i in (0..n).rev() {
            let mut sum = augmented.mat[i][n];
            for j in (i + 1)..n {
                sum -= augmented.mat[i][j] * solution[j];
            }
            solution[i] = sum / augmented.mat[i][i];
        }

        Ok(solution)
    }
}
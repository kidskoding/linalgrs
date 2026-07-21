# linalgrs TODO

Roadmap to a full linear algebra library. Checked items already exist in the codebase.

1. **Matrix (core)**
    - [x] 1.1 - Create a `Matrix` type that represents a matrix space
    - [x] 1.2 - Compute the shape of the matrix
    - [x] 1.3 - Compute the determinant of the matrix
    - [x] 1.4 - Code a function that retrieves a sub matrix of a matrix
    - [x] 1.5 - Code the addition, subtraction, and multiplication operators for the matrices
    - [x] 1.6 - Create a function that multiplies a matrix by a scalar constant value
    - [x] 1.7 - Code dot product for two matrices
    - [x] 1.8 - Compute row echelon form and rref for a matrix
    - [x] 1.9 - Solve for pivots in a matrix using gaussian elimination
    - [x] 1.10 - Calculate the transpose of a matrix
    - [x] 1.11 - Calculate the inverse of a matrix
    - [x] 1.12 - Gauss-Jordan elimination
    - [x] 1.13 - Identity matrix constructor
    - [x] 1.14 - Append row / append multiple rows
    - [x] 1.15 - `matrix!` macro for literal construction
    - [x] 1.16 - `Display` pretty-printing
    - [x] 1.17 - `PartialEq` for matrices
    - [x] 1.18 - Rank of a matrix
    - [ ] 1.19 - Zeros / ones / fill / diagonal / random constructors
    - [ ] 1.20 - Trace of a matrix
    - [ ] 1.21 - Matrix power (`A^n`) and exponential
    - [ ] 1.22 - Element-wise (Hadamard) product
    - [ ] 1.23 - Kronecker (tensor) product
    - [ ] 1.24 - Row/column swap, scale, and combine elementary operations
    - [ ] 1.25 - Operator overloads (`+ - * /`, `Index`) instead of util calls
    - [ ] 1.26 - Block / augmented matrix construction and splitting
    - [ ] 1.27 - Matrix norms (Frobenius, 1-norm, inf-norm, 2-norm)

2. **Vector**
    - [x] 2.1 - `Vector` type backed by `Arc<[T]>`
    - [x] 2.2 - Inner / dot product of two vectors
    - [ ] 2.3 - `PartialEq` / equality for vectors
    - [ ] 2.4 - `ColumnVector` / `RowVector` distinction (matrix-backed)
    - [ ] 2.5 - Vector addition, subtraction, scalar multiply
    - [ ] 2.6 - Vector norms (L1, L2, L-inf, p-norm)
    - [ ] 2.7 - Normalize / unit vector
    - [ ] 2.8 - Cross product (3D)
    - [ ] 2.9 - Outer product (vector -> matrix)
    - [ ] 2.10 - Projection of one vector onto another
    - [ ] 2.11 - Angle between vectors / cosine similarity

3. **Decompositions**
    - [x] 3.1 - LU decomposition
    - [ ] 3.2 - PLU (LU with partial pivoting)
    - [ ] 3.3 - QR decomposition (Gram-Schmidt / Householder)
    - [ ] 3.4 - Cholesky decomposition
    - [ ] 3.5 - Eigenvalue decomposition
    - [ ] 3.6 - Singular value decomposition (SVD)
    - [ ] 3.7 - Schur decomposition

4. **Solvers & Systems**
    - [x] 4.1 - Solve `Ax = b` via Gaussian elimination
    - [ ] 4.2 - Forward / back substitution (using LU)
    - [ ] 4.3 - Least squares solver (`min ||Ax - b||`)
    - [ ] 4.4 - Determinant / inverse via LU (numerically stable path)
    - [ ] 4.5 - Iterative solvers (Jacobi, Gauss-Seidel, conjugate gradient)

5. **Subspaces & Properties**
    - [x] 5.1 - Linear independence of a set of vectors
    - [x] 5.2 - Orthogonality check between two vectors
    - [ ] 5.3 - Null space / kernel basis
    - [ ] 5.4 - Column space / row space basis
    - [ ] 5.5 - Rank-nullity computation
    - [ ] 5.6 - Eigenvalues and eigenvectors
    - [ ] 5.7 - Characteristic polynomial
    - [ ] 5.8 - Orthonormal basis (Gram-Schmidt)
    - [ ] 5.9 - Condition number
    - [ ] 5.10 - Definiteness checks (positive/negative definite)

6. **Numeric foundation**
    - [x] 6.1 - `Number` trait abstracting the scalar type
    - [x] 6.2 - `color_eyre` error handling for vector ops
    - [ ] 6.3 - Float tolerance / approximate-equality helpers
    - [ ] 6.4 - Complex number scalar support
    - [ ] 6.5 - Generic over integer vs float behavior (pivoting, division)

7. **Quality & Infrastructure**
    - [ ] 7.1 - Unit tests across all operations
    - [ ] 7.2 - Property-based tests (e.g. `A * A^-1 == I`)
    - [ ] 7.3 - Benchmarks
    - [ ] 7.4 - Doc examples that compile (`cargo test --doc`)
    - [ ] 7.5 - README with usage examples
    - [ ] 7.6 - Publish to crates.io

# Matlab Sparse Matrix

This example demonstrates how to compute the tomographic back-projection system matrix and export it in Matlab .mat format using RTK's [JosephBackProjectionImageFilter](../../documentation/docs/Projectors.md)

## Using in Matlab

Load and analyze the generated sparse matrix:

```matlab
% Load the sparse matrix (stored as variable 'A')
load('backprojection_matrix.mat')

% Visualize the sparsity pattern
spy(A)

% Display matrix statistics
fprintf('Matrix size: %d x %d\n', size(A,1), size(A,2))
fprintf('Non-zero elements: %d\n', nnz(A))
fprintf('Sparsity: %.2f%%\n', 100 * (1 - nnz(A)/(size(A,1)*size(A,2))))
```

## Code

```{literalinclude} MatlabSparseMatrixExample.cxx
:language: c++
```

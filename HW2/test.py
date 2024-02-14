import sympy as sp

# Step 2: Define your square matrix
A = sp.Matrix([
    [0, 1],
    [-2, -2]
])

# Step 3: Calculate the characteristic polynomial
lambda_symbol = sp.symbols('lambda')
char_poly = A.charpoly(lambda_symbol)

# Step 4: Replace lambda with the matrix A and get the Cayley-Hamilton theorem expression
cayley_hamilton_expr = char_poly.as_expr().subs(lambda_symbol, A)

# Step 5: Print the Cayley-Hamilton theorem result
print("Cayley-Hamilton Theorem result:")
print(cayley_hamilton_expr)
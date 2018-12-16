# Linear Regression - Library Free, i.e. no numpy or scipy 
import sys

def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have

        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M

def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix

        :return: a square identity matrix
    """
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0

    return I

def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied

        :return: A copy of the given matrix
    """
    # Section 1: Get matrix dimensions
    rows = len(M); cols = len(M[0])

    # Section 2: Create a new matrix of zeros
    MC = zeros_matrix(rows, cols)

    # Section 3: Copy values of M into the copy
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC

def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x,decimals)+0 for x in row])

def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed

        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0],list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M); cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros_matrix(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

def matrix_addition(A, B):
    """
    Adds two matrices and returns the sum
        :param A: The first matrix
        :param B: The second matrix

        :return: Matrix sum
    """
    # Section 1: Ensure dimensions are valid for matrix addition
    rowsA = len(A); colsA = len(A[0])
    rowsB = len(B); colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix sum
    C = zeros_matrix(rowsA, colsB)

    # Section 3: Perform element by element sum
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] + B[i][j]

    return C

def matrix_subtraction(A, B):
    """
    Subtracts matrix B from matrix A and returns difference
        :param A: The first matrix
        :param B: The second matrix

        :return: Matrix difference
    """
    # Section 1: Ensure dimensions are valid for matrix subtraction
    rowsA = len(A); colsA = len(A[0])
    rowsB = len(B); colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix difference
    C = zeros_matrix(rowsA, colsB)

    # Section 3: Perform element by element subtraction
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] - B[i][j]

    return C

def matrix_multiply(A, B):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix

        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A); colsA = len(A[0])
    rowsB = len(B); colsB = len(B[0])
    if colsA != rowsB:
        raise ArithmeticError(
            'Number of A columns must equal number of B rows.')

    # Section 2: Store matrix multiplication in a new matrix
    C = zeros_matrix(rowsA, colsB)
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C

def multiply_matrices(list):
    """
    Find the product of a list of matrices from first to last
        :param list: The list of matrices IN ORDER

        :return: The product of the matrices
    """
    # Section 1: Start matrix product using 1st matrix in list
    matrix_product = list[0]

    # Section 2: Loop thru list to create product
    for matrix in list[1:]:
        matrix_product = matrix_multiply(matrix_product, matrix)

    return matrix_product

def check_matrix_equality(A, B, tol=None):
    """
    Checks the equality of two matrices.
        :param A: The first matrix
        :param B: The second matrix
        :param tol: The decimal place tolerance of the check

        :return: The boolean result of the equality check
    """
    # Section 1: First ensure matrices have same dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    # Section 2: Check element by element equality
    #            use tolerance if given
    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol == None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j],tol) != round(B[i][j],tol):
                    return False

    return True

def dot_product(A, B):
    """
    Perform a dot product of two vectors or matrices
        :param A: The first vector or matrix
        :param B: The second vector or matrix
    """
    # Section 1: Ensure A and B dimensions are the same
    rowsA = len(A); colsA = len(A[0])
    rowsB = len(B); colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Sum the products 
    total = 0
    for i in range(rowsA):
        for j in range(colsB):
            total += A[i][j] * B[i][j]

    return total

def unitize_vector(vector):
    """
    Find the unit vector for a vector
        :param vector: The vector to find a unit vector for

        :return: A unit-vector of vector
    """
    # Section 1: Ensure that a vector was given
    if len(vector) > 1 and len(vector[0]) > 1:
        raise ArithmeticError(
            'Vector must be a row or column vector.')

    # Section 2: Determine vector magnitude
    rows = len(vector); cols = len(vector[0])
    mag = 0
    for row in vector:
        for value in row:
            mag += value ** 2
    mag = mag ** 0.5

    # Section 3: Make a copy of vector
    new = copy_matrix(vector)

    # Section 4: Unitize the copied vector
    for i in range(rows):
        for j in range(cols):
            new[i][j] = new[i][j] / mag

    return new

def scale_matrix(scaler, M):
    """
    Scale a matrix by a given value
        :param scaler: The value for scaling the matrix
        :param M: The matrix to be scaled

        :return: The scaled matrix
    """
    # Section 1: Make a copy of the given matrix
    new = copy_matrix(M)
    rows = len(new); cols = len(new[0])

    # Section 2: Scale each element of the copied matrix
    for i in range(rows):
        for j in range(cols):
            new[i][j] = new[i][j] * scaler

    return new 

def scale_matrix_by_max(A):
    """
    Scale a matrix by it's largest value
        :param A: The matrix to be scaled

        :return: The scaled matrix
    """
    # Section 1: Find the max value of the matrix
    max = 0
    for row in A:
        for col in row:
            if abs(col) > max:
                max = col

    # Section 2: Create a copy of the matrix A
    new = copy_matrix(A)
    rows = len(new)
    cols = len(new[0])

    # Section 3: Reduce each value of copied matrix by max value
    for i in range(rows):
        for j in range(cols):
            new[i][j] = new[i][j] / max

    return new 

def insert_at_nth_column_of_matrix(column_vector, M, column_num):
    """
    Inserts a new column into an existing matrix
        :param column_vector: The column vector to insert
            IF a value is passed in, a column is created
            with all elements equal to the value
        :param M: The matrix to insert the new column into
        :param column_num: The column index to insert at
            NOTE: index is "zero" based

        :return: The altered matrix
    """
    # Section 1: Obtain matrix dimensions
    rows = len(M); cols = len(M[0])

    # Section 2: If a value has been passed in for column vector ...
    if not isinstance(column_vector,list):
        column_value = column_vector
        column_vector = [] # ... create a column vector ...
        for i in range(rows): # ... full of that value
            column_vector.append([column_value]) 

    # Section 3: IF column vector received, check for correct rows
    if rows != len(column_vector):
        raise ArithmeticError('Column and Matrix rows do NOT match.')

    # Section 4: Insert the column vector values one row at a time
    for i in range(rows):
        M[i].insert(column_num,column_vector[i][0])

    return M

def replace_nth_column_of_matrix(column_vector, M, column_num):
    """
    Replace a column in an existing matrix
        :param column_vector: The new column vector
        :param M: The matrix needing column update
        :param column_num: The location of the column in M

        :return: The matrix with the column updated
    """
    # Section 1: Obtain matrix dimensions
    rows = len(M); cols = len(M[0])
    
    # Section 2: If a value has been passed in for column vector ...
    if not isinstance(column_vector,list):
        column_value = column_vector
        column_vector = [] # ... create a column vector ...
        for i in range(rows): # ... full of that value
            column_vector.append([column_value]) 

    # Section 3: IF column vector received, check for correct rows
    if rows != len(column_vector):
        raise ArithmeticError('Column and Matrix rows do NOT match.')

    # Section 2: Update the specified column
    for i in range(rows):
        M[i][column_num] = column_vector[i][0]

    return M

def check_squareness(A):
    """
    Makes sure that a matrix is square
        :param A: The matrix to be checked.
    """
    if len(A) != len(A[0]):
        raise ArithmeticError("Matrix must be square to inverse.")

def determinant_recursive(A, total=0):
    """
    Find determinant of a square matrix using full recursion
        :param A: the matrix to find the determinant for
        :param total=0: safely establish a total at each recursion level

        :returns: the running total for the levels of recursion
    """
    # Section 1: store indices in list for flexible row referencing
    indices = list(range(len(A)))
    
    # Section 2: when at 2x2 submatrices recursive calls end
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    # Section 3: define submatrix for focus column and call this function
    for fc in indices: # for each focus column, find the submatrix ...
        As = copy_matrix(A) # make a copy, and ...
        As = As[1:] # ... remove the first row
        height = len(As)

        for i in range(height): # for each remaining row of submatrix ...
            As[i] = As[i][0:fc] + As[i][fc+1:] # remove the focus column elements

        sign = (-1) ** (fc % 2) # alternate signs for submatrix multiplier
        sub_det = determinant_recursive(As) # pass submatrix recursively
        total += sign * A[0][fc] * sub_det # total all returns from recursion

    return total

def determinant_fast(A):
    """
    Create an upper triangle matrix using row operations.
        Then product of diagonal elements is the determinant

        :param A: the matrix to find the determinant for

        :return: the determinant of the matrix
    """
    # Section 1: Establish n parameter and copy A
    n = len(A)
    AM = copy_matrix(A)

    # Section 2: Row manipulate A into an upper triangle matrix
    for fd in range(n): # fd stands for focus diagonal
        if AM[fd][fd] == 0: 
            AM[fd][fd] = 1.0e-18 # Cheating by adding zero + ~zero
        for i in range(fd+1,n): # skip row with fd in it.
            crScaler = AM[i][fd] / AM[fd][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    
    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(n):
        product *= AM[i][i] # ... product of diagonals is determinant

    return product

def check_non_singular(A):
    """
    Ensure matrix is NOT singular
        :param A: The matrix under consideration

        :return: determinant of A - nonzero is positive boolean
                  otherwise, raise ArithmeticError
    """
    det = determinant_fast(A)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Singular Matrix!")

def invert_matrix(A, tol=None):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed

        :return: The inverse of the matrix A
    """
    # Section 1: Make sure A can be inverted.
    check_squareness(A)
    check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    IM = copy_matrix(I)

    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]

    # Section 4: Make sure that IM is an inverse of A within the specified tolerance
    if check_matrix_equality(I,matrix_multiply(A,IM),tol):
        return IM
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")

def solve_equations(A, B, tol=None):
    """
    Returns the solution of a system of equations in matrix format.
        :param A: The system matrix

        :return: The solution X where AX = B
    """
    # Section 1: Make sure A can be inverted.
    check_squareness(A)
    check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    BM = copy_matrix(B)

    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    for fd in range(n): # fd stands for focus diagonal
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
        BM[fd][0] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

    # Section 4: Make sure that BM is the solution for X
    if check_matrix_equality(B,matrix_multiply(A,BM),tol):
        return BM
    else:
        raise ArithmeticError("Solution for X out of tolerance.")

def least_squares(X, Y, tol=3):
    """
    Find least squares fit for coefficients of X given Y
        :param X: The input parameters
        :param Y: The output parameters or labels

        :return: The coefficients of X 
                 including the constant for X^0
    """
    # Section 1: If X and/or Y are 1D arrays, make them 2D
    if not isinstance(X[0],list):
        X = [X]
    if not isinstance(type(Y[0]),list):
        Y = [Y]

    # Section 2: Make sure we have more rows than columns
    #            This is related to section 1
    if len(X) < len(X[0]):
        X = transpose(X)
    if len(Y) < len(Y[0]):
        Y = transpose(Y)

    # Section 3: Add the column to X for the X^0, or
    #            for the Y intercept
    for i in range(len(X)):
        X[i].append(1)

    # Section 4: Perform Least Squares Steps
    AT = transpose(X)
    ATA = matrix_multiply(AT, X)
    ATB = matrix_multiply(AT, Y)
    coefs = solve_equations(ATA,ATB,tol=tol)
    
    return coefs


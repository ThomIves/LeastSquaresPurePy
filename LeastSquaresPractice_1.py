import LinearAlgebraPurePython as la 
import matplotlib.pyplot as plt

X = [2,4]
Y = [2,3]
plt.scatter(X,Y)

coefs = la.least_squares(X, Y)
la.print_matrix(coefs)

XLS = [0,1,2,3,4,5]
XLST = la.transpose(XLS)
XLST1 = la.insert_at_nth_column_of_matrix(1,XLST,1)
YLS = la.matrix_multiply(XLST1, coefs)

plt.plot(XLS, YLS)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Pure Python Least Squares Line Fit')
plt.show()

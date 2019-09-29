import LinearAlgebraPurePython as la
import conditioned_data as cd
import sys


# Import conditioned data
X_train = cd.X_train
Y_train = cd.Y_train
X_test = cd.X_test
Y_test = cd.Y_test

# Solve for coefficients
coefs = la.least_squares(X_train, Y_train)
print('Pure Coefficients:')
la.print_matrix(coefs)
print()

# Make a prediction
XLS1 = la.insert_at_nth_column_of_matrix(1, X_test, len(X_test[0]))
YLS = la.matrix_multiply(XLS1, coefs)
YLST = la.transpose(YLS)

# Look at our predictions and the actual values
print('PurePredictions:\n', YLST[0], '\n')

# Compare to sklearn
SKLearnData = [103015.20159796, 132582.27760816, 132447.73845175,
               71976.09851258, 178537.48221056, 116161.24230165,
               67851.69209676, 98791.73374687, 113969.43533013,
               167921.06569551]

print('Delta Between SKLearnPredictions and Pure Predictions:')
for i in range(len(SKLearnData)):
    delta = round(SKLearnData[i], 6) - round(YLST[0][i], 6)
    print('\tDelta for outputs {} is {}'.format(i, delta))

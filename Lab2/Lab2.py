import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kernels import *

# debug mode
# np.random.seed(100)

# define the sizes
# effective ones:A1_X_CENTER,A2_X_CENTER,B_X_CENTER
#
DATA_A_ROW = 10
DATA_A_COL = 2
A1_X_CENTER = 1.5
A1_Y_CENTER = 0.5
A2_Y_CENTER = 0.5
A2_X_CENTER = -1.5

DATA_B_ROW = 20
DATA_B_COL = 2
B_X_CENTER = 0.0
B_Y_CENTER = -0.5

STANDARD_DEVIATION = 0.2

C = np.inf

kernels = [Linear_kernel, Polynomial_kernel, RBF_kernel]
kernel = kernels[1]
# generate the data

# randn generates an array of shape (d0, d1, ..., dn),
# filled with random floats sampled from a univariate “normal” (Gaussian) distribution
# of mean 0 and variance 1

classA = np.concatenate((np.random.randn(DATA_A_ROW, DATA_A_COL) * STANDARD_DEVIATION + [A1_X_CENTER, A1_Y_CENTER],
                         np.random.randn(DATA_A_ROW, DATA_A_COL) * STANDARD_DEVIATION + [A2_X_CENTER, A2_Y_CENTER]))
classB = np.random.randn(DATA_B_ROW, DATA_B_COL) * STANDARD_DEVIATION + [B_X_CENTER, B_Y_CENTER]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# Pij = t_i t_j K(x_i,x_j)
# A suitable kernel function
# kernel was defined in the beginning
P = np.ndarray((N, N))
for pi in range(N):
    for pj in range(N):
        P[pi][pj] = targets[pi] * targets[pj] * kernel(inputs[pi], inputs[pj])


# objective
def objective(the_alpha):
    sum1 = 0
    sum2 = 0
    for oi in range(N):
        for oj in range(N):
            sum1 += 0.5 * the_alpha[oi] * the_alpha[oj] * P[oi][oj]
    sum2 += np.sum(the_alpha)
    return sum1 - sum2


# zerofun
def zerofun(the_alpha):
    return np.dot(the_alpha, targets)


# call minimize
print("number of samples:" + str(N))
# C defined in the beginning
start = np.zeros(N)
B = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}
ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']
if ret['success']:
    print("found a solution\n")
print("alpha:\n")
print(alpha)

# Extract the non-zero α values
print("after find the non-zeros:")
nonzero = []
for i in range(N):
    if 1e-5 <= alpha[i]:
        nonzero.append([alpha[i], inputs[i], targets[i]])
for i in range(N):
    if alpha[i] < 1e-5:
        alpha[i] = 0

svs = []
for i in range(N):
    if 0 < alpha[i] < C - 1e-5:
        svs.append(inputs[i])

# nonzero structure [0]alpha_i [1]x_i [2]t_i
print(nonzero)

# calculate b
sv_sample = nonzero[0]
for n in nonzero:
    if n[0] < C - 1e-5:
        sv_sample = n
        break

b = -1 * sv_sample[2]
for i in range(N):
    b += alpha[i] * targets[i] * kernel(sv_sample[1], inputs[i])
print("calculate b : " + str(b) + "\n")


# indicator function
def indicator(new_sample):
    indicate = -b
    for sv in nonzero:
        indicate += sv[0] * sv[2] * kernel(sv[1], new_sample)
    return indicate


# show the data
plt.scatter([p[0] for p in classA], [p[1] for p in classA], c='blue', label='classA')
plt.scatter([p[0] for p in classB], [p[1] for p in classB], c='red', label='classB')
plt.scatter([p[0] for p in svs], [p[1] for p in svs], c='green', label='vector')
plt.legend()
plt.title('dataset_distribution')
plt.axis('equal')  # Force same scale on both axes
plt.savefig('dataset_distribution.png')  # Save a copy in a file
# plt.show()  # Show the plot on the screen


# plotting the decision boundary

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.ndarray((xgrid.shape[0], ygrid.shape[0]))
for xi in range(xgrid.shape[0]):
    for xj in range(ygrid.shape[0]):
        grid[xj][xi] = indicator(np.array([xgrid[xi], ygrid[xj]]))

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.savefig('SVM.png')
plt.show()

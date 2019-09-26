import numpy as np
import random
from matplotlib import pyplot as plt

# debug mode
np.random.seed(100)

# define the sizes

DATA_A_ROW = 10
DATA_A_COL = 2
DATA_A1_UP = 1.5
DATA_A1_DOWN = 0.5
DATA_A2_UP = 0.5
DATA_A2_DOWN = -1.5

DATA_B_ROW = 20
DATA_B_COL = 2
DATA_B_UP = 0.0
DATA_B_DOWN = -0.5

STANDARD_DEVIATION = 0.2

# generate the data

# randn generates an array of shape (d0, d1, ..., dn),
# filled with random floats sampled from a univariate “normal” (Gaussian) distribution
# of mean 0 and variance 1

classA = np.concatenate((np.random.randn(DATA_A_ROW, DATA_A_COL) * STANDARD_DEVIATION + [DATA_A1_DOWN, DATA_A1_UP],
                         np.random.randn(DATA_A_ROW, DATA_A_COL) * STANDARD_DEVIATION + [DATA_A2_DOWN, DATA_A2_UP]))
classB = np.random.randn(DATA_B_ROW, DATA_B_COL) * STANDARD_DEVIATION + [DATA_B_DOWN, DATA_B_UP]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# show the data
plt.scatter([p[0] for p in classA], [p[1] for p in classA], c='blue', label='classA')
plt.scatter([p[0] for p in classB], [p[1] for p in classB], c='red', label='classB')
plt.legend()
plt.title('dataset_distribution')
plt.axis('equal')  # Force same scale on both axes
plt.savefig('dataset_distribution.png')  # Save a copy in a file
plt.show()  # Show the plot on the screen

from dtree import *
import monkdata as m
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from drawtree_qt5 import drawTree
import time


def myentropy(a):
    return -a * log2(a)


e1 = entropy(m.monk1)
e2 = entropy(m.monk2)
e3 = entropy(m.monk3)

print("\nAssignment 1 :")
print("Entropy of MONK-1", e1)
print("Entropy of MONK-2", e2)
print("Entropy of MONK-3", e3)

print("\nAssignment 2 :")
print("Bernoulli distribution q & 1-q\n")
x = np.arange(0.01, 1, 0.01)
y = np.arange(0.01, 1, 0.01)
for i in range(99):
    a = (i + 1) / 100.0
    p = myentropy(a) + myentropy(1 - a)
    y[i] = p

fig = plt.figure(figsize=(15, 5))
p1 = fig.add_subplot(121)
p1.plot(x, y)
p1.set(title="Bernoulli distribution", xlabel="p", ylabel="Entropy")

x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(x, y)
Z = -X * np.log(X) - Y * np.log(Y) - (1 - X - Y) * np.log(1 - X - Y)

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(10, 60)
# plt.show()
plt.savefig("entropy.png")

print("\nAssignment 3 :")
gainMatrix = np.zeros((3, 6))
monk = [m.monk1, m.monk2, m.monk3]

for i in range(3):
    for j in range(6):
        gainMatrix[i][j] = averageGain(monk[i], m.attributes[j])

print(gainMatrix)
print("best attributes:")
for i in monk:
    print(bestAttribute(i, m.attributes))

print("\nAssignment 4 :")
for i in range(3):
    ba = bestAttribute(monk[i], m.attributes)
    print("MONK-" + str(i + 1))
    for v in ba.values:
        subset = select(monk[i], ba, v)
        print("value = " + str(v) + " : " + "%.16f" % (entropy(subset)) + " " + "%.2f" % (
                100 * len(subset) / len(monk[i])) + "%")

print("\nAssignment 5 :")
print("Error rate:")
monktest = [m.monk1test, m.monk2test, m.monk3test]
t = []
for i in range(3):
    t.append(buildTree(monk[i], m.attributes))
    check_train = check(t[i], monk[i])
    check_test = check(t[i], monktest[i])
    print("check on training set:%.16f\t check on test set:%.16f" % (1 - check_train, 1 - check_test))

print("Trees:")

for tree in t:
    print(tree)

# drawTree(t[2])

print("\nAssignment 6 :")
print("For one try:")
monk_train = []
monk_val = []
best_trees = []
check_val_before = []
check_test_before = []
for i in range(3):
    mt, mv = partition(monk[i], 0.6)
    monk_train.append(mt)
    monk_val.append(mv)
    t = buildTree(monk_train[i], m.attributes)
    t_check = check(t, monk_val[i])
    check_test_before.append(check(t, monktest[i]))
    check_val_before.append(t_check)  # check_before
    flag = 1
    while flag:
        flag = 0
        trees = allPruned(t)
        for tree in trees:
            c = check(tree, monk_val[i])
            if c > t_check:
                t_check = c
                t = tree
                flag = 1

    best_trees.append(t)

for i in range(3):
    check_train = check(best_trees[i], monk_train[i])
    check_val = check(best_trees[i], monk_val[i])
    check_test = check(best_trees[i], monktest[i])
    print("check on training set:%.16f\t check on validation set:%.16f\\%.16f\t check on test set:%.16f\\%.16f" % (
        1 - check_train, 1 - check_val_before[i], 1 - check_val, 1 - check_val_before[i], 1 - check_test))

print("Assignment 7 :")

n_iter = 1000
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# MONK-1
err_avg1 = []
for fraction in fractions:
    err_sum = 0
    for i in range(n_iter):
        # one loop
        mt, mv = partition(monk[0], fraction)
        t = buildTree(mt, m.attributes)
        t_check = check(t, mv)
        flag = 1
        while flag:
            flag = 0
            trees = allPruned(t)
            for tree in trees:
                c = check(tree, mv)
                if c > t_check:
                    t_check = c
                    t = tree
                    flag = 1
        err_sum = err_sum + 1 - check(t, monktest[0])
    err_avg1.append(err_sum / n_iter)

# MONK-3
err_avg2 = []
for fraction in fractions:
    err_sum = 0
    for i in range(n_iter):
        # one loop
        mt, mv = partition(monk[2], fraction)
        t = buildTree(mt, m.attributes)
        t_check = check(t, mv)
        flag = 1
        while flag:
            flag = 0
            trees = allPruned(t)
            for tree in trees:
                c = check(tree, mv)
                if c > t_check:
                    t_check = c
                    t = tree
                    flag = 1
        err_sum = err_sum + 1 - check(t, monktest[2])
    err_avg2.append(err_sum / n_iter)
print("\nMONK-1:")
print(err_avg1)
print("\nMONK-3:")
print(err_avg2)
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax1.plot(fractions, err_avg1, c='b', marker="s", label="MONK-1")
ax1.plot(fractions, err_avg2, c='r', marker="o", label="MONK-3")
ax1.set_title("EER-Fraction")
ax1.set_xlabel("fractions")
ax1.set_ylabel("error rate")
plt.legend(loc='upper right');
# plt.show()
plt.savefig("EER_fraction.png")
print("finished!")

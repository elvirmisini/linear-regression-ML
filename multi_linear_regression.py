# This Python script is using multiple linear regression to predict the price of a house based on 
# its size (in square feet) and the number of bedrooms it has.

import numpy as np
import matplotlib.pyplot as plt

# This function normalizes the features in your dataset. Each feature in the dataset is 
# subtracted from its mean and then divided by its standard deviation.
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

print('Loading data ...\n')
data = np.loadtxt('ex1data2.txt', delimiter=',', skiprows=1)
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

print('\tFirst 10 examples from the dataset:')
for i in range(10):
    print('\t x =', X[i, :], ', y =', y[i])

input('Program paused. Press enter to continue.\n')

print('Normalizing Features ...\n')
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones((m, 1)), X), axis=1)

print('\tFirst 10 examples from the normalized features:')
for i in range(10):
    print('\t x =', X[i, :])

input('Program paused. Press enter to continue.\n')

print('Running gradient descent ...\n')
alpha = 0.01
num_iters = 400
theta = np.zeros(3)

# This function computes the cost of using theta as the parameter for linear regression to 
# fit the data points in X and y.
def computeCostMulti(X, y, theta):
    m = len(y)
    J = 1 / (2 * m) * np.sum((np.dot(X, theta) - y) ** 2)
    return J

# This function performs gradient descent to learn theta parameters. It returns an updated 
# theta vector and a record of the cost function over the number of iterations.
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for iter in range(num_iters):
        delta = (1 / m) * np.dot(X.T, np.dot(X, theta) - y)
        theta = theta - alpha * delta
        J_history[iter] = computeCostMulti(X, y, theta)
    
    return theta, J_history

theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.plot(np.arange(1, num_iters + 1), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('\tTheta computed from gradient descent:')
print('\t', theta)

d = np.array([1, 1650, 3])
d = (d[1:] - mu) / sigma
d = np.concatenate(([1], d))
price = np.dot(d, theta)

print('\n\tPredicted price of a 1650 sq-ft, 3 br house (using gradient descent):')
print('\t$', price)

input('Program paused. Press enter to continue.\n')

print('Solving with normal equations...\n')

X = np.concatenate((np.ones((m, 1)), data[:, 0:2]), axis=1)

# This function computes the closed-form solution to linear regression using the normal equations.
def normalEqn(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# The normal equation is a mathematical equation that gives the result directly. It provides 
# an analytical solution to the linear regression problem as opposed to the numerical solution provided by gradient descent.
theta = normalEqn(X, y)

print('\tTheta computed from the normal equations:')
print('\t', theta)

d = np.array([1, 1650, 3])
price = np.dot(d, theta)

print('\n\tPredicted price of a 1650 sq-ft, 3 br house (using normal equations):')
print('\t$', price)

import numpy as np

def featureNormalize(X):
    # Normalizes the features in X
    # Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.

    X_norm = X.copy()
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    for iter in range(X.shape[1]):
        if sigma[iter] != 0:
            X_norm[:, iter] = (X[:, iter] - mu[iter]) / sigma[iter]
        else:
            X_norm[:, iter] = 0

    return X_norm, mu, sigma



import numpy as np

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # Performs gradient descent to learn theta
    # Updates theta by taking num_iters gradient steps with learning rate alpha

    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        delta = (1/m * np.dot(X, theta) - y).T.dot(X).T
        theta = theta - alpha * delta
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history

def computeCostMulti(X, y, theta):
    # Compute cost for linear regression with multiple variables
    # Returns the cost of using theta as the parameter for linear regression to fit the data points in X and y

    m = len(y)  # number of training examples
    J = 1 / (2 * m) * np.sum(np.square(np.dot(X, theta) - y))
    
    return J


def normalEqn(X, y):
    # Computes the closed-form solution to linear regression
    # Returns the calculated theta using the normal equations

    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    return theta


data = np.loadtxt('ex1data2.txt', delimiter=',', skiprows=1)

# Separate the features (X) and target variable (y)
X = data[:, :-1]
y = data[:, -1]

# Add a column of ones to X for the intercept term
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# Normalize the features
X_norm, mu, sigma = featureNormalize(X)

# Set the initial theta values
theta = np.zeros(X_norm.shape[1])

# Set the learning rate and number of iterations for gradient descent
alpha = 0.01
num_iters = 400

# Perform gradient descent
theta_gd, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

# Calculate theta using the normal equations
theta_ne = normalEqn(X, y)

# Print the learned theta values
print("Theta values using gradient descent:", theta_gd)
print("Theta values using normal equations:", theta_ne)
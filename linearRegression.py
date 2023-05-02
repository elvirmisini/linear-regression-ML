# import numpy as np

# identity_matrix = np.identity(5)
# print(identity_matrix)

# import csv

# input_file = 'data.txt'
# output_file = 'data.csv'

# # Open input and output files
# with open(input_file, 'r') as in_file, open(output_file, 'w', newline='') as out_file:
#     # Create CSV writer object
#     writer = csv.writer(out_file)
    
#     # Read lines from input file and write to CSV file
#     for line in in_file:
#         # Remove newline character and split line on comma separator
#         row = line.strip().split(',')
#         # Write row to CSV file
#         writer.writerow(row)





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read data from CSV file
df=pd.read_csv('data.csv')

df.insert(loc=1, column='x0', value=1)
# df['x0'] = 1

print(df)

# # Extract population and profit values as lists
# population = [float(row['population']) for row in df]
# profit = [float(row['profit']) for row in df]

# Define the colors and marker style
colors = ['red']
markers = ['x']

# Create scatter plot with colors and marker style
# for i in range(len(population)):
plt.scatter(df['population'], df['profit'], color='red', marker='x')

# Add axis labels and title
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Population vs. Profit (m={})'.format(len(df)))

# Display the plot
#plt.show()


theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01
X = df[['population', 'x0']]
y = df['profit']

# Define cost function
def ComputeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum((h - y) ** 2)
    return J

# Define gradient descent algorithm
def gradient_descent(X, y, theta, alpha=0.01, iterations=1500):
    m = len(y)
    J_history = np.zeros((iterations, 1))
    
    for i in range(1, iterations):
        h = X.dot(theta)
        setParameter = 2
        for j in range(0, setParameter):
            error = h - y
            theta[j] = theta[j] - (alpha*(1/m * sum(error * X.iloc[:, j])))
        # J_history[i] = ComputeCost(X, y, theta)
    
    return theta, J_history


# Run gradient descent
# theta, J_history = GradientDescent(X, y, theta, alpha, iterations)

# Compute final cost
J = ComputeCost(X, y, [0, 0])


print('J', J)


# Run gradient descent
theta, J_history = gradient_descent(X, y, [0, 0], alpha, iterations)

print('Theta', theta)
import numpy as np
import matplotlib.pyplot as plt

def cost(x, y, w, b, n):
    J = 1/(2*n) * np.sum((y - (w*x + b))**2)
    return J

def weight_update(x, y, w, b, alpha, n):
    y_prediction = w*x + b
    w = w + alpha * 1/n * np.dot(y - y_prediction, x)
    b = b + alpha * 1/n * np.sum(y - y_prediction)
    return w, b
    
def plot_regression_line(x, y, w, b, i=None):
    # Plot the points
    plt.scatter(x, y)
    
    # Plot the regression line
    x_values = np.linspace(x.min(), x.max(), 100)
    y_prediction = w*x_values + b
    legend = "Iteration" + str(i)
    plt.plot(x_values, y_prediction, label=legend);
    plt.xlabel("Average number of rooms");
    plt.ylabel("Median value of homes in $1000’s");
    plt.title("Fitting a linear regression model");
    if i:
        plt.legend();
    
def plot_cost_function(J):
    plt.figure(figsize=(8, 4))
    plt.plot(J, label="Cost function");
    plt.xlabel("Iterations for the weight update");
    plt.title("Decline in the cost function");
    plt.legend();
    
    
def iterations(x, y, w, b, alpha, n, m=5):
    print("Initial Slope:", w)
    print("Intial Intercept:", b)
    print("Initial Cost:", cost(x, y, w, b, n))
    plt.figure(figsize=(12, 6))
    plot_regression_line(x, y, w, b, i=0)
    J = np.zeros(m)
    for i in range(m):
        w, b = weight_update(x, y, w, b, alpha, n)
        J[i] = cost(x, y, w, b, n)
        print("\nAfter {} iteration".format(i+1))
        print("Updated Slope :", w)
        print("Updated Intercept:", b)
        print("Cost:", J[i])
        plot_regression_line(x, y, w, b, i+1)
    plot_cost_function(J)
    
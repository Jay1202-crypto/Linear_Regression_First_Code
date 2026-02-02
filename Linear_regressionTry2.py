import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost(w, b, x, y):
    cost = 1 / (2 * len(x)) * np.sum((w * x + b - y) ** 2)
    return cost
def gradient_descent_step(w, b, x, y):
    alpha = 0.01
    dw_dj = 1 / len(x) * np.sum((w * x + b - y) * x)
    db_dj = 1 / len(x) * np.sum(w * x + b - y)
    for i in range(10000):
        temp_w = w - alpha * dw_dj
        temp_b = b - alpha * db_dj
        w = temp_w
        b = temp_b
        dw_dj = 1 / len(x) * np.sum((w * x + b - y) * x)
        db_dj = 1 / len(x) * np.sum(w * x + b - y)
    return w, b

def main():
    Dataframe1=pd.read_csv("single_variable_linear_regression_data.csv")
    x_values = np.array(Dataframe1['x'])/50
    y_values = np.array(Dataframe1['y'])/151.18479922
    #print("X values:", x_values)
    #print("Y values:", y_values)
    print(cost(0.5, 0, x_values, y_values))
    w, b = gradient_descent_step(0.5, 0, x_values, y_values)
    print("Optimized w:", w)
    print("Optimized b:", b)
    plt.scatter(x_values, y_values, color='blue', label='Data Points')
    plt.plot(x_values, w * x_values + b, color='red', label='Regression Line')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear Regression Fit')
    plt.show()

if __name__ == "__main__":
    main()
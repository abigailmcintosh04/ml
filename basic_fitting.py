import numpy as np
import matplotlib.pyplot as plt

def linear_fit(x, y):
    # Minimise chi-squared.
    sum1 = len(x)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumxx = np.sum(x ** 2)
    sumxy = np.sum(x * y)
    delta = (sum1 * sumxx - sumx ** 2)

    a = (sum1 * sumxy - sumx * sumy) / delta
    b = (sumxx * sumy - sumx * sumxy) / delta

    return a, b


def plot_fit(x, y, a, b):
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="+", color="tab:orange")
    ax.plot(x, a * x + b, color="tab:blue")
    plt.show()


# Uniformly random x values.
x = np.random.uniform(-5, 5, 100)

# Linear relationship with noise.
y = (x * 2 + 3) + np.random.normal(0, 3, 100)

a, b = linear_fit(x, y)
plot_fit(x, y, a, b)
print(a, b)
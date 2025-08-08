import numpy as np 
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5])
y = np.array([3,4,2,5,6])

m = 0
c = 0


learning_rate = 0.01
epochs = 1000

n = len(X)

for _ in range(epochs):
    y_pred = m * X + c

    dm = (-2/n) * sum(X * (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

print(f"Final slope (m): {m}")
print(f"Final intercept (c): {c}")


# 6. Plot the dataset and regression line
plt.scatter(X, y, color='blue', label='Data points')          # original points
plt.plot(X, m * X + c, color='red', label='Regression line')  # fitted line
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.legend()
plt.show()

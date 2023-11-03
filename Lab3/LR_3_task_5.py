import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Генерація даних
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

regressor = linear_model.LinearRegression()
regressor.fit(X, y)

polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial.fit_transform(X)
polynomial.fit(X_poly, y)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_poly, y)
y_pred = poly_linear_model.predict(X_poly)

print("\nR2: ", sm.r2_score(y, y_pred))

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue', linewidth=2)
plt.title("Лінійна регресія")
plt.show()

XY = [[X[i], y_pred[i]] for i in range(100)]
XY.sort(key=lambda a: a[0][0])

plt.scatter(X, y, color='red')
plt.plot([i[0][0] for i in XY], [i[1][0] for i in XY], color='blue', linewidth=2)
plt.title("Поліноміальна регресія")
plt.show()

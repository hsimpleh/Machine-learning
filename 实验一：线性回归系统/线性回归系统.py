import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# MSE（自己编写的均方根值）
def MSE_self(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 十折交叉验证均方根值
def ten_fold_cross_validation(model, X, Y):
    folden = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X, Y, cv=folden, scoring='neg_mean_squared_error')
    return -mse_scores.mean()

# 岭回归模型（Ridge Regression Model）
def Train_Ridge_Regression_Model(X_train, Y_train, alphas):
    Ridge_results = []
    best_alpha = None
    best_mse = float('inf')

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        mse_cv = ten_fold_cross_validation(ridge_model, X_train, Y_train)
        Ridge_results.append((alpha, mse_cv))

        if mse_cv < best_mse:
            best_mse = mse_cv
            best_alpha = alpha

    final_ridge_model = Ridge(alpha=best_alpha)
    final_ridge_model.fit(X_train, Y_train)
    return final_ridge_model, best_alpha, Ridge_results
# 定义一系列的alpha值进行测试
alphas = np.logspace(-10, 10, 50)

class MyLinearRegression:
    def __init__(self):
        self.gradient = None  # 斜率
        self.bia = None  # 偏置
        self._beta = np.array([])  # 初始化为空数组

    def calculate(self, X, Y):
        # 因为 y = β0 + β1X1 + β2X2 + ... + βnXn，所以需要加一列1充当截距
        X_b = np.hstack([np.ones((len(X), 1)), X])
        # 使用最小二乘法求参数
        self._beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

    def fit(self, X, Y):
        # 调用 calculate 方法计算参数
        self.calculate(X, Y)
        # 将 _beta 赋值给 beta
        self.intercept_ = self._beta[0]
        self.coef_ = self._beta[1:]

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])  # 同样，给X增加一列1以考虑截距
        return X_b.dot(self._beta)


# 读取数据
data = pd.read_csv("boston.csv")

# 提取特征和目标变量
X = data.drop("MEDV", axis=1)  # 删掉房价这一预测变量
Y = data["MEDV"]  # 提取房价这一预测变量

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集和训练集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

# 训练岭回归模型并找出最佳的alpha
final_ridge_model, best_alpha, ridge_cv_results = Train_Ridge_Regression_Model(X_train, y_train, alphas)

print(f"Best alpha for Ridge Regression: {best_alpha}")
print(f"Ridge Regression - Best 10-fold CV MSE: {min([mse for _, mse in ridge_cv_results])}")

model = MyLinearRegression()
model.calculate(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

#绘制图像
#1.绘制目标变量分布
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
sns.histplot(data["MEDV"], bins=30, kde=True)
plt.title("Distribution of Target Variable (MEDV)")

# 创建测量变量之间线性关系的相关矩阵并绘制相关矩阵
correlation_matrix = data.corr()
plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(alphas, [mse for alpha, mse in ridge_cv_results], marker='o')
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MSE")
plt.title("Ridge Regression - 10-fold CV MSE vs. Alpha")
plt.show()

# 绘制原始房价和预测房价之间的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Scatter Plot of Actual vs. Predicted Prices")
plt.show()
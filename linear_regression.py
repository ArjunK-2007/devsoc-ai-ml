# Script version of the DevSoc notebook.
# Assignment sources and assistance are documented in README.md.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("https://raw.githubusercontent.com/cronan03/DevSoc_AI-ML/main/train_processed_splitted.csv")
print(df.shape)


print(df.head())


print(df.columns)
print(df.describe())
print(df.isna().sum())
print(df["Utilities"].value_counts())


df = pd.get_dummies(df, columns=["Utilities"], dtype=float)
print(df.head())


# Keep these training values for scaling the test data too.
data_min = df.min()
data_max = df.max()
data_range = data_max - data_min
data_range = data_range.replace(0, 1)

df_scaled = (df - data_min) / data_range
print(df_scaled.head())


x = df_scaled.drop(columns="SalePrice").to_numpy()
y = df_scaled["SalePrice"].to_numpy().reshape(-1, 1)
print(x.shape, y.shape)


np.random.seed(42)
w = np.random.randn(1, x.shape[1])
b = 0.0

y_pred = x @ w.T + b
print("Number of parameters:", w.size + 1)


print("Actual values:")
print(y[:5])
print("Initial predictions:")
print(y_pred[:5])


def mse_loss_fn(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

print("Initial MSE:", mse_loss_fn(y, y_pred))


def get_gradients(y_true, y_pred, W, b, X):
    m = len(X)
    error = y_pred - y_true
    dW = (2 / m) * (error.T @ X)
    db = 2 * np.mean(error)
    return dW, db


def update(weights, bias, gradients_weights, gradients_bias, lr):
    weights = weights - lr * gradients_weights
    bias = bias - lr * gradients_bias
    return weights, bias


NUM_EPOCHS = 20000
LEARNING_RATE = 0.05

# Start from the same random weights when this cell is rerun.
np.random.seed(42)
w = np.random.randn(1, x.shape[1])
b = 0.0
losses = []

for epoch in range(NUM_EPOCHS):
    y_pred = x @ w.T + b
    loss = mse_loss_fn(y, y_pred)
    losses.append(loss)
    dw, db = get_gradients(y, y_pred, w, b, x)
    w, b = update(w, b, dw, db, LEARNING_RATE)

train_mse = mse_loss_fn(y, x @ w.T + b)
print("Initial MSE:", losses[0])
print("Final training MSE:", train_mse)


plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Training loss")
plt.show()


df_test = pd.read_csv("https://raw.githubusercontent.com/cronan03/DevSoc_AI-ML/main/test_processed_splitted.csv")
print(df_test.head())
print(df_test.isna().sum())

df_test = pd.get_dummies(df_test, columns=["Utilities"], dtype=float)

# The test set may not contain every Utilities category.
missing_cols = []
for column in df.columns:
    if column not in df_test.columns:
        missing_cols.append(column)
        df_test[column] = 0

# Use the same column order as the training data.
df_test = df_test[df.columns]


df_test_scaled = (df_test - data_min) / data_range

x_test = df_test_scaled.drop(columns="SalePrice").to_numpy()
y_test = df_test_scaled["SalePrice"].to_numpy().reshape(-1, 1)
print(x_test.shape, y_test.shape)


print("Columns added to the test set:", missing_cols)
print("Training columns:", df.columns)
print("Test columns:", df_test.columns)


y_pred_test = x_test @ w.T + b
loss_test = mse_loss_fn(y_test, y_pred_test)

# Convert predictions and targets back to original price units.
y_pred_test_scaled = y_pred_test * data_range["SalePrice"] + data_min["SalePrice"]
y_test_scaled = y_test * data_range["SalePrice"] + data_min["SalePrice"]

print("Training MSE:", train_mse)
print("Test MSE:", loss_test)


np.random.seed(42)
idx = np.random.randint(0, x_test.shape[0], 5)
y_pred_test_sample = y_pred_test_scaled[idx].round().astype(int)
y_true_test_sample = y_test_scaled[idx].round().astype(int)

print("Predicted SalePrice:")
print(y_pred_test_sample)
print("Actual SalePrice:")
print(y_true_test_sample)

if loss_test > train_mse:
    print("Test MSE is higher than training MSE.")
else:
    print("Test MSE is lower than or equal to training MSE.")

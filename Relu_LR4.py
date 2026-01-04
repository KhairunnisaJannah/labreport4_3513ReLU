import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

st.title("ReLU Activation Function Visualization")
st.subheader("ReLU Activation Function")

x = np.linspace(-5, 5, 50)
z = [max(0, i) for i in x]

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(x, z)
ax1.set_xlabel("Input")
ax1.set_ylabel("Output")
ax1.set_title("ReLU Activation Function")
ax1.grid()

st.pyplot(fig1)
st.subheader("Dataset Visualization (make_circles)")

X, y = make_circles(
    n_samples=10000,
    noise=0.05,
    random_state=26
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=26
)

fig2, (train_ax, test_ax) = plt.subplots(
    ncols=2, sharex=True, sharey=True, figsize=(10, 5)
)

train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
test_ax.set_title("Testing Data")
test_ax.set_xlabel("Feature #0")

st.pyplot(fig2)


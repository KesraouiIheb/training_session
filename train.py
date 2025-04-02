# train_model.py
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Create some synthetic data for training
# For example, y = 2 * x + 3 with some noise
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.squeeze() + 3 + np.random.normal(0, 1, X.shape[0])

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model and training data
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the training data (for plotting later)
with open("training_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("Model and training data saved.")

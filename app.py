# app.py
import io
import pickle
import base64
import numpy as np
from flask import Flask, render_template, request
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model and training data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("training_data.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    plot_url = None
    input_value = None

    if request.method == "POST":
        try:
            # Get user input from the form
            input_value = float(request.form["input_value"])
            input_array = np.array([[input_value]])

            # Get the model's prediction
            prediction = model.predict(input_array)[0]

            # Generate the plot
            fig, ax = plt.subplots(figsize=(6, 4))
            # Plot training data
            ax.scatter(X_train, y_train, color="blue", label="Training data")
            # Plot the regression line over a range of x values
            x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, color="red", label="Regression line")
            # Mark the user's input and prediction
            ax.scatter(input_value, prediction, color="green", s=100, label="Your prediction")
            ax.set_xlabel("Input Feature")
            ax.set_ylabel("Target")
            ax.legend()

            # Save plot to a string in base64 encoding
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
            plt.close(fig)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, plot_url=plot_url, input_value=input_value)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

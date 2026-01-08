from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("EDA_FE.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        brand = int(request.form["brand"])
        model_name = int(request.form["model"])
        condition = int(request.form["condition"])
        ram = float(request.form["ram"])
        storage = float(request.form["storage"])
        battery = float(request.form["battery"])
        age = float(request.form["age"])

        # Arrange input in same order as training data
        features = np.array([[brand, model_name, condition, ram, storage, battery, age]])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Resale Price: ₹{round(prediction, 2)}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

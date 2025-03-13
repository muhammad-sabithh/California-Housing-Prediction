from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    scaled_features = scaler.transform(features) 
    prediction = model.predict(scaled_features)[0]
    return jsonify({"predicted_price": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)


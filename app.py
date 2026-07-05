"""
app.py
------
Flask backend that connects your website to YOUR trained model.pkl
(the one sitting next to model.ipynb / Clean_Dataset.csv in your project folder).

FOLDER LAYOUT EXPECTED:
  FLIGHT.../
    ├── app.py                <- this file
    ├── model.pkl             <- your trained model (already have it)
    ├── Clean_Dataset.csv
    ├── model.ipynb
    └── templates/
         └── index.html       <- the website (move it here, rename to index.html)

RUN:
  pip install flask pandas scikit-learn joblib flask-cors
  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# ---- Load YOUR trained model ----
model = joblib.load("model.pkl")

# If your notebook trained on a DataFrame with these exact column names,
# this will just work. If your column order/names differ, adjust COLUMNS below
# to match what you used in model.ipynb when calling model.fit(X, y).
COLUMNS = [
    "airline", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class", "duration", "days_left"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        row = {
            "airline": data.get("airline"),
            "source_city": data.get("source_city"),
            "departure_time": data.get("departure_time"),
            "stops": data.get("stops"),
            "arrival_time": data.get("arrival_time"),
            "destination_city": data.get("destination_city"),
            "class": data.get("class"),
            "duration": float(data.get("duration")),
            "days_left": float(data.get("days_left")),
        }

        X = pd.DataFrame([row], columns=COLUMNS)
        pred = model.predict(X)[0]
        pred = max(float(pred), 1100)

        return jsonify({
            "success": True,
            "price": round(pred),
            "low": round(pred * 0.88),
            "high": round(pred * 1.12)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

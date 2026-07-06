# ✈️ AEROVANE — Flight Fare Prediction

AEROVANE is a machine learning–powered web app that predicts Indian domestic flight fares in real time. It's trained on real ticketing data spanning six major airlines and six metro cities, and serves predictions through a Flask API connected to a live, aesthetic booking-style frontend.

**Live demo:** _add your Render link here once deployed_

---

## 🧠 What it does

Fill in a route, airline, cabin class, timing, stop count, flight duration, and days left before departure — AEROVANE returns an estimated fare instantly, along with a typical price range, using a model trained on real fare data rather than a hardcoded guess.

---

## 📊 Dataset

- **Source:** `Clean_Dataset.csv`
- **Size:** 300,153 flight records
- **Airlines covered:** Vistara, Air India, Indigo, GO FIRST, AirAsia, SpiceJet
- **Cities covered:** Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai
- **Features used:**
  - `airline`
  - `source_city`
  - `destination_city`
  - `departure_time`
  - `arrival_time`
  - `stops`
  - `class` (Economy / Business)
  - `duration` (hours)
  - `days_left` (days before departure)
- **Target:** `price`

---

## 🧹 Data cleaning & EDA (`model.ipynb`)

- Loaded `Clean_Dataset.csv` and checked shape, dtypes, and summary stats
- Verified no missing values (`df.isnull().sum()`) and no duplicate rows
- Dropped `Unnamed: 0` (stray index column) and `flight` (flight number — not predictive) since they add no value to fare prediction
- Explored relationships visually before modeling:
  - Price vs. `days_left` (fares rise sharply as departure gets closer, then spike right before the flight)
  - Price distribution (right-skewed, most fares clustered lower with a long tail toward expensive business fares)
  - Price vs. `duration` (weak positive correlation)
  - Price by `class`, `airline`, `stops` (boxplots — class and airline show the biggest spread)
  - Correlation heatmap across numeric columns

## 🛠️ Feature engineering & pipeline

Features were split into three groups and handled differently inside a single `ColumnTransformer`:

| Group | Columns | Transform |
|---|---|---|
| Polynomial numeric | `days_left` | `PolynomialFeatures(degree=2)` — captures the non-linear "prices spike as departure nears" curve seen in EDA |
| Linear numeric | `duration` | `StandardScaler()` |
| Categorical | `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class` | `OneHotEncoder(handle_unknown="ignore")` |

```python
preprocessor = ColumnTransformer([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False), Poly_feature),
    ("scale",  StandardScaler(), Linear_numerical_feature),
    ("encode", OneHotEncoder(handle_unknown="ignore"), Categorical_feature)
])
```

Data was split 80/20 (`train_test_split`, `random_state=42`).

## 🤖 Model selection

Three regressors were compared inside the same pipeline using 5-fold cross-validation (RMSE):

| Model | CV RMSE |
|---|---|
| Linear Regression | 6663.35 |
| Ridge Regression | 6666.88 |
| Lasso Regression | 6663.15 |

`GridSearchCV` then tuned the polynomial degree for `days_left` (tested 1–4). **Linear Regression** was selected (tied with Lasso, and preferred for simplicity/interpretability) with the best polynomial degree = **2**.

### Final test-set performance
| Metric | Score |
|---|---|
| MAE | 4442.47 |
| RMSE | 6673.87 |
| R² | 0.91 |

Error was also broken down by cabin `class` to check the model isn't systematically worse for Economy vs. Business fares.

The final tuned pipeline (preprocessing + model, as one object) was exported with `joblib.dump(final_model, "model.pkl")` — this is the exact file `app.py` loads to serve live predictions, so the deployed app uses the same preprocessing and weights as the notebook.

---

## 🗂️ Project structure

```
Flight Price Prediction/
├── app.py                 # Flask backend — loads model.pkl, exposes /predict API
├── model.ipynb             # Notebook: data cleaning, EDA, model training
├── model.pkl                # Trained model (LinearRegression pipeline)
├── Clean_Dataset.csv        # Source dataset
├── requirements.txt         # Python dependencies
├── Procfile                 # Render/Gunicorn start config
└── templates/
    └── index.html            # Frontend — fare predictor UI
```

---

## ⚙️ Tech stack

| Layer      | Tech                              |
|------------|------------------------------------|
| Frontend   | HTML, CSS, vanilla JS               |
| Backend    | Flask, Flask-CORS                   |
| ML         | scikit-learn (LinearRegression), pandas, joblib |
| Deployment | Render (Gunicorn WSGI server)       |

---

## 🚀 Run locally

```bash
git clone https://github.com/YOUR_USERNAME/flight-price-predictor.git
cd flight-price-predictor
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. On [render.com](https://render.com) → **New Web Service** → connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy — Render gives you a public URL

---

## 📌 API

**POST** `/predict`

Request body:
```json
{
  "airline": "Vistara",
  "source_city": "Delhi",
  "destination_city": "Mumbai",
  "departure_time": "Morning",
  "arrival_time": "Evening",
  "stops": "zero",
  "class": "Economy",
  "duration": 12.5,
  "days_left": 20
}
```

Response:
```json
{
  "success": true,
  "price": 5230,
  "low": 4600,
  "high": 5860
}
```

---

## ⚠️ Disclaimer

Predicted fares are model estimates based on historical data patterns, not live ticket prices from any airline or booking platform.

---


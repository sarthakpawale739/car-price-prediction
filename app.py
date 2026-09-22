from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd
from pathlib import Path

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
model = joblib.load(BASE/"final_used_car_price_model.pkl")
preprocessor = joblib.load(BASE/"used_car_preprocessor.pkl")

METRICS = {"r2":0.9330, "mae":70906.35, "rmse":121236.86}

def make_prediction(d):
    year=int(d["year"])
    df=pd.DataFrame({
        "brand":[d["brand"]],"year":[year],"km_driven":[float(d["km_driven"])],
        "mileage":[float(d["mileage"])],"engine":[float(d["engine"])],
        "max_power":[float(d["max_power"])],"torque":[float(d["torque"])],
        "seats":[float(d["seats"])],"owner":[d["owner"]],
        "seller_type":[d["seller_type"]],"fuel":[d["fuel"]],
        "transmission":[d["transmission"]],"car_age":[2026-year]
    })
    x=preprocessor.transform(df)
    return float(model.predict(x)[0]), 2026-year

@app.route("/")
def home(): return render_template("index.html", metrics=METRICS)

@app.post("/predict")
def predict():
    try:
        price, age=make_prediction(request.get_json())
        return jsonify(success=True, price=round(price,2), car_age=age,
                       model="Tuned Random Forest Regressor", r2=METRICS["r2"])
    except Exception as e:
        return jsonify(success=False, error=str(e)),400

if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

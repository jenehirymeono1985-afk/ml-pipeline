import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bank Marketing Term Deposit Prediction API")

# Load the saved scikit-learn pipeline once on startup
model = joblib.load("model.joblib")


class BankMarketingInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


@app.post("/predict")
def predict(features: BankMarketingInput):
    # Map snake_case API fields to the exact CSV column names used during training
    row = {
        "age": features.age,
        "job": features.job,
        "marital": features.marital,
        "education": features.education,
        "default": features.default,
        "housing": features.housing,
        "loan": features.loan,
        "contact": features.contact,
        "month": features.month,
        "day_of_week": features.day_of_week,
        "campaign": features.campaign,
        "pdays": features.pdays,
        "previous": features.previous,
        "poutcome": features.poutcome,
        "emp.var.rate": features.emp_var_rate,
        "cons.price.idx": features.cons_price_idx,
        "cons.conf.idx": features.cons_conf_idx,
        "euribor3m": features.euribor3m,
        "nr.employed": features.nr_employed,
    }
    input_df = pd.DataFrame([row])
    pred = model.predict(input_df)[0]
    return {"prediction": pred.item() if hasattr(pred, "item") else pred}
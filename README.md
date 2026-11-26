# Bank Term Deposit Subscription Pipeline

## Business Problem & Dataset
Direct phone marketing campaigns for a bank waste agent calling time when outreach teams dial clients unlikely to subscribe to a term deposit. This project predicts whether a client will subscribe to a term deposit (`y`: `yes` or `no`) so marketing teams can prioritize higher-likelihood leads before placing calls.

* **Dataset:** [Bank Marketing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`, 41,188 rows).
* **Dropped Column:** `duration` (last contact duration in seconds) was removed prior to modeling because call length is only known after a call ends and would leak post-outcome information unavailable at prediction time.

## How to Run
Run the PyCaret discovery and scikit-learn pipeline training script from the project root:

```bash
uv run python discovery.py
```







## Workflow Comparison
PyCaret was much faster to write—requiring only setup(), compare_models(n_select=3), and plot_model(best_model, plot="confusion_matrix", save=True)—but took 143.46s to compare models across cross-validation folds. Rebuilding GradientBoostingClassifier manually in scikit-learn with a ColumnTransformer and Pipeline took more code to write, yet trained and evaluated in 6.55s.
``
Scores also differ because PyCaret reports cross-validation averages across training folds, whereas classification_report evaluates a single held-out test set (12,357 rows from a 30% stratified split). In PyCaret, GradientBoostingClassifier (gbc) achieved 0.9011 accuracy, 0.7919 AUC, 0.9011 recall, 0.8837 precision, and 0.8804 F1. On the held-out test set, the manual scikit-learn pipeline achieved 0.90 overall accuracy, with 0.91 precision, 0.98 recall, and 0.95 F1 for no (support 10,965), versus 0.67 precision, 0.24 recall, and 0.36 F1 for yes (support 1,392).
``
For outbound bank telemarketing, precision (and F1 on the yes class) matters most when call-center agent capacity is limited so reps avoid wasting calls on non-subscribers, while recall on yes matters if the bank wants to avoid missing potential term deposit signups.
``
## Screenshots & Sample API Payload
``
### PyCaret Comparison Table
![PyCaret Comparison Table](images/pycaret_comparison.png)
``
### Confusion Matrix
![Confusion Matrix](images/Confusion%20Matrix.png)
``
### FastAPI /docs Successful Prediction
![FastAPI Predict Response](images/fastapi_predict.png)
``
### Sample JSON Input
 ```json
{
  "age": 35,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": 1.1,
  "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4,
  "euribor3m": 4.857,
  "nr_employed": 5191.0
}
```
``
### Sample JSON Output
 ```json
{
  "prediction": "no"
}
```
``
## Prediction Experiment
* **Request 1 Input:** Baseline profile above ("poutcome": "nonexistent", "euribor3m": 4.857, "age": 35, etc.) → **Prediction:** {"prediction": "no"}.
* **Request 2 Input:** Changed one field ("poutcome": "success") while keeping all other fields identical → **Prediction:** {"prediction": "no"}.
* **Business Interpretation:** Even after changing "poutcome" to "success", the model still predicted "no". This makes sense in context because GradientBoostingClassifier combines multiple features together—including macroeconomic rate indicators ("euribor3m": 4.857, "emp_var_rate": 1.1, "nr_employed": 5191.0) and "pdays": 999—and because "no" is much more common in the training data, changing one feature alone did not push the predicted probability over the decision threshold.
``
## AI Use
I set up the workspace, extracted the dataset, ran the scripts, tested /docs, and verified the outputs myself, while using an AI assistant to draft boilerplate code for discovery.py, main.py, and the README.md structure so I could focus on checking results and understanding the pipeline design. One thing I had to check and organize was saving the dataset CSV inside data/, checking .gitignore so model.joblib and data/ were not excluded, and confirming my screenshots were saved in images/.
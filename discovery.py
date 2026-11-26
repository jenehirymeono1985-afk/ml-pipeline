import time
import joblib
import pandas as pd
from pycaret.classification import compare_models, plot_model, setup
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- 1. PyCaret Workflow ---
pycaret_start = time.perf_counter()

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
# Drop post-call duration because it is only known after a call finishes (leakage)
df = df.drop(columns=["duration"])

setup(data=df, target="y", session_id=42, verbose=False)
top_3 = compare_models(n_select=3)
best_model = top_3[0]
plot_model(best_model, plot="confusion_matrix", save=True)

pycaret_elapsed = time.perf_counter() - pycaret_start
print(f"PyCaret workflow runtime: {pycaret_elapsed:.2f}s")

# --- 2. Manual scikit-learn Pipeline ---
sklearn_start = time.perf_counter()

X = df.drop(columns=["y"])
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

sklearn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42)),
])

sklearn_pipeline.fit(X_train, y_train)
y_pred = sklearn_pipeline.predict(X_test)

sklearn_elapsed = time.perf_counter() - sklearn_start

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))
print(f"scikit-learn workflow runtime: {sklearn_elapsed:.2f}s")

joblib.dump(sklearn_pipeline, "model.joblib")
print("Saved fitted pipeline to model.joblib")
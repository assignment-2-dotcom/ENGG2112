import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.over_sampling import BorderlineSMOTE

import shap
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

df = pd.read_csv(r"C:\Users\bayrj\OneDrive\Documents\engg2112\cardio_train.csv", sep=';')

print("=== RAW DATA ===")
print(df.head())
print("Shape:", df.shape)

# Convert age from days to years
df["age"] = (df["age"] / 365).round(1)

# Drop the id column
df = df.drop(columns=["id"])

# Define unrealistic value masks
unrealistic = {
    "age (< 0 or > 120 years)":               (df["age"] < 0) | (df["age"] > 120),
    "gender (not 1 or 2)":                     ~df["gender"].isin([1, 2]),
    "height (< 100 cm or > 250 cm)":           (df["height"] < 100) | (df["height"] > 250),
    "weight (< 20 kg or > 300 kg)":            (df["weight"] < 20) | (df["weight"] > 300),
    "ap_hi systolic (< 50 or > 250)":          (df["ap_hi"] < 50) | (df["ap_hi"] > 250),
    "ap_lo diastolic (< 30 or > 200)":         (df["ap_lo"] < 30) | (df["ap_lo"] > 200),
    "blood pressure (ap_lo > ap_hi)":          df["ap_lo"] > df["ap_hi"],
    "cholesterol (not 1,2,3)":                 ~df["cholesterol"].isin([1, 2, 3]),
    "gluc (not 1,2,3)":                        ~df["gluc"].isin([1, 2, 3]),
    "smoke (not 0 or 1)":                      ~df["smoke"].isin([0, 1]),
    "alco (not 0 or 1)":                       ~df["alco"].isin([0, 1]),
    "active (not 0 or 1)":                     ~df["active"].isin([0, 1]),
    "cardio (not 0 or 1)":                     ~df["cardio"].isin([0, 1]),
}

total = len(df)
results = []
bad_mask = pd.Series(False, index=df.index)

for name, condition in unrealistic.items():
    count = condition.sum()
    results.append([name, count, round(count / total * 100, 2)])
    bad_mask = bad_mask | condition

result_df = pd.DataFrame(results, columns=["Category", "Unrealistic Count", "Percent"])
print("\n=== UNREALISTIC VALUES ===")
print(result_df)
print(f"\nTotal rows removed: {bad_mask.sum()}")

df_clean = df[~bad_mask].copy()
print(f"Remaining rows: {len(df_clean)}")

# Handle missing values
print("\n=== MISSING VALUES ===")
print(df_clean.isnull().sum())
df_clean = df_clean.dropna()


# =============================================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================================

# BMI is a strong cardiovascular predictor
df_clean["bmi"] = df_clean["weight"] / (df_clean["height"] / 100) ** 2

# Pulse pressure (systolic - diastolic) — another useful cardiac indicator
df_clean["pulse_pressure"] = df_clean["ap_hi"] - df_clean["ap_lo"]


# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n=== CLASS BALANCE ===")
print(df_clean["cardio"].value_counts())
print(df_clean["cardio"].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Class balance bar chart
df_clean["cardio"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Class Balance (0=No CVD, 1=CVD)")
axes[0].set_xlabel("Cardio")
axes[0].set_ylabel("Count")
axes[0].tick_params(rotation=0)

# Age distribution by class
df_clean.groupby("cardio")["age"].plot(kind="kde", ax=axes[1], legend=True)
axes[1].set_title("Age Distribution by Class")
axes[1].set_xlabel("Age (years)")
axes[1].legend(["No CVD", "CVD"])

# BMI distribution by class
df_clean.groupby("cardio")["bmi"].plot(kind="kde", ax=axes[2], legend=True)
axes[2].set_title("BMI Distribution by Class")
axes[2].set_xlabel("BMI")
axes[2].legend(["No CVD", "CVD"])

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()


# =============================================================================
# STEP 4 — FEATURE SELECTION
# =============================================================================

X = df_clean.drop(columns=["cardio"])
y = df_clean["cardio"]

selector = SelectKBest(f_classif, k="all")
selector.fit(X, y)

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "F-Score": selector.scores_
}).sort_values("F-Score", ascending=False)

print("\n=== FEATURE SCORES ===")
print(feature_scores)

plt.figure(figsize=(10, 5))
sns.barplot(data=feature_scores, x="F-Score", y="Feature", palette="viridis")
plt.title("Feature Importance (SelectKBest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# Keep top features (drop height & weight since bmi captures both)
features_to_drop = ["height", "weight"]
X = X.drop(columns=features_to_drop)
print(f"\nFeatures used: {list(X.columns)}")


# =============================================================================
# STEP 5 — HANDLE CLASS IMBALANCE WITH BORDERLINE SMOTE
# =============================================================================

print("\n=== APPLYING BORDERLINE SMOTE ===")
sm = BorderlineSMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(f"Before SMOTE: {y.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")


# =============================================================================
# STEP 6 — TRAIN/TEST SPLIT + SCALING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_sc  = scaler.transform(X_test)        # transform both

print(f"\nTrain size: {X_train_sc.shape}, Test size: {X_test_sc.shape}")


# =============================================================================
# STEP 7 — TRAIN & COMPARE MODELS
# =============================================================================

models = {
    "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN":                   KNeighborsClassifier(n_neighbors=5),
}

results_list = []

print("\n=== MODEL COMPARISON ===")
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_proba)
    cv_f1  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="f1").mean()

    results_list.append({
        "Model":     name,
        "Accuracy":  round(report["accuracy"], 4),
        "Precision": round(report["1"]["precision"], 4),
        "Recall":    round(report["1"]["recall"], 4),
        "F1 Score":  round(report["1"]["f1-score"], 4),
        "AUC":       round(auc, 4),
        "CV F1":     round(cv_f1, 4),
    })

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))

comparison_df = pd.DataFrame(results_list).sort_values("F1 Score", ascending=False)
print("\n=== SUMMARY TABLE ===")
print(comparison_df.to_string(index=False))


# =============================================================================
# STEP 8 — CONFUSION MATRICES FOR ALL MODELS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test_sc, y_test,
        display_labels=["No CVD", "CVD"],
        ax=axes[i], colorbar=False, cmap="Blues"
    )
    axes[i].set_title(name)

plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()


# =============================================================================
# STEP 9 — ROC CURVES FOR ALL MODELS
# =============================================================================

plt.figure(figsize=(8, 6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test_sc, y_test, name=name, ax=plt.gca())
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()


# =============================================================================
# STEP 10 — HYPERPARAMETER TUNING (Best model = Logistic Regression)
# =============================================================================

print("\n=== HYPERPARAMETER TUNING (Logistic Regression) ===")

param_grid = {
    "C":       [0.001, 0.01, 0.1, 1, 10],
    "solver":  ["lbfgs", "saga"],
    "penalty": ["l2"],
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train_sc, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV F1:  {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best  = best_model.predict(X_test_sc)
y_proba_best = best_model.predict_proba(X_test_sc)[:, 1]

print("\n=== TUNED MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred_best))
print(f"AUC: {roc_auc_score(y_test, y_proba_best):.4f}")


# =============================================================================
# STEP 11 — SHAP FEATURE INTERPRETATION
# =============================================================================

print("\n=== SHAP ANALYSIS ===")

# Use a sample for speed
X_test_sample = X_test_sc[:500]

explainer   = shap.LinearExplainer(best_model, X_train_sc)
shap_values = explainer(X_test_sample)

plt.figure()
shap.summary_plot(shap_values, X_test_sample,
                  feature_names=list(X.columns), show=False)
plt.title("SHAP Summary Plot — Logistic Regression")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_test_sample,
                  feature_names=list(X.columns),
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP|)")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150)
plt.show()

print("\n=== DONE ===")
print("Saved plots: eda_distributions.png, correlation_matrix.png,")
print("             feature_importance.png, confusion_matrices.png,")
print("             roc_curves.png, shap_summary.png, shap_bar.png")
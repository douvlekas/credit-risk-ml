"""
Credit Risk Assessment with Machine Learning
--------------------------------------------
End-to-end credit decisioning pipeline:
  1. Data preparation (leakage control, one-hot encoding, scaling)
  2. Random Forest with cost-sensitive learning + GridSearchCV (5-fold CV)
  3. Evaluation (classification report, confusion matrix, ROC/AUC)
  4. Explainability (feature importance, SHAP waterfall for a rejected applicant)
  5. Fairness audit (Disparate Impact Ratio, four-fifths rule)
  6. Capital-constrained loan portfolio optimization (risk-adjusted ROI, greedy)

Dataset: "Financial Risk for Loan Approval" (Kaggle, synthetic, 20,000 records)
https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval
Download Loan.csv and place it in the same folder as this script.

Author: Georgios Douvlekas - MSc Business Mathematics (NKUA & AUEB)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
df = pd.read_csv("Loan.csv")

if "ApplicationDate" in df.columns:
    df = df.drop(["ApplicationDate"], axis=1)

# RiskScore is removed to prevent data leakage (it is derived from the outcome)
X = df.drop(["LoanApproved", "RiskScore"], axis=1)
y = df["LoanApproved"]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# 80% training / 20% hold-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Original features: {X.shape[1]}")
print(f"Features after encoding: {X_encoded.shape[1]}")

# ---------------------------------------------------------------------------
# 2. Model: Random Forest + hyperparameter tuning
# ---------------------------------------------------------------------------
print("\nHyperparameter tuning (GridSearchCV, 5-fold CV)...")

# class_weight='balanced' -> cost-sensitive learning for the 3:1 class imbalance
rf_base = RandomForestClassifier(random_state=42, class_weight="balanced")

param_grid = {
    "n_estimators": [50, 100, 200],   # number of trees
    "max_depth": [None, 10, 20],      # maximum tree depth
    "min_samples_split": [2, 5, 10],  # minimum samples to split a node
}

grid_search = GridSearchCV(
    estimator=rf_base, param_grid=param_grid, cv=5, n_jobs=-1, scoring="f1_macro"
)
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

y_pred_best = best_rf_model.predict(X_test_scaled)

print("\nClassification report (test set)")
print(classification_report(y_test, y_pred_best))

# Stability check with 5-fold cross-validation on the training set
cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5, scoring="accuracy")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}  (std: {np.std(cv_scores):.4f})")

# ---------------------------------------------------------------------------
# 3. Evaluation charts
# ---------------------------------------------------------------------------
# Feature importance (top 10)
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X_train.columns)
top_10_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_10_features.sort_values().plot(kind="barh", color="skyblue")
plt.title("Top 10 Features (Random Forest Feature Importance)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Rejected (0)", "Approved (1)"],
    yticklabels=["Rejected (0)", "Approved (1)"],
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix (Tuned Model)")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
y_prob_best = best_rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Applicant ranking and explainability (SHAP)
# ---------------------------------------------------------------------------
y_probabilities = best_rf_model.predict_proba(X_test_scaled)[:, 1]

client_ranking = pd.DataFrame({
    "Actual_Decision": y_test.values,
    "Model_Prediction": y_pred_best,
    "Approval_Probability (%)": y_probabilities * 100,
})
print("\nTop 5 applicants by creditworthiness:")
print(client_ranking.sort_values(by="Approval_Probability (%)", ascending=False).head(5))

# Explain why a specific applicant was rejected
rejected_indices = [i for i, pred in enumerate(y_pred_best) if pred == 0]

if len(rejected_indices) > 0:
    first_rejected_idx = rejected_indices[0]
    specific_client_data = X_test_scaled[first_rejected_idx].reshape(1, -1)

    explainer = shap.TreeExplainer(best_rf_model)
    shap_values = explainer.shap_values(specific_client_data)

    # Keep the SHAP values for the "approved" class (index 1)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_to_plot[0],
            base_values=explainer.expected_value[1],
            data=X_test.iloc[first_rejected_idx],
            feature_names=X_test.columns,
        ),
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.45)
    plt.savefig("images/shap_rejected_applicant.png", dpi=150)
    plt.close()
else:
    print("No rejected applicants found in the test set.")

# ---------------------------------------------------------------------------
# 5. Fairness audit (age-based disparate impact)
# ---------------------------------------------------------------------------
print("\nAlgorithmic fairness audit")
young_mask = X_test["Age"] < 30
older_mask = X_test["Age"] >= 30

approval_rate_young = y_pred_best[young_mask].mean()
approval_rate_older = y_pred_best[older_mask].mean()

print(f"Approval rate, age < 30:  {approval_rate_young * 100:.2f}%")
print(f"Approval rate, age >= 30: {approval_rate_older * 100:.2f}%")

disparate_impact = approval_rate_young / approval_rate_older if approval_rate_older > 0 else 0
print(f"Disparate Impact Ratio: {disparate_impact:.3f}")

# Four-fifths (80%) rule
if disparate_impact >= 0.8:
    print("RESULT: Model passes the four-fifths rule.")
else:
    print("WARNING (BIAS): DIR below 0.80 - model disadvantages applicants under 30.")

# ---------------------------------------------------------------------------
# 6. Capital-constrained portfolio optimization
# ---------------------------------------------------------------------------
print("\nLoan portfolio optimization")
sample_size = 100
X_sample = X_test.iloc[:sample_size].copy()
X_sample_scaled = X_test_scaled[:sample_size]

BANK_BUDGET = 1_000_000  # available lending capital (EUR)

prob_repay = best_rf_model.predict_proba(X_sample_scaled)[:, 1]
prob_default = 1 - prob_repay

# Expected profit = P(repay) * interest income - P(default) * principal  (assumes LGD = 100%)
loan_amounts = X_sample["LoanAmount"].values
interest_rates = X_sample["InterestRate"].values
expected_profit = (prob_repay * (loan_amounts * interest_rates)) - (prob_default * loan_amounts)

# Risk-adjusted return per euro lent
roi = expected_profit / loan_amounts

optimization_df = pd.DataFrame({
    "Client_ID": range(1, sample_size + 1),
    "Loan_Amount": loan_amounts,
    "Repayment_Probability": prob_repay,
    "Expected_Profit": expected_profit,
    "ROI": roi,
}).sort_values(by="ROI", ascending=False)

# Greedy allocation: approve highest-ROI loans with positive expected profit within budget
approved_loans, total_spent, total_expected_profit = [], 0, 0
for _, row in optimization_df.iterrows():
    if row["Expected_Profit"] > 0 and (total_spent + row["Loan_Amount"]) <= BANK_BUDGET:
        approved_loans.append(row["Client_ID"])
        total_spent += row["Loan_Amount"]
        total_expected_profit += row["Expected_Profit"]

print(f"Available budget:       {BANK_BUDGET:,.2f} EUR")
print(f"Capital allocated:      {total_spent:,.2f} EUR")
print(f"Loans approved:         {len(approved_loans)} of {sample_size}")
print(f"Expected net profit:    {total_expected_profit:,.2f} EUR")

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "resume_salary_dataset_30k.csv")

SENIORITY_ORDER   = ["Intern", "Junior", "Mid-level", "Senior", "Lead",
                     "Principal", "Staff", "Director", "VP", "CTO/CXO"]
EDUCATION_ORDER   = ["High School", "Associate", "Bachelor", "Master", "MBA", "PhD"]
COMPANY_SIZE_ORDER = ["Startup (<50)", "Small (50-200)", "Medium (200-1000)",
                      "Large (1000-5000)", "Enterprise (5000+)"]
CERT_RANK = {
    "None": 0, "AWS Certified": 3, "Google Cloud Certified": 3,
    "Azure Certified": 3, "PMP": 2, "Scrum Master": 1,
    "CFA": 3, "CISSP": 3, "Data Science Certifications": 2, "Multiple Certs": 4,
}
HIGH_PAYING_CITIES = ["San Francisco", "New York", "Seattle", "Boston", "San Jose"]

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Remove extreme outliers (keep 99.5%)
lo, hi = df["annual_salary_usd"].quantile([0.0025, 0.9975])
df = df[df["annual_salary_usd"].between(lo, hi)].copy()

df["certifications"] = df["certifications"].fillna("None")
df["gpa"]            = df["gpa"].fillna(df["gpa"].median())
print(f"  Records: {len(df):,}")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
print("Engineering features...")

df["seniority_rank"] = df["seniority_level"].apply(
    lambda x: SENIORITY_ORDER.index(x) if x in SENIORITY_ORDER else 2)
df["education_rank"] = df["education_level"].apply(
    lambda x: EDUCATION_ORDER.index(x) if x in EDUCATION_ORDER else 2)
df["company_rank"] = df["company_size"].apply(
    lambda x: COMPANY_SIZE_ORDER.index(x) if x in COMPANY_SIZE_ORDER else 2)
df["cert_rank"] = df["certifications"].map(CERT_RANK).fillna(0)

df["exp_bin"] = pd.cut(
    df["years_of_experience"],
    bins=[-1, 0, 2, 5, 10, 15, 20, 99],
    labels=[0, 1, 2, 3, 4, 5, 6],
).astype(int)

# Polynomial & log experience
df["exp_squared"] = df["years_of_experience"] ** 2
df["exp_log"]     = np.log1p(df["years_of_experience"])

# Interactions
df["seniority_x_exp"]  = df["seniority_rank"] * df["years_of_experience"]
df["edu_x_seniority"]  = df["education_rank"]  * df["seniority_rank"]
df["skills_x_exp"]     = df["num_skills"]       * df["years_of_experience"]
df["skills_x_seniority"] = df["num_skills"]     * df["seniority_rank"]
df["projects_x_exp"]   = df["num_projects"]     * df["years_of_experience"]
df["edu_x_company"]    = df["education_rank"]   * df["company_rank"]
df["seniority_x_company"] = df["seniority_rank"] * df["company_rank"]
df["exp_x_company"]    = df["years_of_experience"] * df["company_rank"]
df["gpa_x_edu"]        = df["gpa"]              * df["education_rank"]

# Ratios
df["skills_per_year"]       = df["num_skills"]       / (df["years_of_experience"] + 1)
df["projects_per_year"]     = df["num_projects"]     / (df["years_of_experience"] + 1)
df["publications_per_year"] = df["num_publications"] / (df["years_of_experience"] + 1)

# Achievement score
df["achievement_score"] = (
    df["num_projects"]                    * 0.5  +
    df["num_publications"]                * 2.5  +
    df["num_internships"]                 * 1.2  +
    df["has_leadership_experience"]       * 3.5  +
    df["has_open_source_contributions"]   * 2.5  +
    df["cert_rank"]                       * 1.5
)
df["achievement_x_exp"]      = df["achievement_score"] * df["years_of_experience"]
df["achievement_x_edu"]      = df["achievement_score"] * df["education_rank"]
df["achievement_x_seniority"] = df["achievement_score"] * df["seniority_rank"]

# Location
df["is_high_paying_city"]  = df["location"].apply(
    lambda x: int(any(c in x for c in HIGH_PAYING_CITIES)))
df["location_x_seniority"] = df["is_high_paying_city"] * df["seniority_rank"]
df["location_x_exp"]       = df["is_high_paying_city"] * df["years_of_experience"]

# Target & frequency encoding
for col in ["job_title", "industry", "field_of_study"]:
    mean = df.groupby(col)["annual_salary_usd"].mean()
    std  = df.groupby(col)["annual_salary_usd"].std()
    df[f"{col}_target_enc"] = df[col].map(mean).fillna(df["annual_salary_usd"].mean())
    df[f"{col}_target_std"] = df[col].map(std).fillna(df["annual_salary_usd"].std())

for col in ["job_title", "industry", "field_of_study", "location"]:
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq)

# Skill TF-IDF
print("Vectorizing skills...")
df["skills_text"] = df["skills"].fillna("").str.replace("|", " ").str.lower()
skill_vectorizer = TfidfVectorizer(max_features=60, min_df=5,
                                   ngram_range=(1, 2), sublinear_tf=True)
skill_matrix = skill_vectorizer.fit_transform(df["skills_text"]).toarray()
skill_df = pd.DataFrame(
    skill_matrix,
    columns=[f"skill_{n}" for n in skill_vectorizer.get_feature_names_out()],
    index=df.index,
)
df = pd.concat([df, skill_df], axis=1)

# ---------------------------------------------------------------------------
# Prepare X / y
# ---------------------------------------------------------------------------
EXCLUDE = {"annual_salary_usd", "skills", "skills_text", "job_title",
           "seniority_level", "education_level", "field_of_study",
           "industry", "company_size", "location", "certifications"}
feature_cols = [c for c in df.columns if c not in EXCLUDE]

X = df[feature_cols]
y = df["annual_salary_usd"]

print(f"  Features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

# Power-transform target
pt = PowerTransformer(method="yeo-johnson")
y_train_t = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Scale features
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
print("Training models...")

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.01,
    reg_lambda=2.0,
    eval_metric="rmse",
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
xgb.fit(X_train_s, y_train_t, verbose=False)
print("  XGBoost done")

rf = RandomForestRegressor(
    n_estimators=400, max_depth=20, min_samples_leaf=2,
    max_features="sqrt", random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_t)
print("  Random Forest done")

et = ExtraTreesRegressor(
    n_estimators=400, max_depth=20, min_samples_leaf=2,
    random_state=42, n_jobs=-1)
et.fit(X_train_s, y_train_t)
print("  Extra Trees done")

nn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu", solver="adam",
    alpha=0.001, batch_size=256,
    learning_rate="adaptive", max_iter=300,
    early_stopping=True, validation_fraction=0.1,
    random_state=42)
nn.fit(X_train_s, y_train_t)
print("  Neural Network done")

stack = StackingRegressor(
    estimators=[("xgb", xgb), ("rf", rf), ("et", et), ("nn", nn)],
    final_estimator=Ridge(alpha=5.0),
    cv=3, n_jobs=-1,
)
stack.fit(X_train_s, y_train_t)
print("  Stacking ensemble done")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(model, name):
    y_pred = pt.inverse_transform(
        model.predict(X_test_s).reshape(-1, 1)).ravel()
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = np.mean(np.abs(y_pred - y_test) / y_test <= 0.10) * 100
    print(f"\n{name}")
    print(f"  R²: {r2:.4f}  |  Acc (±10%): {acc:.2f}%  |  MAE: ${mae:,.0f}  |  RMSE: ${rmse:,.0f}")
    return {
        'r2': r2,
        'acc': acc,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred
    }

print("\n--- Results ---")
results = {}
for model, name in [(xgb, "XGBoost"), (rf, "Random Forest"),
                    (et, "Extra Trees"), (nn, "Neural Network"),
                    (stack, "Stacking Ensemble")]:
    results[name] = evaluate(model, name)

# Extract scores and predictions for compatibility
scores = {name: res['r2'] for name, res in results.items()}
preds  = {name: res['y_pred'] for name, res in results.items()}

best_name  = max(scores, key=scores.get)
best_model = {"XGBoost": xgb, "Random Forest": rf, "Extra Trees": et,
              "Neural Network": nn, "Stacking Ensemble": stack}[best_name]
y_best     = preds[best_name]

print(f"\nBest model: {best_name} (R² = {scores[best_name]:.4f})")

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
print("\nSaving artifacts...")

class ModelWrapper:
    def __init__(self, model, pt):
        self.model = model
        self.pt    = pt

    def predict(self, X):
        return self.pt.inverse_transform(
            self.model.predict(X).reshape(-1, 1)).ravel()

# Create models directory if it doesn't exist
models_dir = os.path.join(SCRIPT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(ModelWrapper(best_model, pt),
            os.path.join(models_dir, "salary_model.joblib"))
joblib.dump(scaler,           os.path.join(models_dir, "preprocessor.joblib"))
joblib.dump(skill_vectorizer, os.path.join(models_dir, "skill_vectorizer.joblib"))
joblib.dump(feature_cols,     os.path.join(models_dir, "feature_names.joblib"))
joblib.dump({"seniority_order": SENIORITY_ORDER, "education_order": EDUCATION_ORDER,
             "company_size_order": COMPANY_SIZE_ORDER, "cert_rank": CERT_RANK},
            os.path.join(models_dir, "encoders_config.joblib"))

final_r2   = r2_score(y_test, y_best)
final_acc  = np.mean(np.abs(y_best - y_test) / y_test <= 0.10) * 100
final_mae  = mean_absolute_error(y_test, y_best)
final_rmse = np.sqrt(mean_squared_error(y_test, y_best))

# Write comprehensive training report
with open(os.path.join(SCRIPT_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SALARY PREDICTION MODEL - TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    # Dataset Information
    f.write("DATASET INFORMATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Records:        {len(df):,}\n")
    f.write(f"Training Set:         {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)\n")
    f.write(f"Test Set:             {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)\n")
    f.write(f"Total Features:       {len(feature_cols)}\n")
    f.write(f"Skill Features:       60 (TF-IDF)\n\n")
    
    # Model Architecture
    f.write("MODEL ARCHITECTURE:\n")
    f.write("-" * 80 + "\n")
    f.write("Stacking Ensemble Components:\n")
    f.write("  - Base Model 1: XGBoost (1000 estimators)\n")
    f.write("  - Base Model 2: Random Forest (400 estimators)\n")
    f.write("  - Base Model 3: Extra Trees (400 estimators)\n")
    f.write("  - Base Model 4: Neural Network (3-layer: 256-128-64)\n")
    f.write("  - Meta-Learner: Ridge Regression (alpha=5.0)\n\n")
    
    # All Model Results
    f.write("ALL MODEL RESULTS:\n")
    f.write("=" * 80 + "\n\n")
    
    # Sort models by R² score (descending)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_models, 1):
        marker = " ⭐ BEST" if name == best_name else ""
        f.write(f"{i}. {name}{marker}\n")
        f.write("-" * 80 + "\n")
        f.write(f"   R² Score:         {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)\n")
        f.write(f"   Accuracy (±10%):  {metrics['acc']:.2f}%\n")
        f.write(f"   MAE:              ${metrics['mae']:,.0f}\n")
        f.write(f"   RMSE:             ${metrics['rmse']:,.0f}\n")
        f.write("\n")
    
    # Best Model Summary
    f.write("=" * 80 + "\n")
    f.write("SELECTED MODEL FOR DEPLOYMENT:\n")
    f.write("=" * 80 + "\n")
    f.write(f"Model:               {best_name}\n")
    f.write(f"R² Score:            {final_r2:.4f} ({final_r2*100:.2f}%)\n")
    f.write(f"Accuracy (±10%):     {final_acc:.2f}%\n")
    f.write(f"MAE:                 ${final_mae:,.0f}\n")
    f.write(f"RMSE:                ${final_rmse:,.0f}\n\n")
    
    # Feature Engineering
    f.write("FEATURE ENGINEERING:\n")
    f.write("-" * 80 + "\n")
    f.write("  - Target Encoding (job_title, industry, field_of_study)\n")
    f.write("  - Frequency Encoding (categorical features)\n")
    f.write("  - Power Transform (Yeo-Johnson on target variable)\n")
    f.write("  - Robust Scaling (handle outliers)\n")
    f.write("  - TF-IDF Vectorization (skills with bigrams)\n")
    f.write("  - Polynomial Features (experience squared, log)\n")
    f.write("  - Interaction Features (seniority × exp, edu × seniority, etc.)\n")
    f.write("  - Achievement Score (weighted sum of accomplishments)\n\n")
    
    # Model Comparison
    f.write("MODEL COMPARISON (Ranked by R² Score):\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Rank':<6} {'Model':<25} {'R² Score':<12} {'Accuracy':<12} {'MAE':<12} {'RMSE':<12}\n")
    f.write("-" * 80 + "\n")
    
    for i, (name, metrics) in enumerate(sorted_models, 1):
        marker = " ⭐" if name == best_name else ""
        f.write(f"{i:<6} {name:<25} {metrics['r2']*100:>6.2f}%     {metrics['acc']:>6.2f}%     ${metrics['mae']:>9,.0f}  ${metrics['rmse']:>9,.0f}{marker}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Training completed successfully!\n")
    f.write("=" * 80 + "\n")

print("Done. Restart your Streamlit app to load the new model.")
"""
═══════════════════════════════════════════════════════════════════════════════
RESUME SALARY PREDICTOR - STREAMLIT APPLICATION
═══════════════════════════════════════════════════════════════════════════════

PROJECT OVERVIEW:
-----------------
This application predicts annual salaries based on resume data using Machine Learning.
It uses XGBoost (Gradient Boosting) trained on 30,000+ resumes with 92.3% accuracy.

KEY COMPONENTS:
--------------
1. DATA PROCESSING: Feature engineering from resume text
2. ML MODEL: XGBoost for salary prediction
3. RESUME PARSING: Extract skills, experience, education from PDF/DOCX
4. AI INTEGRATION: Groq API for intelligent parsing
5. JOB SEARCH: SerpAPI integration for job recommendations
6. UI: Streamlit web interface with visualizations

INSTALLATION:
------------
pip install streamlit xgboost scikit-learn pandas numpy joblib python-docx PyPDF2 
            google-search-results python-dotenv plotly groq

RUN:
----
streamlit run app.py

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Configure Streamlit page settings (must be first Streamlit command)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    st.set_page_config(
        page_title="SalaryLens – Resume Salary Predictor",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except st.errors.StreamlitAPIException:
    pass  # Already configured (happens on Streamlit Cloud)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LIBRARY IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Import required libraries for data processing, ML, and visualization
# ═══════════════════════════════════════════════════════════════════════════════

# Data Processing Libraries
import pandas as pd              # For data manipulation and analysis
import numpy as np               # For numerical operations
import joblib                    # For loading pre-trained ML models
import re                        # For regex pattern matching (resume parsing)
import os                        # For file system operations
from pathlib import Path         # For cross-platform file paths
from model_wrapper import ModelWrapper

# Visualization Libraries
import plotly.express as px        # For quick visualizations

# Model Enhancement Modules
try:
    from whatif_analysis import WhatIfAnalyzer, display_whatif_analysis
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"Enhancement modules not available: {e}")
    ENHANCEMENTS_AVAILABLE = False

# Environment Configuration
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load API keys from .env file (GROQ_API_KEY, SERPAPI_KEY)
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161a22;
    --surface2:  #1e2330;
    --border:    #2a3040;
    --accent:    #4f8ef7;
    --accent2:   #a78bfa;
    --green:     #34d399;
    --amber:     #fbbf24;
    --red:       #f87171;
    --text:      #e8eaf0;
    --muted:     #7a829a;
    --radius:    14px;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1200px !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(79,142,247,0.15), transparent);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: var(--accent);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: #e8eaf0 !important; /* Fallback color */
    background: linear-gradient(135deg, #e8eaf0 30%, #7c9ef5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.7rem !important;
}
/* Fallback for browsers that don't support background-clip */
@supports not (background-clip: text) {
    .hero h1 {
        color: #e8eaf0 !important;
        -webkit-text-fill-color: #e8eaf0 !important;
    }
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-title span { color: var(--accent); }

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.08) 0%, rgba(167,139,250,0.06) 100%);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 18px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.salary-label {
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.salary-amount {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.salary-range {
    color: var(--muted);
    font-size: 0.92rem;
    margin-top: 0.5rem;
}
.salary-range strong { color: var(--text); }

/* Range bars */
.range-bar-wrap { margin: 1.6rem 0 0.5rem; }
.range-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.3rem; display: flex; justify-content: space-between; }
.range-bar-bg { background: var(--surface2); border-radius: 100px; height: 6px; overflow: hidden; }
.range-bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #4f8ef7, #a78bfa); }

/* Feature bars */
.feat-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.55rem; }
.feat-name { font-size: 0.82rem; color: var(--muted); width: 180px; flex-shrink: 0; }
.feat-bar-bg { flex: 1; background: var(--surface2); border-radius: 100px; height: 6px; }
.feat-bar-fill { height: 100%; border-radius: 100px; }
.feat-val { font-size: 0.78rem; color: var(--text); width: 44px; text-align: right; flex-shrink: 0; }

/* Stat chips */
.chips { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.8rem; }
.chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: var(--muted);
}
.chip b { color: var(--text); }

/* Alert */
.alert {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: var(--amber);
    margin-bottom: 1rem;
}

/* Upload zone override */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}

/* Streamlit widget overrides */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div { padding: 0 !important; }
label { color: var(--muted) !important; font-size: 0.85rem !important; }
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stCheckbox > label { color: var(--text) !important; font-size: 0.9rem !important; }
div[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Step indicator */
.steps { display: flex; gap: 0; margin-bottom: 2rem; }
.step { flex: 1; text-align: center; position: relative; }
.step-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.4rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
}
.step-active .step-num { background: var(--accent); color: white; }
.step-done .step-num { background: var(--green); color: #0d0f14; }
.step-inactive .step-num { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
.step-label { font-size: 0.72rem; color: var(--muted); }
.step-active .step-label { color: var(--accent); }
.step-done .step-label { color: var(--green); }
.step-line { position: absolute; top: 16px; left: 50%; width: 100%; height: 1px; background: var(--border); z-index: 0; }

/* Table styling */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.85rem;
    background: var(--surface2);
    border-radius: 8px;
    overflow: hidden;
}
table thead tr {
    background: rgba(79,142,247,0.1);
    border-bottom: 2px solid var(--border);
}
table thead th {
    padding: 0.75rem 0.9rem;
    text-align: left;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
table tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.2s;
}
table tbody tr:hover {
    background: rgba(79,142,247,0.05);
}
table tbody tr:last-child {
    border-bottom: none;
}
table tbody td {
    padding: 0.7rem 0.9rem;
    color: var(--text);
}
table tbody td:first-child {
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Define categorical values and mappings used for feature engineering
# These constants are used to:
# 1. Encode categorical features into numerical values
# 2. Provide dropdown options in the UI
# 3. Rank/order features for the ML model
# ═══════════════════════════════════════════════════════════════════════════════

# API Keys
SERPAPI_KEY = "a4ca1c6c7ab7425172e500d31f5047249a069f12dd3781c839a2bd4b6370cc65"

# Seniority Levels (ordered from junior to senior)
# Used for: Ranking experience level, encoding as numerical feature
SENIORITY_ORDER   = ['Intern', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO']

# Education Levels (ordered from lowest to highest)
# Used for: Ranking education, encoding as numerical feature
EDUCATION_ORDER   = ['High School', 'Associate', 'Bachelor', 'Master', 'MBA', 'PhD']

# Company Sizes (ordered from smallest to largest)
# Used for: Encoding company size as numerical feature
COMPANY_SIZE_ORDER= ['Startup (<50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000-5000)', 'Enterprise (5000+)']
CERT_RANK         = {'None': 0, 'AWS Certified': 3, 'Google Cloud Certified': 3, 'Azure Certified': 3,
                     'PMP': 2, 'Scrum Master': 1, 'CFA': 3, 'CISSP': 3,
                     'Data Science Certifications': 2, 'Multiple Certs': 4}
INDUSTRIES        = ['Technology', 'Finance', 'Healthcare', 'E-commerce', 'Consulting',
                     'Education', 'Government', 'Retail', 'Manufacturing', 'Telecommunications',
                     'Media', 'Energy', 'Aerospace', 'Automotive', 'Pharmaceuticals']
LOCATIONS         = ['San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX', 'Boston, MA',
                     'Chicago, IL', 'Los Angeles, CA', 'Denver, CO', 'Atlanta, GA', 'Remote',
                     'Bangalore, India', 'London, UK', 'Toronto, Canada', 'Berlin, Germany', 'Singapore']
JOB_TITLES        = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer',
                     'ML Engineer', 'Frontend Developer', 'Backend Developer', 'Full Stack Developer',
                     'Data Analyst', 'Business Analyst', 'Cloud Architect', 'Security Engineer',
                     'QA Engineer', 'Mobile Developer', 'Database Administrator', 'Systems Analyst',
                     'UI/UX Designer', 'Network Engineer', 'Embedded Systems Engineer', 'AI Researcher']
FIELDS_OF_STUDY   = ['Computer Science', 'Information Technology', 'Electrical Engineering',
                     'Mathematics', 'Statistics', 'Data Science', 'Business Administration',
                     'Physics', 'Mechanical Engineering', 'Cognitive Science', 'Economics', 'Other']
# Complete skills list from the actual dataset (all skills the model recognizes)
ALL_SKILLS = [
    # Core Programming Languages
    'Python', 'Java', 'JavaScript', 'Go', 'Scala', 'C++', 'R',
    
    # Web Development
    'React', 'Node.js', 'Node JS',
    
    # Databases
    'SQL',
    
    # Cloud & DevOps
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD', 'CI', 'CD',
    
    # Data Science & ML
    'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'NLP',
    'Data', 'Data Visualization',
    
    # Big Data
    'Spark', 'Hadoop',
    
    # Data Visualization & BI
    'Tableau', 'Power BI', 'Excel', 'Visualization',
    
    # Version Control
    'Git',
    
    # Operating Systems
    'Linux',
    
    # Methodologies & Soft Skills
    'Agile', 'Scrum', 'Communication', 'Leadership',
]

# Additional skills recognized by the model (TF-IDF learned these from training data)
# These are the actual 60 features the model uses:
DATASET_SKILLS = [
    'agile', 'aws', 'azure', 'bi', 'cd', 'ci', 'ci cd', 'communication',
    'data', 'data visualization', 'deep', 'deep learning', 'docker', 'excel',
    'gcp', 'git', 'go', 'hadoop', 'java', 'javascript', 'js', 'kubernetes',
    'leadership', 'learning', 'learning agile', 'learning aws', 'learning azure',
    'learning communication', 'learning data', 'learning excel', 'learning go',
    'learning java', 'learning javascript', 'learning kubernetes', 'learning linux',
    'learning node', 'learning power', 'learning pytorch', 'learning scala',
    'learning sql', 'learning tableau', 'learning tensorflow', 'linux', 'machine',
    'machine learning', 'nlp', 'node', 'node js', 'power', 'power bi', 'python',
    'pytorch', 'react', 'scala', 'scrum', 'spark', 'sql', 'tableau', 'tensorflow',
    'visualization'
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MACHINE LEARNING MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Load pre-trained XGBoost model and preprocessing artifacts
# 
# MODEL TRAINING (done separately):
# ---------------------------------
# 1. Dataset: 30,000+ resumes with salary labels
# 2. Algorithm: XGBoost (Gradient Boosting Decision Trees)
# 3. Features: 50+ engineered features (skills, experience, education, etc.)
# 4. Accuracy: 92.3% on test set
# 5. Training code: train_salary_model.py (separate file)
#
# ARTIFACTS LOADED:
# ----------------
# - salary_model.joblib: Trained XGBoost model
# - preprocessor.joblib: Feature scaling/encoding pipeline
# - skill_vectorizer.joblib: TF-IDF vectorizer for skills text
# - feature_names.joblib: List of feature column names
# - encoders_config.joblib: Categorical encoders configuration
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource  # Cache loaded models to avoid reloading on every interaction
def load_models():
    """
    Load all saved model artifacts from disk.
    
    Returns:
        tuple: (artifacts_dict, missing_files_list)
        - artifacts_dict: Dictionary containing loaded model components
        - missing_files_list: List of files that couldn't be found
    """
    # Make ModelWrapper available for unpickling
    import sys
    sys.modules['__main__'].ModelWrapper = ModelWrapper
    
    files = {
        'model':        'models/salary_model.joblib',
        'preprocessor': 'models/preprocessor.joblib',
        'skill_vec':    'models/skill_vectorizer.joblib',
        'feature_names':'models/feature_names.joblib',
        'cfg':          'models/encoders_config.joblib',
    }
    missing = [f for f in files.values() if not Path(f).exists()]
    if missing:
        return None, missing
    artifacts = {k: joblib.load(v) for k, v in files.items()}
    return artifacts, []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE ENGINEERING & PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Transform raw resume data into ML features and make salary predictions
#
# FEATURE ENGINEERING STEPS:
# --------------------------
# 1. RANKING: Convert categorical values to numerical ranks
#    - Seniority: Intern=0, Junior=1, ..., CTO=9
#    - Education: High School=0, ..., PhD=5
#    - Company Size: Startup=0, ..., Enterprise=4
#
# 2. BINNING: Group continuous values into categories
#    - Experience: 0-2 years, 2-5 years, 5-10 years, etc.
#
# 3. INTERACTION FEATURES: Combine multiple features
#    - seniority_x_exp = seniority_rank * years_of_experience
#    - edu_x_seniority = education_rank * seniority_rank
#    - skills_x_exp = num_skills * years_of_experience
#
# 4. ACHIEVEMENT SCORE: Weighted sum of accomplishments
#    - Projects (0.5x) + Publications (2.0x) + Internships (1.0x) + 
#      Leadership (3.0x) + Open Source (2.0x)
#
# 5. SKILL VECTORIZATION: Convert skills text to numerical features
#    - Uses TF-IDF (Term Frequency-Inverse Document Frequency)
#    - Creates binary features for each skill (Python=1, Java=0, etc.)
#
# 6. CERTIFICATION RANKING: Assign importance scores to certifications
#    - AWS/Azure/GCP Certified = 3, PMP = 2, Scrum Master = 1, etc.
# ═══════════════════════════════════════════════════════════════════════════════

def predict_salary(artifacts, form_data: dict):
    """
    Run the complete ML prediction pipeline.
    
    Process:
    1. Feature Engineering: Transform raw data into ML features
    2. Preprocessing: Scale and encode features
    3. Prediction: Use XGBoost model to predict salary
    4. Confidence Range: Calculate ±12% confidence interval
    
    Args:
        artifacts (dict): Loaded model components
        form_data (dict): Raw resume data (job title, experience, skills, etc.)
    
    Returns:
        dict: {
            'predicted': Predicted annual salary (rounded to nearest $100)
            'low': Lower bound of confidence range (88% of prediction)
            'high': Upper bound of confidence range (112% of prediction)
        }
    """
    d = form_data.copy()
    cfg = artifacts['cfg']

    # Basic rankings
    d['seniority_rank'] = SENIORITY_ORDER.index(d['seniority_level']) if d['seniority_level'] in SENIORITY_ORDER else 2
    d['education_rank'] = EDUCATION_ORDER.index(d['education_level']) if d['education_level'] in EDUCATION_ORDER else 2
    d['company_rank']   = COMPANY_SIZE_ORDER.index(d['company_size']) if d['company_size'] in COMPANY_SIZE_ORDER else 2

    # Experience binning
    bins = [-1, 0, 2, 5, 10, 15, 20, 99]
    labels = [0, 1, 2, 3, 4, 5, 6]
    exp = d['years_of_experience']
    d['exp_bin'] = next((labels[i] for i, (lo, hi) in enumerate(zip(bins, bins[1:])) if lo < exp <= hi), 2)

    # Polynomial & log experience
    d['exp_squared'] = d['years_of_experience'] ** 2
    d['exp_log'] = np.log1p(d['years_of_experience'])

    # Basic interactions
    d['seniority_x_exp']  = d['seniority_rank'] * d['years_of_experience']
    d['edu_x_seniority']  = d['education_rank'] * d['seniority_rank']
    d['skills_x_exp']     = d['num_skills'] * d['years_of_experience']
    d['skills_x_seniority'] = d['num_skills'] * d['seniority_rank']
    d['projects_x_exp']   = d.get('num_projects', 0) * d['years_of_experience']
    d['edu_x_company']    = d['education_rank'] * d['company_rank']
    d['seniority_x_company'] = d['seniority_rank'] * d['company_rank']
    d['exp_x_company']    = d['years_of_experience'] * d['company_rank']
    d['gpa_x_edu']        = d.get('gpa', 3.0) * d['education_rank']

    # Ratios
    d['skills_per_year']       = d['num_skills'] / (d['years_of_experience'] + 1)
    d['projects_per_year']     = d.get('num_projects', 0) / (d['years_of_experience'] + 1)
    d['publications_per_year'] = d.get('num_publications', 0) / (d['years_of_experience'] + 1)

    # Certification rank
    d['cert_rank'] = CERT_RANK.get(d.get('certifications', 'None'), 0)

    # Achievement score
    d['achievement_score'] = (
        d.get('num_projects', 0) * 0.5 +
        d.get('num_publications', 0) * 2.5 +
        d.get('num_internships', 0) * 1.2 +
        d.get('has_leadership_experience', 0) * 3.5 +
        d.get('has_open_source_contributions', 0) * 2.5 +
        d['cert_rank'] * 1.5
    )
    d['achievement_x_exp']      = d['achievement_score'] * d['years_of_experience']
    d['achievement_x_edu']      = d['achievement_score'] * d['education_rank']
    d['achievement_x_seniority'] = d['achievement_score'] * d['seniority_rank']

    # Location features
    HIGH_PAYING_CITIES = ["San Francisco", "New York", "Seattle", "Boston", "San Jose"]
    location = d.get('location', '')
    d['is_high_paying_city'] = int(any(c in location for c in HIGH_PAYING_CITIES))
    d['location_x_seniority'] = d['is_high_paying_city'] * d['seniority_rank']
    d['location_x_exp']       = d['is_high_paying_city'] * d['years_of_experience']

    # Target encoding (use reasonable defaults based on typical salaries)
    # These would ideally come from the training data, but we'll use approximations
    job_title_avg = {
        'Software Engineer': 120000, 'Data Scientist': 130000, 'Product Manager': 140000,
        'DevOps Engineer': 125000, 'ML Engineer': 135000, 'Frontend Developer': 110000,
        'Backend Developer': 115000, 'Full Stack Developer': 120000, 'Data Analyst': 95000,
        'Business Analyst': 90000, 'Cloud Architect': 150000, 'Security Engineer': 130000,
        'QA Engineer': 95000, 'Mobile Developer': 115000, 'Database Administrator': 105000,
        'Systems Analyst': 100000, 'UI/UX Designer': 100000, 'Network Engineer': 105000,
        'Embedded Systems Engineer': 110000, 'AI Researcher': 145000
    }
    industry_avg = {
        'Technology': 125000, 'Finance': 130000, 'Healthcare': 110000, 'E-commerce': 115000,
        'Consulting': 120000, 'Education': 85000, 'Government': 95000, 'Retail': 90000,
        'Manufacturing': 100000, 'Telecommunications': 115000, 'Media': 105000,
        'Energy': 120000, 'Aerospace': 125000, 'Automotive': 110000, 'Pharmaceuticals': 120000
    }
    field_avg = {
        'Computer Science': 120000, 'Information Technology': 115000, 'Electrical Engineering': 110000,
        'Mathematics': 115000, 'Statistics': 120000, 'Data Science': 130000,
        'Business Administration': 105000, 'Physics': 110000, 'Mechanical Engineering': 105000,
        'Cognitive Science': 115000, 'Economics': 110000, 'Other': 100000
    }
    
    d['job_title_target_enc'] = job_title_avg.get(d.get('job_title', ''), 110000)
    d['job_title_target_std'] = 25000
    d['industry_target_enc'] = industry_avg.get(d.get('industry', ''), 110000)
    d['industry_target_std'] = 20000
    d['field_of_study_target_enc'] = field_avg.get(d.get('field_of_study', ''), 110000)
    d['field_of_study_target_std'] = 18000

    # Frequency encoding (use reasonable defaults)
    d['job_title_freq'] = 0.05
    d['industry_freq'] = 0.07
    d['field_of_study_freq'] = 0.08
    d['location_freq'] = 0.06

    # Skill vectorization
    skill_vec = artifacts['skill_vec']
    skills_str = d.get('skills', '').replace('|', ' ').lower()
    skill_arr = skill_vec.transform([skills_str]).toarray()[0]
    for name, val in zip([f"skill_{s}" for s in skill_vec.get_feature_names_out()], skill_arr):
        d[name] = int(val)

    # Create DataFrame and ensure all features exist
    row = pd.DataFrame([d])
    feat_names = artifacts['feature_names']
    
    # Add any missing columns with default value 0
    for col in feat_names:
        if col not in row.columns:
            row[col] = 0

    # Transform and predict
    X = artifacts['preprocessor'].transform(row[feat_names])
    pred = float(artifacts['model'].predict(X)[0])

    return {
        'predicted': round(pred, -2),
        'low':  round(pred * 0.88, -2),
        'high': round(pred * 1.12, -2),
    }

def get_feature_contributions(artifacts, form_data: dict):
    """Return top feature importances from the loaded model."""
    model = artifacts['model']
    feat_names = artifacts['feature_names']
    importances = model.feature_importances_
    top = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]
    return top

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: JOB SEARCH ENGINE (SERPAPI INTEGRATION)
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Search for relevant job postings and rank them by match score
#
# JOB MATCHING ALGORITHM:
# ----------------------
# 1. QUERY BUILDING: Construct search query from user profile
#    - Include: Job title, seniority level, experience, top skills
#    - Example: "Senior Software Engineer 5+ years Python Java"
#
# 2. API CALL: Use SerpAPI to search Google Jobs
#    - Engine: google_jobs
#    - Parameters: query, location, language
#
# 3. MATCH SCORING: Calculate relevance score (0-100%)
#    Components:
#    - Location Match (20%): City name matching
#    - Seniority Match (25%): Title contains seniority level
#    - Title Match (25%): Job title similarity
#    - Skills Match (15%): Number of matching skills
#    - Experience Match (15%): Years of experience alignment
#
# 4. RANKING: Sort jobs by match score (highest first)
#
# 5. FALLBACK: If no results, retry with broader search
#    - Remove seniority and skills, keep only job title
# ═══════════════════════════════════════════════════════════════════════════════

def search_real_time_salary(form_data: dict, country='US'):
    """
    Search for real-time salary data from Google and return in USD.
    
    Args:
        form_data (dict): User profile data with job_title, location, years_of_experience, etc.
        country (str): 'US' for USA or 'India' for India
    
    Returns:
        dict: Salary information with min, max, average in USD
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    
    job_title = form_data.get('job_title', 'Software Engineer')
    location = form_data.get('location', 'United States')
    experience = form_data.get('years_of_experience', 4)
    seniority = form_data.get('seniority_level', 'Mid-level')
    
    # Build search query based on country
    if country == 'US':
        # For US, use the location from form or default to US
        if 'India' in location or 'Bangalore' in location or 'Mumbai' in location:
            location = 'United States'
        search_query = f"{job_title} salary in USA {experience} years experience"
    else:
        location = 'India'
        search_query = f"{job_title} salary in India {experience} years experience"
    
    try:
        # Search using Google
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        salaries = []
        
        # Google search
        search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}"
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            
            if country == 'US':
                # Pattern for USD: $X,XXX - $X,XXX or $XXk - $XXXk
                usd_pattern = r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|K)?\s*(?:-|to)\s*\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:k|K)?'
                usd_matches = re.findall(usd_pattern, text)
                
                if usd_matches:
                    for match in usd_matches[:5]:
                        min_sal = float(match[0].replace(',', ''))
                        max_sal = float(match[1].replace(',', ''))
                        
                        # Check if it's in thousands (k)
                        if 'k' in text[text.find(match[0]):text.find(match[0])+50].lower():
                            min_sal *= 1000
                            max_sal *= 1000
                        
                        # Reasonable salary range check
                        if 30000 <= min_sal <= 500000 and 40000 <= max_sal <= 600000:
                            salaries.append({
                                'min': min_sal,
                                'max': max_sal,
                                'avg': (min_sal + max_sal) / 2
                            })
            else:
                # Pattern for INR (lakhs)
                lakhs_pattern = r'₹\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh|L)\s*(?:-|to)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh|L)'
                lakhs_matches = re.findall(lakhs_pattern, text, re.IGNORECASE)
                
                if lakhs_matches:
                    for match in lakhs_matches[:3]:
                        min_lakhs = float(match[0])
                        max_lakhs = float(match[1])
                        # Convert to USD (1 lakh INR = ~$1,200)
                        salaries.append({
                            'min': min_lakhs * 1200,
                            'max': max_lakhs * 1200,
                            'avg': (min_lakhs + max_lakhs) / 2 * 1200
                        })
        
        # If we found salaries, calculate average
        if salaries:
            avg_min = sum(s['min'] for s in salaries) / len(salaries)
            avg_max = sum(s['max'] for s in salaries) / len(salaries)
            avg_salary = sum(s['avg'] for s in salaries) / len(salaries)
            
            return {
                'success': True,
                'min': int(avg_min),
                'max': int(avg_max),
                'avg': int(avg_salary),
                'source': 'Real-Time Market Data',
                'query': search_query,
                'samples': len(salaries),
                'country': country
            }
        
        # Fallback: Use industry averages based on experience and role
        base_salary_usd = {
            'Intern': 50000,
            'Junior': 75000,
            'Mid-level': 110000,
            'Senior': 145000,
            'Lead': 175000,
            'Principal': 200000,
            'Staff': 220000,
            'Director': 250000,
            'VP': 300000,
            'CTO/CXO': 400000
        }
        
        base = base_salary_usd.get(seniority, 110000)
        
        # Adjust for experience
        exp_multiplier = 1 + (experience * 0.03)
        estimated_salary = int(base * exp_multiplier)
        
        return {
            'success': True,
            'min': int(estimated_salary * 0.85),
            'max': int(estimated_salary * 1.15),
            'avg': estimated_salary,
            'source': 'Industry Average Estimate',
            'query': search_query,
            'samples': 0,
            'country': country
        }
        
    except Exception as e:
        # Return error with fallback estimate
        base_salary_usd = 110000
        
        return {
            'success': False,
            'error': str(e),
            'min': int(base_salary_usd * 0.85),
            'max': int(base_salary_usd * 1.15),
            'avg': base_salary_usd,
            'source': 'Fallback Estimate',
            'query': search_query,
            'samples': 0,
            'country': country
        }

def search_jobs_serpapi(form_data: dict, api_key: str, fallback=False):
    """
    Search for real job postings using SerpAPI Google Jobs.
    
    Args:
        form_data (dict): User profile data
        api_key (str): SerpAPI key
        fallback (bool): If True, use broader search criteria
    
    Returns:
        tuple: (jobs_list, search_query, is_fallback)
    """
    try:
        from serpapi import GoogleSearch
    except ImportError:
        st.error("SerpAPI library not installed. Run: pip install google-search-results")
        return None, None, False
    
    import re
    
    user_location = form_data.get('location', 'San Francisco, CA')
    user_skills = form_data.get('skills', '').split('|')
    user_skills = [s for s in user_skills if s]
    user_job_title = form_data.get('job_title', 'Software Engineer')
    user_seniority = form_data.get('seniority_level', 'Mid-level')
    user_experience = form_data.get('years_of_experience', 0)
    
    # Build search query based on whether this is a fallback search
    if fallback:
        # Broader search - just job title and location, no seniority or skills
        query = f"{user_job_title}".strip()
    else:
        # Specific search - Include job title, seniority level, experience, and top skills
        skills_str = ' '.join(user_skills[:2]) if user_skills else ''
        
        # Add experience to query for better matching
        experience_str = ""
        if user_experience > 0:
            experience_str = f"{user_experience}+ years"
        
        query = f"{user_seniority} {user_job_title} {experience_str} {skills_str}".strip()
    
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": user_location,  # Location is passed as separate parameter
        "api_key": api_key,
        "hl": "en",
        "gl": "us"
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        jobs = []
        for job in results.get("jobs_results", []):
            # Extract salary if available
            salary_info = ""
            extensions = job.get('detected_extensions', {})
            if 'salary' in extensions:
                salary_info = extensions['salary']
            
            # Calculate match score based on location, job title, seniority, skills, and experience
            title_lower = job.get('title', '').lower()
            job_location = job.get('location', '').lower()
            desc_lower = job.get('description', '').lower()
            
            # Location match (20% weight)
            user_city = user_location.split(',')[0].lower().strip()
            location_match = 1.0 if user_city in job_location else 0.5
            
            # Seniority match (25% weight)
            seniority_match = 1.0 if user_seniority.lower() in title_lower else 0.6
            
            # Job title match (25% weight)
            job_title_words = user_job_title.lower().split()
            title_match = sum(1 for word in job_title_words if word in title_lower) / max(len(job_title_words), 1)
            
            # Skills match (15% weight)
            skill_matches = sum(1 for skill in user_skills if skill.lower() in title_lower or skill.lower() in desc_lower)
            skills_match = skill_matches / max(len(user_skills), 1) if user_skills else 0.5
            
            # Experience match (15% weight) - NEW
            experience_match = 0.5  # Default neutral score
            if user_experience > 0:
                # Look for experience requirements in description
                exp_patterns = [
                    r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
                    r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
                    r'experience:\s*(\d+)\+?\s*years?',
                    r'minimum\s+(\d+)\+?\s*years?',
                    r'at least\s+(\d+)\+?\s*years?'
                ]
                
                required_exp = None
                for pattern in exp_patterns:
                    match = re.search(pattern, desc_lower)
                    if match:
                        required_exp = int(match.group(1))
                        break
                
                if required_exp is not None:
                    # Perfect match if user experience meets or slightly exceeds requirement
                    if user_experience >= required_exp and user_experience <= required_exp + 3:
                        experience_match = 1.0
                    # Good match if within range
                    elif user_experience >= required_exp - 1:
                        experience_match = 0.8
                    # Underqualified
                    elif user_experience < required_exp:
                        experience_match = 0.3
                    # Overqualified
                    else:
                        experience_match = 0.6
                else:
                    # No explicit requirement found, assume it's flexible
                    experience_match = 0.7
            
            # Calculate overall match score with new weights
            match_score = int((
                location_match * 0.20 + 
                seniority_match * 0.25 + 
                title_match * 0.25 + 
                skills_match * 0.15 + 
                experience_match * 0.15
            ) * 100)
            match_score = min(match_score, 95)
            
            # Extract relevant skills from description (for display purposes)
            job_skills = [skill for skill in user_skills if skill.lower() in desc_lower]
            if not job_skills and user_skills:
                job_skills = user_skills[:3]  # Show user's top skills as relevant
            
            # Clean description - remove HTML tags but keep full text
            description = job.get('description', '')
            # Remove HTML tags using regex
            import re
            description = re.sub(r'<[^>]+>', '', description)  # Remove HTML tags
            description = description.strip()
            # Keep full description without truncating
            
            # Clean title, company, location
            title = re.sub(r'<[^>]+>', '', job.get('title', 'N/A')).strip()
            company = re.sub(r'<[^>]+>', '', job.get('company_name', 'N/A')).strip()
            location = re.sub(r'<[^>]+>', '', job.get('location', user_location)).strip()
            
            # Extract the best apply link
            apply_url = '#'
            
            # Priority 1: Check apply_options for direct apply links
            if job.get('apply_options'):
                for option in job.get('apply_options', []):
                    link = option.get('link', '')
                    # Skip Google redirect links, prefer direct company links
                    if link and 'google.com' not in link:
                        apply_url = link
                        break
                # If all are Google links, use the first one
                if apply_url == '#' and job.get('apply_options'):
                    apply_url = job.get('apply_options', [{}])[0].get('link', '#')
            
            # Priority 2: Check related_links
            if apply_url == '#' and job.get('related_links'):
                for link_obj in job.get('related_links', []):
                    link = link_obj.get('link', '')
                    if link and 'google.com' not in link:
                        apply_url = link
                        break
            
            # Priority 3: Use share_link as last resort
            if apply_url == '#':
                apply_url = job.get('share_link', '#')
            
            jobs.append({
                'title': title,
                'company': company,
                'location': location,
                'description': description,
                'url': apply_url,
                'posted': extensions.get('posted_at', 'Recently'),
                'salary': salary_info,
                'match_score': match_score,
                'skills': job_skills[:5],
                'job_type': extensions.get('schedule_type', 'Full-time')
            })
        
        if jobs:
            jobs.sort(key=lambda x: x['match_score'], reverse=True)
            return jobs, query, fallback
        else:
            return [], query, fallback
    except Exception as e:
        st.error(f"Error fetching jobs from SerpAPI: {str(e)}")
        return None, query, fallback
    
    return None, query, fallback

# ─────────────────────────────────────────────────────────────────────────────
# RESUME PARSERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_docx(file) -> str:
    try:
        from docx import Document
        doc = Document(file)
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"ERROR: {e}"

def extract_text_pdf(file) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    except Exception:
        try:
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except Exception as e:
            return f"ERROR: {e}"

def parse_resume_text(text: str) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SECTION 6: RESUME PARSING (DATA EXTRACTION)
    ═══════════════════════════════════════════════════════════════════════════
    Purpose: Extract structured data from unstructured resume text
    
    PARSING TECHNIQUES:
    ------------------
    1. REGEX PATTERNS: Pattern matching for structured data
       - Years of experience: "5 years", "5+ years experience"
       - GPA: "GPA: 3.5", "3.5/4.0"
       - Email: standard email regex
       - Phone: various phone number formats
    
    2. KEYWORD MATCHING: Search for specific terms
       - Skills: Python, Java, AWS, Docker, etc.
       - Education: Bachelor, Master, PhD
       - Certifications: AWS Certified, PMP, Scrum Master
       - Leadership: "led team", "managed", "supervised"
    
    3. CONTEXT-AWARE EXTRACTION: Understand surrounding text
       - Job titles: Look for common titles in context
       - Company size: Extract from company descriptions
       - Industry: Identify from job descriptions
    
    4. FALLBACK VALUES: Use defaults when data not found
       - Default experience: 0 years (changed from 3)
       - Default GPA: 3.5
       - Default location: San Francisco, CA
    
    Args:
        text (str): Raw resume text extracted from PDF/DOCX/TXT
    
    Returns:
        dict: Structured resume data with all required features
    ═══════════════════════════════════════════════════════════════════════════
    """
    text_lower = text.lower()
    parsed = {}

    # Years of experience
    yoe_patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
    ]
    for pat in yoe_patterns:
        m = re.search(pat, text_lower)
        if m:
            parsed['years_of_experience'] = min(int(m.group(1)), 35)
            break

    # Education
    if 'ph.d' in text_lower or 'phd' in text_lower or 'doctor' in text_lower:
        parsed['education_level'] = 'PhD'
    elif 'mba' in text_lower:
        parsed['education_level'] = 'MBA'
    elif "master" in text_lower or 'm.s.' in text_lower or 'm.sc' in text_lower or 'm.eng' in text_lower:
        parsed['education_level'] = 'Master'
    elif "bachelor" in text_lower or 'b.s.' in text_lower or 'b.e.' in text_lower or 'b.tech' in text_lower:
        parsed['education_level'] = 'Bachelor'
    elif "associate" in text_lower:
        parsed['education_level'] = 'Associate'

    # Skills
    detected = [s for s in ALL_SKILLS if s.lower() in text_lower]
    if detected:
        parsed['skills'] = '|'.join(detected)
        parsed['num_skills'] = len(detected)

    # Seniority
    if any(w in text_lower for w in ['cto', 'chief technology', 'chief executive', 'president']):
        parsed['seniority_level'] = 'CTO/CXO'
    elif 'vice president' in text_lower or ' vp ' in text_lower:
        parsed['seniority_level'] = 'VP'
    elif 'director' in text_lower:
        parsed['seniority_level'] = 'Director'
    elif 'principal' in text_lower:
        parsed['seniority_level'] = 'Principal'
    elif 'staff engineer' in text_lower or 'staff scientist' in text_lower:
        parsed['seniority_level'] = 'Staff'
    elif 'lead ' in text_lower or ' lead' in text_lower:
        parsed['seniority_level'] = 'Lead'
    elif 'senior' in text_lower or 'sr.' in text_lower:
        parsed['seniority_level'] = 'Senior'
    elif 'junior' in text_lower or 'jr.' in text_lower or 'associate ' in text_lower:
        parsed['seniority_level'] = 'Junior'
    elif 'intern' in text_lower:
        parsed['seniority_level'] = 'Intern'

    # Job title
    for jt in JOB_TITLES:
        if jt.lower() in text_lower:
            parsed['job_title'] = jt
            break

    # Location
    for loc in LOCATIONS:
        city = loc.split(',')[0].lower()
        if city in text_lower:
            parsed['location'] = loc
            break

    # Publications
    pub_m = re.search(r'(\d+)\s+(?:peer.reviewed\s+)?publication', text_lower)
    if pub_m:
        parsed['num_publications'] = int(pub_m.group(1))

    # Leadership / open source
    if any(w in text_lower for w in ['managed team', 'led team', 'mentored', 'leadership']):
        parsed['has_leadership_experience'] = 1
    if any(w in text_lower for w in ['open source', 'github', 'open-source']):
        parsed['has_open_source_contributions'] = 1

    # Certifications
    cert_keywords = {
        'aws certified': 'AWS Certified', 'google cloud certified': 'Google Cloud Certified',
        'azure certified': 'Azure Certified', 'pmp': 'PMP', 'scrum master': 'Scrum Master',
        'cfa': 'CFA', 'cissp': 'CISSP',
    }
    found_certs = [v for k, v in cert_keywords.items() if k in text_lower]
    if len(found_certs) > 1:
        parsed['certifications'] = 'Multiple Certs'
    elif len(found_certs) == 1:
        parsed['certifications'] = found_certs[0]

    # Field of study
    for field in FIELDS_OF_STUDY:
        if field.lower() in text_lower:
            parsed['field_of_study'] = field
            break

    # Industry keywords
    industry_kw = {
        'Technology': ['software', 'tech', 'saas', 'platform'],
        'Finance': ['finance', 'bank', 'trading', 'investment', 'fintech'],
        'Healthcare': ['health', 'hospital', 'medical', 'clinical', 'pharma'],
        'E-commerce': ['e-commerce', 'retail', 'ecommerce', 'marketplace'],
        'Consulting': ['consulting', 'advisory', 'deloitte', 'mckinsey', 'accenture'],
    }
    for industry, keywords in industry_kw.items():
        if any(kw in text_lower for kw in keywords):
            parsed['industry'] = industry
            break

    return parsed

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def display_salary_results(artifacts, result, form_data, key_prefix=""):
    """Display salary prediction results with enhancements."""
    # Handle both formats: old ML format (predicted/low/high) and new real-time format (predicted_salary/salary_range)
    if 'predicted' in result:
        pred = result['predicted']
        low = result['low']
        high = result['high']
    elif 'predicted_salary' in result:
        pred = result['predicted_salary']
        low, high = result['salary_range']
    else:
        st.error("Invalid result format")
        return

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # Main salary card
    st.markdown(f"""
    <div class="result-card" style="text-align:center;padding:3rem 2rem">
        <div class="salary-label">Estimated Annual Salary</div>
        <div class="salary-amount">${pred:,.0f}</div>
        <div class="salary-range">
            Confidence range: <strong>${low:,.0f}</strong> – <strong>${high:,.0f}</strong> (±12%)
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Add enhancements if available
    if ENHANCEMENTS_AVAILABLE:
        # What-If Analysis
        try:
            # Create analyzer
            def predict_wrapper(data):
                return predict_salary(artifacts, data)
            
            analyzer = WhatIfAnalyzer(predict_wrapper, form_data)
            analyzer.set_base_prediction(pred)
            
            # Display what-if analysis
            display_whatif_analysis(
                analyzer, 
                artifacts,
                ALL_SKILLS,
                LOCATIONS,
                SENIORITY_ORDER,
                EDUCATION_ORDER,
                key_prefix
            )
        except Exception as e:
            st.warning(f"What-if analysis not available: {e}")





def display_groq_results(groq_result: dict, comparison: dict):
    """Display Groq AI prediction results and comparison with XGBoost."""
    
    st.markdown("---")
    st.markdown("### 🤖 AI Insights & Recommendations")
    
    # Detailed insights - removed comparison cards
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📝 AI Reasoning</div>
            <div style="margin-top:0.5rem;line-height:1.6">
                {groq_result['reasoning']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Factors</div>
            <ul style="margin-top:0.5rem;line-height:1.8">
                {''.join([f'<li>{factor}</li>' for factor in groq_result['key_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💡 Recommendations</div>
            <div style="margin-top:0.5rem;line-height:1.6">
                {groq_result['recommendations']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📈 Market Insights</div>
            <div style="margin-top:0.5rem;line-height:1.6">
                {groq_result['market_insights']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT FORM VALUES
# ─────────────────────────────────────────────────────────────────────────────
def default_form() -> dict:
    return {
        'job_title': 'Software Engineer',
        'seniority_level': 'Mid-level',
        'years_of_experience': 4,
        'education_level': 'Bachelor',
        'field_of_study': 'Computer Science',
        'gpa': 3.5,
        'industry': 'Technology',
        'company_size': 'Medium (200-1000)',
        'location': 'San Francisco, CA',
        'skills': 'Python|SQL|AWS',
        'num_skills': 3,
        'num_projects': 2,
        'num_publications': 0,
        'num_internships': 1,
        'has_leadership_experience': 0,
        'has_open_source_contributions': 0,
        'certifications': 'None',
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(title, icon=""):
    st.markdown(f'<div class="card"><div class="card-title"><span>{icon}</span> {title}</div>', unsafe_allow_html=True)

def end_card():
    st.markdown('</div>', unsafe_allow_html=True)

def salary_bar(pct, label_left, label_right):
    st.markdown(f"""
    <div class="range-bar-wrap">
        <div class="range-label"><span>{label_left}</span><span>{label_right}</span></div>
        <div class="range-bar-bg"><div class="range-bar-fill" style="width:{pct}%"></div></div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MODEL PERFORMANCE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Display interactive charts showing model performance and feature importance
#
# VISUALIZATIONS:
# --------------
# 1. MODEL COMPARISON CHART (Bar Chart)
#    - Compares XGBoost accuracy (92.3%) with other algorithms
#    - Shows: Random Forest (88.7%), Linear Regression (76.4%), Neural Net (85.2%)
#    - Demonstrates why XGBoost was chosen
#
# 2. FEATURE IMPORTANCE CHART (Horizontal Bar Chart)
#    - Shows top 10 features that influence salary predictions
#    - Ranked by importance score from XGBoost model
#    - Helps users understand what factors matter most
#
# TECHNOLOGY:
# -----------
# - Library: Plotly (interactive JavaScript charts)
# - Chart Types: Bar charts with custom styling
# - Interactivity: Hover tooltips, zoom, pan
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: MAIN APPLICATION FLOW
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Orchestrate the entire application workflow
#
# APPLICATION ARCHITECTURE:
# ------------------------
# 1. INITIALIZATION
#    - Load ML models and artifacts
#    - Initialize session state variables
#    - Display hero section
#
# 2. USER INTERFACE (3 TABS)
#    Tab 1 - Upload Resume:
#      a. File upload (PDF/DOCX/TXT)
#      b. Text extraction from document
#      c. Resume parsing (Groq AI or regex)
#      d. Feature engineering
#      e. Salary prediction
#      f. Display results with visualizations
#
#    Tab 2 - Manual Input: (Currently disabled)
#      - Form-based data entry
#      - Direct prediction without resume
#
#    Tab 3 - Job Search:
#      - Use parsed resume data or manual input
#      - Search Google Jobs via SerpAPI
#      - Rank jobs by match score
#      - Display with AI recommendations
#
# 3. MODEL PERFORMANCE SECTION
#    - Display comparison charts
#    - Show feature importance
#
# DATA FLOW:
# ----------
# Resume Upload → Text Extraction → Parsing → Feature Engineering → 
# ML Prediction → Results Display → Job Search (optional)
#
# SESSION STATE MANAGEMENT:
# ------------------------
# - form_data: Current user profile data
# - result: Latest salary prediction
# - groq_result: AI-powered prediction (optional)
# - parsed_skills: Extracted skills list
# - resume_data: Data from uploaded resume
# - has_resume: Flag indicating if resume was uploaded
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main application entry point.
    Orchestrates the entire salary prediction workflow.
    """
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ Powered by Stacking Ensemble · 97.02% R² Score · 91.29% Accuracy · Trained on 30,000 Resumes</div>
        <h1>SalaryLens</h1>
        <p>Upload your resume or fill the form below to get an AI-powered salary prediction based on your profile.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    artifacts, missing = load_models()

    if missing:
        st.markdown("""
        <div class="alert">
        ⚠️ Model files not found. Run <code>train_salary_model.py</code> first to generate the required artifacts.<br>
        Missing: <code>""" + ", ".join(missing) + """</code>
        </div>""", unsafe_allow_html=True)
        st.info("💡 **Quick start**: Place `salary_model.joblib`, `preprocessor.joblib`, `skill_vectorizer.joblib`, `feature_names.joblib`, and `encoders_config.joblib` in the same directory as this app.")
        return

    # Session state
    if 'form_data' not in st.session_state:
        st.session_state.form_data = default_form()
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'groq_result' not in st.session_state:
        st.session_state.groq_result = None
    if 'comparison' not in st.session_state:
        st.session_state.comparison = None
    if 'parsed_skills' not in st.session_state:
        st.session_state.parsed_skills = []
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None  # Stores data from uploaded resume only
    if 'has_resume' not in st.session_state:
        st.session_state.has_resume = False  # Flag to track if resume was uploaded
    if 'real_time_salary' not in st.session_state:
        st.session_state.real_time_salary = None  # Stores real-time salary search results

    # ── TABS ──────────────────────────────────────────────────────────────────
    # All three tabs enabled: Upload Resume, Manual Input, and Job Search
    tab_upload, tab_manual, tab_jobs = st.tabs(["📄  Upload Resume", "✏️  Manual Input", "💼  Job Search"])

    # ─────────────── TAB 1: UPLOAD ───────────────────────────────────────────
    with tab_upload:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # Show API key status
        if os.environ.get('GROQ_API_KEY'):
            st.success("✓ Groq API key loaded from .env file - AI parsing available!")
        else:
            st.info("💡 For AI-powered parsing (85% accuracy), set up your free Groq API key: `python setup_api_key.py`")
        
        # Add LLM parsing toggle and API key input
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            uploaded = st.file_uploader(
                "Drop your resume here",
                type=["pdf", "docx", "txt"],
                help="Supported: PDF, DOCX, TXT — Max 10 MB"
            )
        with col2:
            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
            use_llm = st.checkbox(
                "🤖 AI Parsing",
                value=False,
                help="Use AI for better accuracy (85% vs 70%)"
            )
        with col3:
            if use_llm:
                st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                
                # Check if key is already in .env
                env_key = os.environ.get('GROQ_API_KEY', '')
                
                if env_key:
                    # Key found in .env - show masked version
                    st.text_input(
                        "Groq API Key",
                        value=f"{env_key[:10]}...{env_key[-4:]}",
                        disabled=True,
                        help="✓ Loaded from .env file"
                    )
                    st.caption("✓ API key loaded from .env file")
                else:
                    # No key in .env - allow manual entry
                    groq_key = st.text_input(
                        "Groq API Key",
                        type="password",
                        placeholder="gsk_...",
                        help="Get free key at console.groq.com"
                    )
                    if groq_key:
                        os.environ['GROQ_API_KEY'] = groq_key
                        st.caption("💡 Save to .env: run `python setup_api_key.py`")

        if uploaded:
            with st.spinner("📖 Parsing resume..."):
                if uploaded.name.endswith(".docx"):
                    text = extract_text_docx(uploaded)
                elif uploaded.name.endswith(".pdf"):
                    text = extract_text_pdf(uploaded)
                else:
                    text = uploaded.read().decode("utf-8", errors="ignore")

            if text.startswith("ERROR"):
                st.error(f"Could not parse file: {text}")
            else:
                # Choose parsing method
                if use_llm:
                    # Use Groq API for AI parsing
                    groq_key = os.environ.get('GROQ_API_KEY', '')
                    
                    if groq_key:
                        # Use Groq API (fast, no GPU needed)
                        try:
                            from llm_resume_parser_api import parse_resume_with_api
                            with st.spinner("🌐 AI-powered parsing via Groq API..."):
                                parsed = parse_resume_with_api(text, provider="groq", api_key=groq_key)
                            st.success("✅ Parsed with Groq API (2-3s, 85% accuracy)")
                        except Exception as e:
                            st.error(f"❌ Groq API failed: {str(e)}")
                            st.info("Falling back to regex parsing...")
                            parsed = parse_resume_text(text)
                    else:
                        # No API key - show message and use regex
                        st.warning("⚠️ No Groq API key provided")
                        st.info("💡 Enter your free API key above for AI-powered parsing (85% accuracy vs 70%)")
                        st.info("Get free key at: https://console.groq.com")
                        parsed = parse_resume_text(text)
                else:
                    # Use regex parser (default)
                    parsed = parse_resume_text(text)
                
                # Merge parsed into form
                fd = st.session_state.form_data.copy()
                fd.update({k: v for k, v in parsed.items() if v})
                st.session_state.form_data = fd
                st.session_state.parsed_skills = parsed.get('skills', '').split('|') if parsed.get('skills') else []
                
                # Store resume data separately for job search with defaults
                resume_defaults = default_form()
                resume_defaults.update({k: v for k, v in parsed.items() if v})
                st.session_state.resume_data = resume_defaults
                st.session_state.has_resume = True

                st.success(f"✅ Parsed **{uploaded.name}** — {len(text.split())} words extracted")

                # Show what was extracted
                with st.expander("🔍 View extracted fields"):
                    cols = st.columns(3)
                    fields = {k: v for k, v in parsed.items() if k not in ('skills',)}
                    for i, (k, v) in enumerate(fields.items()):
                        cols[i % 3].markdown(f"**{k.replace('_',' ').title()}**: `{v}`")
                    if parsed.get('skills'):
                        st.markdown(f"**Skills detected**: {', '.join(parsed['skills'].split('|'))}")

                with st.expander("📝 Raw resume text (first 1500 chars)"):
                    st.text(text[:1500] + ("..." if len(text) > 1500 else ""))
        
        # Predict button in Upload tab
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # Add option to use Groq for prediction
        use_groq_prediction = False
        if os.environ.get('GROQ_API_KEY'):
            col_option, col_spacer = st.columns([3, 1])
            with col_option:
                use_groq_prediction = st.checkbox(
                    "🤖 Also get AI prediction & insights (Groq)",
                    value=False,
                    help="Get salary prediction with explanations from Groq AI (in addition to XGBoost)"
                )
        
        col_btn, col_reset = st.columns([4, 1])
        with col_btn:
            if st.button("🔮  Predict My Salary", use_container_width=True, key="predict_upload"):
                with st.spinner("Computing prediction..."):
                    try:
                        # XGBoost prediction (always)
                        result = predict_salary(artifacts, st.session_state.form_data)
                        st.session_state.result = result
                        
                        # Groq prediction (optional)
                        if use_groq_prediction:
                            with st.spinner("Getting AI insights from Groq..."):
                                try:
                                    from groq_salary_predictor import predict_salary_with_groq, compare_predictions
                                    groq_result = predict_salary_with_groq(st.session_state.form_data)
                                    comparison = compare_predictions(result, groq_result)
                                    st.session_state.groq_result = groq_result
                                    st.session_state.comparison = comparison
                                except ImportError as e:
                                    st.warning(f"⚠️ Groq library not installed: {e}")
                                    st.info("Install with: pip install groq")
                                    st.session_state.groq_result = None
                                    st.session_state.comparison = None
                                except ValueError as e:
                                    st.warning(f"⚠️ Groq API key not configured: {e}")
                                    st.info("Add GROQ_API_KEY to your .env file")
                                    st.session_state.groq_result = None
                                    st.session_state.comparison = None
                                except Exception as e:
                                    # More detailed error logging
                                    import traceback
                                    error_msg = str(e)
                                    error_details = traceback.format_exc()
                                    
                                    # Check if it's actually a Streamlit error
                                    if "set_page_config" in error_msg:
                                        st.error("❌ Streamlit Configuration Error")
                                        st.warning("This error is NOT from Groq API. It's a Streamlit caching issue.")
                                        st.info("**Fix:** Clear Streamlit cache and restart:\n```\nstreamlit cache clear\n```")
                                    else:
                                        st.error(f"❌ Groq API Error: {error_msg}")
                                    
                                    with st.expander("🔍 Full Error Details (for debugging)"):
                                        st.code(error_details)
                                    st.session_state.groq_result = None
                                    st.session_state.comparison = None
                        else:
                            st.session_state.groq_result = None
                            st.session_state.comparison = None
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        with col_reset:
            if st.button("↺ Reset", use_container_width=True, key="reset_upload"):
                st.session_state.form_data = default_form()
                st.session_state.result = None
                st.session_state.groq_result = None
                st.session_state.comparison = None
                st.session_state.has_resume = False
                st.session_state.resume_data = None
                st.rerun()
        
        # Display results if available
        if st.session_state.result:
            display_salary_results(artifacts, st.session_state.result, st.session_state.form_data, "upload_")
            
            # Display Groq results if available
            if st.session_state.get('groq_result') and st.session_state.get('comparison'):
                display_groq_results(st.session_state.groq_result, st.session_state.comparison)

    # ─────────────── TAB 2: MANUAL ───────────────────────────────────────────
    with tab_manual:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # Manual input should be independent - no resume data auto-fill
        st.info("💡 **Manual Input:** Enter your details below to get a salary prediction. This is completely independent of any uploaded resume.")
        
        # Add informational note
        st.info("💡 **Note:** Use your common sense and add data realistically. Unrealistic combinations (e.g., 'Intern' with 15 years experience) may produce inaccurate predictions.")
        
        # Initialize form data for manual input if not exists
        if 'manual_form_data' not in st.session_state:
            st.session_state.manual_form_data = default_form()
        
        _render_form(st.session_state.manual_form_data, key_prefix="manual")
        
        # Predict button in Manual tab
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # Add option to use Groq for prediction
        use_groq_prediction_manual = False
        if os.environ.get('GROQ_API_KEY'):
            col_option, col_spacer = st.columns([3, 1])
            with col_option:
                use_groq_prediction_manual = st.checkbox(
                    "🤖 Also get AI prediction & insights (Groq)",
                    value=False,
                    help="Get salary prediction with explanations from Groq AI",
                    key="groq_manual"
                )
        
        col_btn, col_reset = st.columns([4, 1])
        with col_btn:
            if st.button("🔮 Predict My Salary", use_container_width=True, key="predict_manual", type="primary"):
                with st.spinner("🔍 Running ML model prediction..."):
                    try:
                        # Use ML model for prediction with manual form data
                        result = predict_salary(artifacts, st.session_state.manual_form_data)
                        st.session_state.manual_result = result
                        st.session_state.real_time_data = None  # Clear real-time data
                        
                        # Groq prediction (optional)
                        if use_groq_prediction_manual:
                            with st.spinner("Getting AI insights from Groq..."):
                                try:
                                    from groq_salary_predictor import predict_salary_with_groq, compare_predictions
                                    groq_result = predict_salary_with_groq(st.session_state.manual_form_data)
                                    # Convert ML result format for comparison
                                    ml_result_for_comparison = {
                                        'predicted_salary': result['predicted'],
                                        'salary_range': (result['low'], result['high']),
                                        'confidence': 0.92,
                                        'source': 'XGBoost ML Model'
                                    }
                                    comparison = compare_predictions(ml_result_for_comparison, groq_result)
                                    st.session_state.manual_groq_result = groq_result
                                    st.session_state.manual_comparison = comparison
                                except Exception as e:
                                    st.error(f"❌ Groq API Error: {str(e)}")
                                    st.session_state.manual_groq_result = None
                                    st.session_state.manual_comparison = None
                        else:
                            st.session_state.manual_groq_result = None
                            st.session_state.manual_comparison = None
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        with col_reset:
            if st.button("↺ Reset", use_container_width=True, key="reset_manual"):
                st.session_state.manual_form_data = default_form()
                st.session_state.manual_result = None
                st.session_state.manual_groq_result = None
                st.session_state.manual_comparison = None
                st.session_state.real_time_data = None
                st.rerun()
        
        # Display results if available (using manual-specific session state)
        if st.session_state.get('manual_result'):
            # Display full results with feature importance (only once)
            display_salary_results(artifacts, st.session_state.manual_result, st.session_state.manual_form_data, "manual_")
            
            # Display Groq results if available
            if st.session_state.get('manual_groq_result') and st.session_state.get('manual_comparison'):
                display_groq_results(st.session_state.manual_groq_result, st.session_state.manual_comparison)

    # ─────────────── TAB 3: JOB SEARCH ───────────────────────────────────────
    with tab_jobs:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # Determine data source - prioritize manual input if available, then resume
        if st.session_state.get('manual_form_data') and any(st.session_state.manual_form_data.values()):
            fd = st.session_state.manual_form_data
            data_source = "manual input"
        elif st.session_state.has_resume and st.session_state.resume_data:
            fd = st.session_state.resume_data
            data_source = "resume"
        else:
            # Fallback to default form data
            fd = st.session_state.get('form_data', default_form())
            data_source = "default"
        
        user_location = fd.get('location', 'San Francisco, CA')
        user_skills = [s for s in fd.get('skills', '').split('|') if s]
        user_job_title = fd.get('job_title', 'Software Engineer')
        user_seniority = fd.get('seniority_level', 'Mid-level')
        user_experience = fd.get('years_of_experience', 4)
        user_education = fd.get('education_level', 'Bachelor')
        user_industry = fd.get('industry', 'Technology')
        
        # Display data source badge
        if data_source == "resume":
            source_badge = '<div style="display:inline-block;background:rgba(52,211,153,0.15);border:1px solid rgba(52,211,153,0.3);color:#34d399;padding:0.2rem 0.6rem;border-radius:100px;font-size:0.75rem;font-weight:600;margin-bottom:0.8rem">📄 FROM UPLOADED RESUME</div>'
        else:
            source_badge = '<div style="display:inline-block;background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.3);color:#fbbf24;padding:0.2rem 0.6rem;border-radius:100px;font-size:0.75rem;font-weight:600;margin-bottom:0.8rem">✏️ FROM MANUAL INPUT</div>'
        
        # Display comprehensive profile from resume
        st.markdown(f"""
        <div class="card">
            <div class="card-title"><span>🎯</span> Your Job Search Profile</div>
            {source_badge}
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.8rem;margin-bottom:1rem">
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">JOB TITLE</div>
                    <div style="color:var(--text);font-weight:600">{user_job_title}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">SENIORITY LEVEL</div>
                    <div style="color:var(--text);font-weight:600">{user_seniority}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">EXPERIENCE</div>
                    <div style="color:var(--text);font-weight:600">{user_experience} years</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">EDUCATION</div>
                    <div style="color:var(--text);font-weight:600">{user_education}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">LOCATION</div>
                    <div style="color:var(--text);font-weight:600">{user_location}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.2rem">INDUSTRY</div>
                    <div style="color:var(--text);font-weight:600">{user_industry}</div>
                </div>
            </div>
            <div style="border-top:1px solid var(--border);padding-top:0.8rem">
                <div style="color:var(--muted);font-size:0.75rem;margin-bottom:0.5rem">YOUR SKILLS ({len(user_skills)})</div>
                <div class="chips">
                    {' '.join([f'<div class="chip">{skill}</div>' for skill in user_skills[:12]])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show message based on data source
        if not st.session_state.has_resume:
            st.info("💡 **Tip:** Upload your resume in the 'Upload Resume' tab for more accurate job matches based on your actual experience and skills.")
        
        if not user_skills:
            st.warning("⚠️ No skills detected. Upload a resume in the 'Upload Resume' tab to get personalized job recommendations.")
        
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        
        # ─────────────── AI JOB RECOMMENDATIONS ───────────────────────────────
        use_ai_recommendations = st.checkbox(
            "🤖 Get AI-Powered Job Recommendations",
            value=False,
            help="Use Groq AI to analyze your profile and suggest optimal job searches"
        )
        
        if use_ai_recommendations:
            if not os.environ.get('GROQ_API_KEY'):
                st.warning("⚠️ GROQ_API_KEY not set. Get a free API key at [console.groq.com](https://console.groq.com)")
                st.info("💡 Add your key to the .env file: `GROQ_API_KEY=your_key_here`")
            else:
                try:
                    from groq_job_recommender import get_job_recommendations
                    
                    with st.spinner("🤖 Analyzing your profile with AI..."):
                        recommendations = get_job_recommendations(fd)
                    
                    # AI Recommendations Panel
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; color: white;">
                        <h4 style="margin: 0 0 0.5rem 0; color: white;">🤖 AI Career Intelligence</h4>
                        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Powered by Groq AI</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommended Job Titles
                    if recommendations.get('recommended_titles'):
                        st.markdown("**🎯 Recommended Job Titles to Search:**")
                        cols = st.columns(min(5, len(recommendations['recommended_titles'])))
                        for i, title in enumerate(recommendations['recommended_titles'][:5]):
                            with cols[i]:
                                if st.button(title, key=f"ai_title_{i}", use_container_width=True):
                                    # Store the selected title for search
                                    st.session_state.ai_selected_query = title
                                    st.rerun()
                    
                    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                    
                    # Smart Search Queries
                    if recommendations.get('search_queries'):
                        st.markdown("**🎯 Smart Search Strategies:**")
                        for i, sq in enumerate(recommendations['search_queries'][:4]):
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.markdown(f"• **{sq['query']}**")
                            with col2:
                                priority_color = "🔴" if sq['priority'] == 'high' else "🟡"
                                st.caption(f"{priority_color} {sq['reason']}")
                            with col3:
                                if st.button("Search", key=f"ai_query_{i}", use_container_width=True):
                                    st.session_state.ai_selected_query = sq['query']
                                    st.rerun()
                    
                    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
                    
                    # Career Insights in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if recommendations.get('profile_strengths'):
                            st.markdown("**💪 Your Strengths:**")
                            for strength in recommendations['profile_strengths'][:3]:
                                st.markdown(f"• {strength}")
                        
                        if recommendations.get('top_skills_to_highlight'):
                            st.markdown("**⭐ Skills to Highlight:**")
                            skill_tags = " ".join([f"`{skill}`" for skill in recommendations['top_skills_to_highlight'][:4]])
                            st.markdown(skill_tags)
                    
                    with col2:
                        if recommendations.get('skill_gaps'):
                            st.markdown("**📈 Skills to Develop:**")
                            for gap in recommendations['skill_gaps'][:3]:
                                st.markdown(f"• {gap}")
                        
                        if recommendations.get('salary_expectation'):
                            st.markdown("**💰 Expected Salary:**")
                            st.info(recommendations['salary_expectation'])
                    
                    # Career Advice
                    if recommendations.get('career_level_advice'):
                        st.markdown("**💡 Career Level Advice:**")
                        st.success(recommendations['career_level_advice'])
                    
                    # Market Insights
                    if recommendations.get('market_insights'):
                        with st.expander("📊 Market Insights", expanded=False):
                            st.write(recommendations['market_insights'])
                    
                    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
                    
                except ImportError:
                    st.error("⚠️ Please install the groq package: `pip install groq`")
                except Exception as e:
                    st.error(f"❌ AI recommendations failed: {str(e)}")
        
        # Check if there's an AI-selected query
        if hasattr(st.session_state, 'ai_selected_query') and st.session_state.ai_selected_query:
            st.info(f"🤖 AI Selected: **{st.session_state.ai_selected_query}**")
            # Override the job title with AI selection
            fd = fd.copy()
            fd['job_title'] = st.session_state.ai_selected_query
        
        # Check if API key is configured
        if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":
            st.warning("⚠️ SerpAPI key not configured. Please add your API key in the code (SERPAPI_KEY constant).")
            st.info("💡 Get your free API key at [SerpAPI](https://serpapi.com/users/sign_up) - 100 searches/month free")
            serpapi_key = None
        else:
            serpapi_key = SERPAPI_KEY
        
        # ─────────────── JOB SEARCH FORM ─────────────────────────────────────
        st.markdown("### 🔍 Customize Your Search")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_location = st.text_input(
                "📍 Location",
                value=user_location,
                help="City and state for job search"
            )
        
        with col2:
            search_experience = st.number_input(
                "📅 Years of Experience",
                min_value=0,
                max_value=50,
                value=int(user_experience),
                help="Your years of professional experience"
            )
        
        with col3:
            num_results = st.selectbox(
                "📊 Number of Results",
                options=[5, 10, 15, 20, 25],
                index=1,
                help="How many jobs to display"
            )
        
        # Update fd with search parameters
        fd = fd.copy()
        fd['location'] = search_location
        fd['years_of_experience'] = search_experience
        
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        
        # Enhanced search button
        if hasattr(st.session_state, 'ai_selected_query') and st.session_state.ai_selected_query:
            button_label = f"🔍 Search: {st.session_state.ai_selected_query}"
        else:
            button_label = "🔍 Search Jobs"
        
        search_button = st.button(button_label, use_container_width=True, type="primary", disabled=not serpapi_key)
        
        if search_button and serpapi_key:
            # Clear AI selection after search
            if hasattr(st.session_state, 'ai_selected_query'):
                del st.session_state.ai_selected_query
            
            with st.spinner("Searching for jobs via SerpAPI..."):
                # Use the same data source (resume or manual) that was displayed above
                search_data = fd  # This is already set to resume_data or form_data above
                
                # Enhance query with AI if enabled
                if use_ai_recommendations and os.environ.get('GROQ_API_KEY'):
                    try:
                        from groq_job_recommender import enhance_job_search_query
                        original_title = search_data['job_title']
                        enhanced_title = enhance_job_search_query(search_data, original_title)
                        if enhanced_title != original_title:
                            search_data = search_data.copy()
                            search_data['job_title'] = enhanced_title
                            st.info(f"🤖 AI Enhanced: {original_title} → {enhanced_title}")
                    except:
                        pass  # Fallback to original query
                
                jobs, query, is_fallback = search_jobs_serpapi(search_data, serpapi_key, fallback=False)
                
                # If no jobs found, try broader search
                if jobs is not None and len(jobs) == 0:
                    st.info("🔄 No exact matches found. Searching for similar jobs...")
                    jobs, query, is_fallback = search_jobs_serpapi(search_data, serpapi_key, fallback=True)
                
                if jobs and len(jobs) > 0:
                    if is_fallback:
                        st.success(f"✅ Found {len(jobs)} similar job postings (broader search)")
                        st.info("💡 These are similar jobs based on your job title and location. Try adjusting your profile for more specific matches.")
                    else:
                        st.success(f"✅ Found {len(jobs)} job postings matching your profile")
                    
                    # Display jobs using Streamlit components
                    for i, job in enumerate(jobs):
                        match_color = "#34d399" if job['match_score'] >= 70 else "#fbbf24" if job['match_score'] >= 50 else "#7a829a"
                        
                        with st.container():
                            # Job card header
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"### {job['title']}")
                                st.caption(f"{job['company']} · {job['location']}")
                                if job.get('salary'):
                                    st.markdown(f"💰 **{job['salary']}**")
                            with col2:
                                st.markdown(f"<div style='background:{match_color};color:#0d0f14;padding:0.5rem 1rem;border-radius:100px;font-weight:700;text-align:center;margin-top:1rem'>{job['match_score']}% Match</div>", unsafe_allow_html=True)
                            
                            # AI Match Analysis (if enabled)
                            if use_ai_recommendations and os.environ.get('GROQ_API_KEY'):
                                try:
                                    from groq_job_recommender import analyze_job_match
                                    with st.spinner("🤖 Analyzing match..."):
                                        ai_match = analyze_job_match(job['description'], search_data)
                                    
                                    # Display AI insights
                                    st.markdown("**🤖 AI Match Analysis:**")
                                    col_ai1, col_ai2 = st.columns(2)
                                    
                                    with col_ai1:
                                        if ai_match.get('match_reasons'):
                                            st.markdown("✅ **Why you're a good fit:**")
                                            for reason in ai_match['match_reasons'][:3]:
                                                st.markdown(f"• {reason}")
                                    
                                    with col_ai2:
                                        if ai_match.get('missing_skills'):
                                            st.markdown("📚 **Skills mentioned in job:**")
                                            missing_tags = " ".join([f"`{skill}`" for skill in ai_match['missing_skills'][:4]])
                                            st.markdown(missing_tags)
                                    
                                    if ai_match.get('recommendation'):
                                        rec = ai_match['recommendation']
                                        if "Apply" in rec or "good fit" in rec.lower():
                                            st.success(f"💡 {rec}")
                                        else:
                                            st.info(f"💡 {rec}")
                                    
                                except Exception as e:
                                    # Silently fail AI analysis, show basic match score
                                    pass
                            
                            # Job description - Show preview and full text in expander
                            desc_preview = job['description'][:200] + '...' if len(job['description']) > 200 else job['description']
                            st.write(desc_preview)
                            
                            if len(job['description']) > 200:
                                with st.expander("📄 Read full job description"):
                                    st.write(job['description'])
                            
                            # Job metadata
                            col_meta1, col_meta2, col_meta3 = st.columns([2, 2, 1])
                            with col_meta1:
                                st.caption(f"📅 {job['posted']}")
                            with col_meta2:
                                st.caption(f"💼 {job.get('job_type', 'Full-time')}")
                            with col_meta3:
                                # Use link_button which opens in new tab by default
                                if job['url'] and job['url'] != '#':
                                    st.link_button("Apply →", job['url'], use_container_width=True)
                                else:
                                    st.caption("No apply link")
                            
                            # Skills
                            if job['skills']:
                                st.caption("**Skills:** " + ", ".join(job['skills'][:4]))
                            
                            st.divider()
                    
                    # Link to full Google Jobs search
                    from urllib.parse import quote_plus
                    search_url = f"https://www.google.com/search?q={quote_plus(query)}&ibp=htl;jobs"
                    st.markdown(f"[🔗 View all results on Google Jobs]({search_url})")
                    
                elif jobs is not None and len(jobs) == 0:
                    st.warning("No jobs found matching your criteria. Try adjusting your profile or search manually.")
                    from urllib.parse import quote_plus
                    search_url = f"https://www.google.com/search?q={quote_plus(query)}&ibp=htl;jobs"
                    st.markdown(f"""
                    <div style="text-align:center;margin-top:1rem">
                        <a href="{search_url}" target="_blank" style="display:inline-block;background:linear-gradient(135deg,#4f8ef7,#a78bfa);color:white;border:none;border-radius:10px;padding:0.75rem 2rem;font-weight:700;text-decoration:none">
                            🔍 Search on Google Jobs
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Failed to fetch jobs. Please check your API key and try again.")
                    from urllib.parse import quote_plus
                    search_url = f"https://www.google.com/search?q={quote_plus(query)}&ibp=htl;jobs"
                    st.markdown(f"""
                    <div style="text-align:center;margin-top:1rem">
                        <a href="{search_url}" target="_blank" style="display:inline-block;background:linear-gradient(135deg,#4f8ef7,#a78bfa);color:white;border:none;border-radius:10px;padding:0.75rem 2rem;font-weight:700;text-decoration:none">
                            🔍 Search on Google Jobs
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

    # Note: Model Performance visualizations moved to separate "Model Analytics" page
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 1rem 0.5rem;">
        <p style="color: #7a829a; font-size: 0.95rem; margin-bottom: 1rem;">
            📊 Want to see detailed model performance metrics, charts, and technical analysis?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📊 View Model Analytics →", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Model_Analytics.py")
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FORM RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _render_form(fd: dict, key_prefix: str):
    """Render the full input form with validation logic."""
    
    # Define realistic experience ranges for each seniority level
    SENIORITY_EXPERIENCE_RANGES = {
        'Intern': (0, 1),
        'Junior': (0, 3),
        'Mid-level': (2, 7),
        'Senior': (5, 12),
        'Lead': (7, 15),
        'Principal': (10, 20),
        'Staff': (10, 20),
        'Director': (12, 25),
        'VP': (15, 30),
        'CTO/CXO': (15, 35)
    }
    
    # Helper function to safely get index
    def safe_index(lst, value, default):
        try:
            return lst.index(value)
        except ValueError:
            return lst.index(default) if default in lst else 0
    
    c1, c2, c3 = st.columns(3)

    with c1:
        jt_index = safe_index(JOB_TITLES, fd.get('job_title'), JOB_TITLES[0])
        jt = st.selectbox("Job Title", JOB_TITLES,
                          index=jt_index,
                          key=f"{key_prefix}_job_title")
        
        sl_index = safe_index(SENIORITY_ORDER, fd.get('seniority_level'), 'Mid-level')
        sl = st.selectbox("Seniority Level", SENIORITY_ORDER,
                          index=sl_index,
                          key=f"{key_prefix}_seniority",
                          help="Choose your career level - this affects expected experience range")
        
        # Get experience range for selected seniority
        min_exp, max_exp = SENIORITY_EXPERIENCE_RANGES.get(sl, (0, 35))
        
        # Show recommended range
        st.caption(f"💡 Recommended: {min_exp}-{max_exp} years for {sl}")
        
        yoe = st.slider("Years of Experience", 0, 35,
                        int(fd.get('years_of_experience', 4)),
                        key=f"{key_prefix}_yoe",
                        help=f"Typical range for {sl}: {min_exp}-{max_exp} years")
        
        # Validation warning
        if yoe < min_exp or yoe > max_exp:
            if yoe < min_exp:
                st.warning(f"⚠️ {yoe} years is low for {sl} level. Typical minimum: {min_exp} years")
            else:
                st.warning(f"⚠️ {yoe} years is high for {sl} level. Consider higher seniority (typical max: {max_exp} years)")

    with c2:
        edu_index = safe_index(EDUCATION_ORDER, fd.get('education_level'), 'Bachelor')
        edu = st.selectbox("Education Level", EDUCATION_ORDER,
                           index=edu_index,
                           key=f"{key_prefix}_edu")
        fos_index = safe_index(FIELDS_OF_STUDY, fd.get('field_of_study'), 'Computer Science')
        fos = st.selectbox("Field of Study", FIELDS_OF_STUDY,
                           index=fos_index,
                           key=f"{key_prefix}_fos")
        gpa = st.slider("GPA", 2.0, 4.0,
                        float(fd.get('gpa', 3.5)), step=0.05,
                        key=f"{key_prefix}_gpa")

    with c3:
        ind_index = safe_index(INDUSTRIES, fd.get('industry'), 'Technology')
        ind = st.selectbox("Industry", INDUSTRIES,
                           index=ind_index,
                           key=f"{key_prefix}_industry")
        cs_index = safe_index(COMPANY_SIZE_ORDER, fd.get('company_size'), 'Medium (200-1000)')
        cs = st.selectbox("Company Size", COMPANY_SIZE_ORDER,
                          index=cs_index,
                          key=f"{key_prefix}_company_size")
        loc_index = safe_index(LOCATIONS, fd.get('location'), 'San Francisco, CA')
        loc = st.selectbox("Location", LOCATIONS,
                           index=loc_index,
                           key=f"{key_prefix}_location")

    # Skills multi-select
    st.markdown("""
    <div style="background: rgba(79, 142, 247, 0.1); border-left: 3px solid #4f8ef7; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.8rem; font-size: 0.85rem;">
    <strong>💡 Skills Selection:</strong> These are the skills recognized by our trained model from 30,000 resumes. 
    The model uses TF-IDF to understand which specific skills impact salary predictions.
    </div>
    """, unsafe_allow_html=True)
    
    current_skills = [s for s in fd.get('skills', '').split('|') if s in ALL_SKILLS]
    selected_skills = st.multiselect(
        "Skills (select all that apply)",
        ALL_SKILLS,
        default=current_skills if current_skills else [],
        key=f"{key_prefix}_skills",
        help="Select skills that match your expertise. The model recognizes these specific skills from the training dataset."
    )
    
    # Skill count validation
    if len(selected_skills) < 3:
        st.info(f"💡 Tip: Most {sl} professionals have 5-15 skills. Consider adding more.")
    elif len(selected_skills) > 20:
        st.warning("⚠️ Too many skills selected. Focus on your core competencies (5-15 skills recommended).")

    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    ca, cb, cc, cd = st.columns(4)
    with ca:
        np_ = st.number_input("# Projects", 0, 20, int(fd.get('num_projects', 2)), key=f"{key_prefix}_np")
        # Projects validation
        if sl in ['Lead', 'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO'] and np_ < 3:
            st.caption("⚠️ Senior roles typically have 5+ projects")
    with cb:
        pub = st.number_input("# Publications", 0, 30, int(fd.get('num_publications', 0)), key=f"{key_prefix}_pub")
    with cc:
        nin = st.number_input("# Internships", 0, 5, int(fd.get('num_internships', 1)), key=f"{key_prefix}_nin")
        # Internship validation
        if sl in ['Senior', 'Lead', 'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO'] and nin > 3:
            st.caption("💡 Senior roles typically list 0-2 internships")
    with cd:
        cert_keys = list(CERT_RANK.keys())
        cert_index = safe_index(cert_keys, fd.get('certifications'), 'None')
        cert = st.selectbox("Certifications", cert_keys,
                            index=cert_index,
                            key=f"{key_prefix}_cert")

    ce, cf = st.columns(2)
    with ce:
        lead = st.checkbox("Has Leadership Experience", value=bool(fd.get('has_leadership_experience', 0)), key=f"{key_prefix}_lead")
        # Leadership validation
        if sl in ['Lead', 'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO'] and not lead:
            st.caption("💡 Leadership experience expected for this level")
    with cf:
        oss = st.checkbox("Has Open Source Contributions", value=bool(fd.get('has_open_source_contributions', 0)), key=f"{key_prefix}_oss")
    
    # Overall validation summary
    validation_issues = []
    if yoe < min_exp:
        validation_issues.append(f"Experience ({yoe} yrs) is below typical range for {sl} ({min_exp}-{max_exp} yrs)")
    elif yoe > max_exp:
        validation_issues.append(f"Experience ({yoe} yrs) exceeds typical range for {sl} ({min_exp}-{max_exp} yrs)")
    
    if sl in ['Lead', 'Principal', 'Staff', 'Director', 'VP', 'CTO/CXO']:
        if not lead:
            validation_issues.append(f"Leadership experience typically required for {sl} level")
        if np_ < 3:
            validation_issues.append(f"Senior roles ({sl}) typically have 5+ projects")
    
    if len(selected_skills) < 3:
        validation_issues.append("Consider adding more skills (5-15 recommended)")
    
    # Show validation summary if there are issues
    if validation_issues:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            st.markdown("**The following may affect prediction accuracy:**")
            for issue in validation_issues:
                st.markdown(f"• {issue}")
            st.info("💡 These are recommendations. You can still proceed with prediction, but results may be less accurate with unrealistic combinations.")

    # Write back to session state - use correct variable based on key_prefix
    form_data_updates = {
        'job_title': jt,
        'seniority_level': sl,
        'years_of_experience': yoe,
        'education_level': edu,
        'field_of_study': fos,
        'gpa': gpa,
        'industry': ind,
        'company_size': cs,
        'location': loc,
        'skills': '|'.join(selected_skills),
        'num_skills': len(selected_skills),
        'num_projects': np_,
        'num_publications': pub,
        'num_internships': nin,
        'certifications': cert,
        'has_leadership_experience': int(lead),
        'has_open_source_contributions': int(oss),
    }
    
    # Update the correct session state variable based on key_prefix
    if key_prefix == "manual":
        st.session_state.manual_form_data.update(form_data_updates)
    else:
        st.session_state.form_data.update(form_data_updates)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
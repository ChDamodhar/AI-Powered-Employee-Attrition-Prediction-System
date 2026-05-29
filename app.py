import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(
    page_title="End-to-End HR Machine Learning Portal",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# 2. CACHED DATA & MODEL PIPELINE (Runs Once on Startup)
# ---------------------------------------------------------
@st.cache_resource
def initialize_ml_pipeline():
    # Load dataset (cached locally, otherwise downloaded from public IBM repo with fallbacks)
    local_csv = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(local_csv):
        urls = [
            "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
            "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/IBM-HR-Data-Employee-Attrition.csv"
        ]
        df_raw = None
        for url in urls:
            try:
                df_raw = pd.read_csv(url)
                df_raw.to_csv(local_csv, index=False)
                break
            except Exception:
                continue
        if df_raw is None:
            raise FileNotFoundError("Could not download the IBM HR Employee Attrition Dataset from any of the public sources.")
    else:
        df_raw = pd.read_csv(local_csv)
    
    # Process target
    df_processed = df_raw.copy()
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop uniform features
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
    df_processed = df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns], errors='ignore')
    
    # Encoding
    df_encoded = pd.get_dummies(df_processed, drop_first=True)
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Supervised Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        "SVM": SVC(kernel='linear', probability=True, random_state=42).fit(X_train_scaled, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train),
        "Naive Bayes": GaussianNB().fit(X_train, y_train)
    }
    
    # Calculate Accuracies
    accuracies = {
        "Logistic Regression": models["Logistic Regression"].score(X_test_scaled, y_test),
        "Decision Tree": models["Decision Tree"].score(X_test, y_test),
        "Random Forest": models["Random Forest"].score(X_test, y_test),
        "SVM": models["SVM"].score(X_test_scaled, y_test),
        "KNN": models["KNN"].score(X_test_scaled, y_test),
        "Naive Bayes": models["Naive Bayes"].score(X_test, y_test)
    }
    
    # Train Unsupervised Models
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    
    return df_raw, X, X_train, X_train_scaled, models, accuracies, scaler, pca, kmeans

# Initialize everything
df_raw, X_features, X_train, X_train_scaled, models, accuracies, scaler, pca, kmeans = initialize_ml_pipeline()

# ---------------------------------------------------------
# 3. INTERACTIVE INPUT SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("👤 Input Profile Metrics")
st.sidebar.markdown("Modify values below to see changes across all Phase 2 models.")

age = st.sidebar.slider("Age", 18, 60, 35)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 4500)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 2)
total_working_years = st.sidebar.number_input("Total Working Years", 0, 40, 8)
years_at_company = st.sidebar.number_input("Years at Company", 0, 40, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work-Life Balance (1-4)", 1, 4, 3)
overtime = st.sidebar.selectbox("Overtime Worked?", ["No", "Yes"])
job_role = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician", 
    "Manufacturing Director", "Healthcare Representative", "Manager", 
    "Sales Representative", "Research Director", "Human Resources"
])

# Build input row vector matching feature names
input_df = pd.DataFrame(columns=X_features.columns)
input_df.loc[0] = 0

input_df.at[0, 'Age'] = age
input_df.at[0, 'MonthlyIncome'] = monthly_income
input_df.at[0, 'JobSatisfaction'] = job_satisfaction
input_df.at[0, 'TotalWorkingYears'] = total_working_years
input_df.at[0, 'YearsAtCompany'] = years_at_company
input_df.at[0, 'EnvironmentSatisfaction'] = environment_satisfaction
input_df.at[0, 'WorkLifeBalance'] = work_life_balance
if overtime == "Yes":
    input_df.at[0, 'OverTime_Yes'] = 1
role_col = f"JobRole_{job_role}"
if role_col in input_df.columns:
    input_df.at[0, role_col] = 1

input_scaled = scaler.transform(input_df)

# ---------------------------------------------------------
# 4. MAIN APP HEADER & NAVIGATION TABS
# ---------------------------------------------------------
st.title("🏆 AI-Powered Employee Attrition System (Phase 2 Workspace)")
st.markdown("This dashboard maps your full project requirements into an interactive interface.")
st.divider()

# Create tabs matching each required machine learning component
tabs = st.tabs([
    "🔮 Multi-Model Predictions", 
    "🌲 Decision Tree Rules", 
    "📈 Accuracy Comparison", 
    "👥 KNN Similar Employees", 
    "🧩 PCA & K-Means Segmentation"
])

# ---------------------------------------------------------
# TAB 1: LOGISTIC REGRESSION, RANDOM FOREST, NAIVE BAYES
# ---------------------------------------------------------
with tabs[0]:
    st.header("🔮 Supervised Model Prediction Center")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Logistic Regression")
        lr_pred = models["Logistic Regression"].predict(input_scaled)[0]
        lr_prob = models["Logistic Regression"].predict_proba(input_scaled)[0][1] * 100
        st.metric(label="Prediction Status", value="Attrition: YES" if lr_pred == 1 else "Attrition: NO")
        st.write(f"Confidence Score: **{lr_prob:.1f}%**")
        
    with col2:
        st.subheader("Random Forest (Production)")
        rf_pred = models["Random Forest"].predict(input_df)[0]
        rf_prob = models["Random Forest"].predict_proba(input_df)[0][1] * 100
        st.metric(label="Prediction Status", value="Attrition: YES" if rf_pred == 1 else "Attrition: NO")
        st.write(f"Confidence Score: **{rf_prob:.1f}%**")
        
    with col3:
        st.subheader("Naive Bayes (Probability-Based)")
        nb_pred = models["Naive Bayes"].predict(input_df)[0]
        nb_prob = models["Naive Bayes"].predict_proba(input_df)[0][1] * 100
        st.metric(label="Prediction Status", value="Attrition: YES" if nb_pred == 1 else "Attrition: NO")
        st.write(f"Probability-based prediction: **{nb_prob:.1f}% risk**")

# ---------------------------------------------------------
# TAB 2: DECISION TREE LOGIC GENERATOR
# ---------------------------------------------------------
with tabs[1]:
    st.header("🌲 Decision Tree Rule Generator")
    st.markdown("Below are the clear HR structural rules extracted from the trained model:")
    
    # Show example of user's targeted custom conditional check
    st.info("💡 **Live Rule Simulator:** IF JobSatisfaction < 2 AND MonthlyIncome < 4000 ➡️ **High Attrition Risk**")
    
    # Real extracted rules
    tree_text = export_text(models["Decision Tree"], feature_names=list(X_features.columns))
    st.code(tree_text, language="text")

# ---------------------------------------------------------
# TAB 3: SVM ACCURACY COMPARISON
# ---------------------------------------------------------
with tabs[2]:
    st.header("📈 Model Performance & Accuracy Benchmarks")
    st.markdown("Comparing classifications scores against the **Support Vector Machine (SVM)** model:")
    
    # Construct comparison table
    acc_data = pd.DataFrame({
        "Algorithm Model": accuracies.keys(),
        "Accuracy Score": [f"{v*100:.2f}%" for v in accuracies.values()]
    })
    st.table(acc_data)
    st.bar_chart(data=pd.DataFrame(accuracies.values(), index=accuracies.keys(), columns=["Accuracy"]))

# ---------------------------------------------------------
# TAB 4: KNN SIMILAR EMPLOYEES
# ---------------------------------------------------------
with tabs[3]:
    st.header("👥 KNN: Pattern-Matching Similar Profiles")
    st.markdown("Locating the closest peer matches from the organizational historical logs based on the current metrics:")
    
    # Query neighbor locations using metric distances
    distances, indices = models["KNN"].kneighbors(input_scaled, n_neighbors=3)
    similar_profiles = df_raw.iloc[indices[0]]
    
    # Clean display output
    display_cols = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'Attrition']
    st.dataframe(similar_profiles[display_cols])

# ---------------------------------------------------------
# TAB 5: PCA & K-MEANS EMPLOYEES SEGMENTATION
# ---------------------------------------------------------
with tabs[4]:
    st.header("🧩 Unsupervised Learning & Clustering")
    
    # Dimensionality Reduction Output
    st.subheader("PCA (Dimensionality Reduction)")
    input_pca = pca.transform(input_scaled)
    st.write(f"The input profile has been reduced to coordinates: **[{input_pca[0][0]:.3f}, {input_pca[0][1]:.3f}]**")
    
    st.divider()
    
    # K-Means Segment Assignment
    st.subheader("K-Means Employee Segmentation")
    cluster_id = kmeans.predict(input_scaled)[0]
    
    # Direct explicit cluster mapping defined in your task rules
    cluster_mapping = {
        0: {"name": "Cluster 1: High Performers", "box": st.success, "desc": "Demonstrates high stability indicators, consistent experience footprints, and solid structural metrics."},
        1: {"name": "Cluster 2: At Risk", "box": st.error, "desc": "Exhibits features heavily overlapping with historical turnover trends. Requires immediate HR feedback cycles."},
        2: {"name": "Cluster 3: New Employees", "box": st.info, "desc": "Features lower tenure lengths and standard adaptive traits typical of fresh organizational onboarding."}
    }
    
    current_cluster = cluster_mapping[cluster_id]
    current_cluster["box"](f"Target Assignment: **{current_cluster['name']}**")
    st.write(current_cluster["desc"])
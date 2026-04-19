import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Setup Path dan Load Model
BASE_DIR = Path(__file__).parent
CLF_MODEL_PATH = BASE_DIR / 'artifacts' / 'classification_pipeline.pkl'
REG_MODEL_PATH = BASE_DIR / 'artifacts' / 'regression_pipeline.pkl'

model_clf = joblib.load(CLF_MODEL_PATH)
model_reg = joblib.load(REG_MODEL_PATH)

def main():
    st.set_page_config(page_title="Student Placement Predictor", layout="wide")
    
    st.title('Student Placement and Salary Prediction')

    # Sidebar & Form
    st.sidebar.header("User Input Form")
    
    with st.sidebar.form(key='input_form'):
        st.subheader("Personal & Academic")
        student_id = st.number_input("Student ID", value=0)
        gender = st.radio("Gender", ["Female", "Male"])
        extracurricular = st.radio("Extracurricular Activities", ["No", "Yes"])
        
        ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 75.0)
        hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 70.0)
        degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 70.0)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 3.5)
        
        st.subheader("Skills & Experience")
        entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 70.0)
        attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 80.0)
        tech_score = st.number_input("Technical Skill Score", 0.0, 100.0, 75.0)
        soft_score = st.number_input("Soft Skill Score", 0.0, 100.0, 80.0)
        
        internship = st.number_input("Internship Count", 0, 10, 0)
        work_exp = st.number_input("Work Experience (Months)", 0, 100, 0)
        live_projects = st.number_input("Live Projects", 0, 10, 0)
        certifications = st.number_input("Certifications", 0, 10, 0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        
        # Tombol Submit di dalam form
        submit_button = st.form_submit_button(label='Predict Now')

    # Mapping Data
    data = {
        'student_id': student_id, 'gender': gender, 'ssc_percentage': ssc_p,
        'hsc_percentage': hsc_p, 'degree_percentage': degree_p,
        'extracurricular_activities': extracurricular, 'entrance_exam_score': entrance_score,
        'cgpa': cgpa, 'internship_count': internship, 'work_experience_months': work_exp,
        'attendance_percentage': attendance, 'technical_skill_score': tech_score,
        'soft_skill_score': soft_score, 'live_projects': live_projects,
        'certifications': certifications, 'backlogs': backlogs
    }
    
    df_input = pd.DataFrame([data])
    expected_columns = model_clf.feature_names_in_
    df_input = df_input[expected_columns]

    # Hasil dan Visualisasi
    if submit_button:
        st.subheader("Analysis Results")
        col1, col2 = st.columns([1, 1])

        with col1:
            # Prediksi Status
            prediction_status = model_clf.predict(df_input)[0]
            if prediction_status == 1:
                st.info("Status: PLACED")
                prediction_salary = model_reg.predict(df_input)[0]
                st.metric(label="Estimated Salary Package", value=f"{prediction_salary:.2f} LPA")
            else:
                st.warning("Status: NOT PLACED")
                st.write("Profil mahasiswa belum memenuhi kriteria penempatan kerja.")

        with col2:
            # Data Visualization
            st.write("**Student Competency Profile**")
            # bar chart
            viz_data = pd.DataFrame({
                'Category': ['SSC', 'HSC', 'Degree', 'Entrance', 'Tech', 'Soft'],
                'Score': [ssc_p, hsc_p, degree_p, entrance_score, tech_score, soft_score]
            })
            st.bar_chart(viz_data.set_index('Category'))

if __name__ == "__main__":
    main()

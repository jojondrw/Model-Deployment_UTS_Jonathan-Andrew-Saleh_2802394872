import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Setup Path dan Load Model
BASE_DIR = Path(__file__).parent
CLF_MODEL_PATH = BASE_DIR / 'artifacts' / 'classification_pipeline.pkl'
REG_MODEL_PATH = BASE_DIR / 'artifacts' / 'regression_pipeline.pkl'

model_clf = joblib.load(CLF_MODEL_PATH)
model_reg = joblib.load(REG_MODEL_PATH)

def main():
    st.set_page_config(page_title="Student Placement and Salary Predictor", layout="wide")
    
    st.title('Student Placement and Salary Prediction')

    st.sidebar.header("Input Data Lengkap")
    
    # Kategori 1: Informasi Dasar
    st.sidebar.subheader("Informasi Dasar")
    student_id = st.sidebar.number_input("Student ID", value=0)
    gender = st.sidebar.radio("Gender", ["Female", "Male"])
    extracurricular = st.sidebar.radio("Extracurricular Activities", ["No", "Yes"])
    
    # Kategori 2: Akademik
    st.sidebar.subheader("Riwayat Akademik")
    ssc_p = st.sidebar.number_input("SSC Percentage", 0.0, 100.0, 75.0)
    hsc_p = st.sidebar.number_input("HSC Percentage", 0.0, 100.0, 70.0)
    degree_p = st.sidebar.number_input("Degree Percentage", 0.0, 100.0, 70.0)
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 3.5)
    entrance_score = st.sidebar.number_input("Entrance Exam Score", 0.0, 100.0, 70.0)
    attendance = st.sidebar.number_input("Attendance Percentage", 0.0, 100.0, 80.0)
    backlogs = st.sidebar.number_input("Backlogs", 0, 10, 0)

    # Kategori 3: Pengalaman
    st.sidebar.subheader("Pengalaman dan Proyek")
    internship = st.sidebar.number_input("Internship Count", 0, 10, 0)
    work_exp = st.sidebar.number_input("Work Experience (Months)", 0, 100, 0)
    live_projects = st.sidebar.number_input("Live Projects", 0, 10, 0)
    certifications = st.sidebar.number_input("Certifications", 0, 10, 0)

    # Kategori 4: Skill
    st.sidebar.subheader("Skor Skill")
    tech_score = st.sidebar.number_input("Technical Skill Score", 0.0, 100.0, 75.0)
    soft_score = st.sidebar.number_input("Soft Skill Score", 0.0, 100.0, 80.0)

    # Mapping data
    data = {
        'student_id': student_id,
        'gender': gender,
        'ssc_percentage': ssc_p,
        'hsc_percentage': hsc_p,
        'degree_percentage': degree_p,
        'extracurricular_activities': extracurricular,
        'entrance_exam_score': entrance_score,
        'cgpa': cgpa,
        'internship_count': internship,
        'work_experience_months': work_exp,
        'attendance_percentage': attendance,
        'technical_skill_score': tech_score,
        'soft_skill_score': soft_score,
        'live_projects': live_projects,
        'certifications': certifications,
        'backlogs': backlogs
    }
    
    # Buat DataFrame
    df_input = pd.DataFrame([data])

    # --- PENTING: Memaksa urutan kolom sesuai training ---
    try:
        # Mengambil urutan kolom yang diharapkan model (dari pipeline)
        expected_columns = model_clf.feature_names_in_
        df_input = df_input[expected_columns]
    except AttributeError:
        # Jika scikit-learn versi lama, kita handle manual urutannya (sesuaikan dengan B.csv)
        pass

    if st.sidebar.button("Run Prediction"):
        st.subheader("Hasil Analisis")
        
        # Prediksi Status
        prediction_status = model_clf.predict(df_input)[0]
        
        if prediction_status == 1:
            st.info("Status: PLACED")
            
            # Prediksi Gaji
            prediction_salary = model_reg.predict(df_input)[0]
            st.metric(label="Estimated Salary Package", value=f"{prediction_salary:.2f} LPA")
            st.write(f"Mahasiswa diprediksi lulus seleksi dengan estimasi gaji {prediction_salary:.2f} LPA.")
        else:
            st.warning("Status: NOT PLACED")
            st.write("Profil mahasiswa saat ini belum memenuhi kriteria penempatan kerja.")

if __name__ == "__main__":
    main()

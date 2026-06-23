import os
import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.trainer_pipeline import TrainerPipeline
from src.pipeline.predict_pipeline import PredictPipeline

# Set Page Config
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling for a premium feel
st.markdown("""
<style>
    .metric-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def load_dataset():
    """
    Tries to load raw or train datasets from standard artifacts/notebook paths.
    """
    paths = [
        os.path.join('artifacts', 'raw_data.csv'),
        os.path.join('notebook', 'data', 'stud.csv'),
        os.path.join('artifacts', 'train_data.csv')
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None


def main():
    st.title("🎓 Student Performance Prediction & Analytics")
    st.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigation", 
        ["Make Prediction", "Batch CSV Prediction", "EDA Dashboard", "Train Model"]
    )
    
    if page == "Train Model":
        train_page()
    elif page == "Make Prediction":
        predict_page()
    elif page == "Batch CSV Prediction":
        batch_predict_page()
    elif page == "EDA Dashboard":
        eda_page()


def train_page():
    st.subheader("⚙️ Train Machine Learning Model")
    st.write("This will run the data ingestion, preprocessing, and model training pipelines.")
    
    if st.button("Run Model Training Pipeline", type="primary"):
        with st.spinner("Executing pipeline modules..."):
            try:
                trainer_pipeline = TrainerPipeline()
                trainer_pipeline.initiate_trainer_pipeline()
                st.success("Pipeline executed! Trained Decision Tree model saved in artifacts/model.pkl")
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")


def predict_page():
    st.subheader("🔮 Predict Student Performance")
    st.write("Enter student profile data to predict their total performance score.")

    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        parent_education = st.selectbox(
            "Parental Level of Education", 
            ["high school", "some college", "associate's degree", 
             "bachelor's degree", "master's degree", "some high school"]
        )
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    
    with col2:
        test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
        math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    if st.button("Predict Total Score", type="primary"):
        try:
            features = {
                'gender': gender,
                'race/ethnicity': ethnicity,
                'parental level of education': parent_education,
                'lunch': lunch,
                'test preparation course': test_prep,
                'math score': math_score,
                'reading score': reading_score,
                'writing score': writing_score
            }
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(features)
            
            st.markdown("### Results Summary")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Predicted Total Score", f"{prediction[0]:.2f}")
            with col_res2:
                # Average score benchmark comparison
                avg_total = 205.0 # baseline average (approx 68 per subject)
                diff = prediction[0] - avg_total
                st.metric(
                    "Difference from Benchmark Avg (205)", 
                    f"{diff:+.2f}", 
                    delta_color="normal"
                )
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")


def batch_predict_page():
    st.subheader("📁 Batch CSV Prediction")
    st.write("Upload a CSV file containing student records to predict total performance scores in bulk.")
    
    # Show expected column template format
    with st.expander("Required CSV Format Template"):
        st.markdown("""
        The uploaded CSV must contain the following columns exactly (order does not matter):
        `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, `math score`, `reading score`, `writing score`
        """)
        sample_df = pd.DataFrame([{
            "gender": "female",
            "race/ethnicity": "group B",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "none",
            "math score": 72,
            "reading score": 72,
            "writing score": 74
        }])
        st.dataframe(sample_df)
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.dataframe(input_df.head())
            
            # Check columns
            required_cols = [
                'gender', 'race/ethnicity', 'parental level of education', 
                'lunch', 'test preparation course', 'math score', 
                'reading score', 'writing score'
            ]
            
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                if st.button("Generate Batch Predictions", type="primary"):
                    with st.spinner("Running batch predictions..."):
                        predict_pipeline = PredictPipeline()
                        predictions = []
                        
                        # Loop through rows to make safe row-by-row prediction
                        for _, row in input_df.iterrows():
                            features = row.to_dict()
                            pred = predict_pipeline.predict(features)
                            predictions.append(round(pred[0], 2))
                            
                        input_df['Predicted Total Score'] = predictions
                        
                        st.subheader("Predictions Output")
                        st.dataframe(input_df)
                        
                        # CSV download option
                        csv = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name="student_predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")


def eda_page():
    st.subheader("📊 Student Performance EDA Dashboard")
    st.write("Exploratory insights and descriptive analysis of the student performance dataset.")
    
    df = load_dataset()
    
    if df is None:
        st.warning("No dataset found in artifacts/raw_data.csv or notebook/data/stud.csv. Please run training first to generate data.")
        return
        
    # Calculate total_score if not present
    if 'total_score' not in df.columns:
        df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
        
    # 1. Summary Cards
    col_card1, col_card2, col_card3, col_card4 = st.columns(4)
    with col_card1:
        st.markdown(f"<div class='metric-box'><h6>Total Records</h6><h3>{len(df)}</h3></div>", unsafe_allow_html=True)
    with col_card2:
        st.markdown(f"<div class='metric-box'><h6>Avg Math Score</h6><h3>{df['math score'].mean():.1f}</h3></div>", unsafe_allow_html=True)
    with col_card3:
        st.markdown(f"<div class='metric-box'><h6>Avg Reading Score</h6><h3>{df['reading score'].mean():.1f}</h3></div>", unsafe_allow_html=True)
    with col_card4:
        st.markdown(f"<div class='metric-box'><h6>Avg Writing Score</h6><h3>{df['writing score'].mean():.1f}</h3></div>", unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 2. Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Impact of Parental Level of Education")
        # Average total score grouped by parental education
        parent_group = df.groupby('parental level of education')['total_score'].mean().reset_index()
        parent_group = parent_group.sort_values(by='total_score', ascending=False)
        st.bar_chart(parent_group, x='parental level of education', y='total_score')
        
        st.markdown("#### Math Score vs. Reading Score (by Gender)")
        st.scatter_chart(df, x='math score', y='reading score', color='gender')
        
    with col_chart2:
        st.markdown("#### Impact of Test Preparation Course")
        prep_group = df.groupby('test preparation course')['total_score'].mean().reset_index()
        st.bar_chart(prep_group, x='test preparation course', y='total_score')
        
        st.markdown("#### Score Distribution (Math, Reading, Writing)")
        # Show histogram chart for scores
        score_df = df[['math score', 'reading score', 'writing score']]
        st.line_chart(score_df)


if __name__ == "__main__":
    main()
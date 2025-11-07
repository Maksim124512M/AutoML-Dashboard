import streamlit as st
import pandas as pd

from src.model_utils import train_model
from src.visual_utils import plot_metrics

st.title("Machine Learning Dashboard")

uploaded_file = st.file_uploader("Download the CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of the dataset:")
    st.dataframe(df.head())

    # User inputs for target variable and task type
    target = st.selectbox("Select the target variable:", df.columns)
    task_type = st.radio("Task type:", ["Classification", "Regression"])

    # Train models button
    if st.button("Train Models"):
        # Train models and display results
        try:
            with st.spinner("Training in progress..."):
                df_results = train_model(df=df, target=target, task_type=task_type)

            st.success('Training completed successfully!')
        
            st.subheader("Model Results:")
            st.dataframe(df_results)

            fig = plot_metrics(df_results, task_type)
            st.pyplot(fig)
        except ValueError as e:
            st.warning(e)
        except Exception as e:
            st.error(f'Unexpected error: {e}')

else:
    st.info("Please upload a CSV file to begin.")
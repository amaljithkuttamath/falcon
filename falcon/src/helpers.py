import numpy as np


import pandas as pd
import numpy as np
from scipy import stats

import threading

from llama_api import query_llama_api
import streamlit as st


def generate_data_quality_report(df):
    try:
        # Initialize the report
        report_parts = []

        # Missing Values
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_values_report = pd.DataFrame({"Count": missing_values, "Percentage": missing_percent})
        if missing_values_report.empty:
            missing_values_report = "No missing values detected."
        missing_values_report = missing_values_report[missing_values_report["Count"] > 0].round(2).to_string()

        report_parts.append(f"Missing Values:\n{missing_values_report}\n")

        # Duplicate Rows
        duplicates_report = f"Number of duplicate rows: {df.duplicated().sum()}"
        report_parts.append(f"Duplicates:\n{duplicates_report}\n")

        # Outliers Detection for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            outliers_detection = df[numerical_cols].apply(lambda x: np.abs(stats.zscore(x)) > 3).sum()
            outliers_report = outliers_detection[outliers_detection > 0].to_string()
            if not outliers_report.strip():
                outliers_report = "No significant outliers detected based on Z-score."
        else:
            outliers_report = "No numerical columns to check for outliers."
        report_parts.append(f"Potential Outliers (Numerical Columns):\n{outliers_report}\n")

        # Zero Variance Variables
        zero_variance_cols = df.loc[:, df.nunique() == 1].columns.tolist()
        if zero_variance_cols:
            zero_variance_report = ", ".join(zero_variance_cols)
        else:
            zero_variance_report = "No zero variance variables detected."
        report_parts.append(f"Zero Variance Variables:\n{zero_variance_report}\n")

        # Data Type Consistency
        data_types_report = df.dtypes.to_string()
        report_parts.append(f"Data Types:\n{data_types_report}\n")

        # Memory Usage
        memory_usage = df.memory_usage(deep=True).sum()
        memory_usage_report = f"Total memory usage: {memory_usage} bytes"
        report_parts.append(f"Memory Usage:\n{memory_usage_report}\n")

        # Compile the Data Quality Report
        data_quality_report = "Data Quality Report:\n" + "\n".join(report_parts)
    except Exception as e:
        data_quality_report = f"An error occurred while generating the data quality report: {e}"
        print(data_quality_report)
        data_quality_report = ""
    return data_quality_report


def get_llm_insights(df, model="gemma:2b", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    Generate insights based on the DataFrame using an LLM (Ollama in this case).
    This function sends a detailed prompt based on df to the LLM and returns the LLM's response as a string formatted for Streamlit.
    """
    summary_stats = df.describe(include="all").to_string()
    missing_values = df.isnull().sum().to_string()

    # Generate a data quality report
    data_quality_report = generate_data_quality_report(df)

    prompt_content = f"""
    Given a dataset with {df.shape[0]} rows and {df.shape[1]} columns, covering the following areas: {', '.join(df.columns.tolist())}.

    **Statistical Summary of the Data:**
    {summary_stats}

    **Missing Values:**
    {missing_values}

    **Data Quality Report:**
    {data_quality_report}
    """
    q = """**Request for Analysis (Answer in MARKDOWN FORMAT):**

    1. **Patterns and Correlations:**
    - Identify any apparent patterns within the data.
    - Analyze correlations between variables, especially focusing on unexpected relationships or strong correlations that could suggest causal links.

    2. **Anomalies and Outliers:**
    - Highlight any anomalies or outliers in the data. Describe their potential impact on the dataset and any insights they may offer.

    3. **Predictive Insights:**
    - Based on the data's patterns, correlations, and anomalies, provide insights that could inform future predictions. Mention any models or statistical methods that might be suitable for further analysis.

    4. **Recommendations:**
    - Offer recommendations for further data analysis, including specific statistical tests or data visualization techniques that could yield deeper insights.
    - Suggest additional data collection that could enhance the dataset's completeness or address current limitations.

    Please ensure your analysis is detailed and considers the statistical summary, missing values, and data quality report provided above. Utilize this information to form a comprehensive understanding of the dataset's characteristics and implications."""

    response = query_llama_api(q + prompt_content)
    return response

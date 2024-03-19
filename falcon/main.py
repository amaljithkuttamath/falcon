import pandas as pd
import openai
import streamlit as st

# import streamlit_nested_layout
from classes import get_primer, format_question, run_request
import warnings

warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)
# st.set_page_config(page_icon="chat2vis.png", layout="wide", page_title="Chat2VIS")
from langchain_community.chat_models import ChatOllama
from scipy import stats

st.markdown(
    "<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
            Falcon</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center;padding-top: 0rem;'>Talk to your data.</h2>",
    unsafe_allow_html=True,
)


available_models = {
    # "deepseek":"deepseek-ai/deepseek-coder-33b-instruct",
    "Code Llama": "CodeLlama-34b-Instruct-hf",
}

datasets = {}
# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Sales"] = pd.read_csv("falcon/Real_Estate_Sales_2001-2021_GL.csv")[:100]
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]
import os

openai_key = ""
hf_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

with st.sidebar:
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    try:
        uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets. Give it a nice name.
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            # We want to default the radio button to the newly added dataset
            index_no = len(datasets) - 1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio(
        ":bar_chart: Choose your data:", datasets.keys(), index=index_no
    )  # ,horizontal=True,)

    # Keep a dictionary of whether models are selected or not
    use_model = {}
    for model_desc, model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label, value=True, key=key)

# Text area for query
question = st.text_area("Lets go!", height=10)
go_btn = st.button("Go...")

# Make a list of the models which have been selected
selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(selected_models)

# Execute chatbot query
if go_btn and model_count > 0:
    api_keys_entered = True
    # Check API keys are entered.
    if (
        "ChatGPT-4" in selected_models
        or "ChatGPT-3.5" in selected_models
        or "GPT-3" in selected_models
        or "GPT-3.5 Instruct" in selected_models
    ):
        if not openai_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
    if "Code Llama" in selected_models:
        if not hf_key.startswith("hf_"):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        # Place for plots depending on how many models
        plots = st.columns(model_count)
        # Get the primer for this dataset
        primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
        # Create model, run the request and print the results
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader("Plot:")
                try:
                    # Format the question
                    question_to_ask = format_question(primer1, primer2, question, model_type)
                    # Run the question
                    answer = ""
                    answer = run_request(
                        question_to_ask,
                        available_models[model_type],
                        key=openai_key,
                        alt_key=hf_key,
                    )
                    # the answer is the completed Python script so add to the beginning of the script to it.
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))
                except Exception as e:
                    st.error(
                        f"Unfortunately the code generated from the model contained errors and was unable to execute. {str(e)}"
                    )

import numpy as np


def generate_data_quality_report(df):
    """
    Generate a data quality report for a given DataFrame.

    Parameters:
    - df: pandas.DataFrame

    Returns:
    - A string containing the data quality report.
    """
    # Missing Values
    missing_values_report = df.isnull().sum()
    missing_values_report = missing_values_report[missing_values_report > 0]
    if missing_values_report.empty:
        missing_values_report = "No missing values detected."

    # Duplicate Rows
    duplicates_report = f"Number of duplicate rows: {df.duplicated().sum()}"

    # Outliers Detection for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:  # Check if there are any numerical columns
        outliers_detection = df[numerical_cols].apply(lambda x: np.abs(stats.zscore(x)) > 3).sum()
        outliers_report = outliers_detection[outliers_detection > 0].to_string()
        if not outliers_report.strip():  # Corrected check for empty string
            outliers_report = "No significant outliers detected based on Z-score."
    else:
        outliers_report = "No numerical columns to check for outliers."

    # Data Type Consistency
    data_types_report = df.dtypes.to_string()

    # Compile the Data Quality Report
    data_quality_report = f"""
                            Data Quality Report:

                            Missing Values:
                            {missing_values_report}

                            Duplicates:
                            {duplicates_report}

                            Potential Outliers (Numerical Columns):
                            {outliers_report}

                            Data Types:
                            {data_types_report}
                            """
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

    **Request for Analysis (Answer in MARKDOWN FORMAT):**

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

    Please ensure your analysis is detailed and considers the statistical summary, missing values, and data quality report provided above. Utilize this information to form a comprehensive understanding of the dataset's characteristics and implications.
    """

    llm = ChatOllama(model=model, temperature=temperature, top_p=top_p)
    md_content = llm.invoke(prompt_content)
    return md_content


# tab_list = st.tabs(list(datasets.keys())+["LLM Insights"])
# # tab_list.append("LLM Insights")
# # Load up each tab with a dataset
# for dataset_num, tab in enumerate(tab_list):
#     with tab:
#         # Can't get the name of the tab! Can't index key list. So convert to list and index
#         dataset_name = list(datasets.keys())[dataset_num]
#         st.subheader(dataset_name)
#         st.dataframe(datasets[dataset_name], hide_index=True)

#         # Simple interaction: Ask the user for input and display a response
#         user_query = st.text_input(f"Ask a question about the {dataset_name} dataset:", key=f"query_{dataset_num}")

#         if st.button(f"Submit", key=f"submit_{dataset_num}"):
#             # Here you would process the user_query to generate a response
#             # This example just echoes the query with a placeholder response
#             st.write(f"You asked: {user_query}")
#             st.write("Response: [Placeholder response based on the dataset.]")

# with tab_list[-1]:  # This is the dedicated tab for LLM insights
#     st.subheader("LLM Insights Across Datasets")
#     dataset_name = list(datasets.keys())[dataset_num]
#     for dataset_name, insights in dataset_insights.items():
#         lol = get_llm_insights(datasets[dataset_name])
#         print(lol)
#         st.markdown(f"### {dataset_name}")
#         st.write(insights)
tab_list = st.tabs(list(datasets.keys()) + ["LLM Insights"])

# Load up each dataset tab
for dataset_num, tab in enumerate(tab_list[:-1]):  # Exclude the last tab meant for LLM Insights
    with tab:
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)

        user_query = st.text_input(f"Ask a question about the {dataset_name} dataset:", key=f"query_{dataset_num}")

        if st.button(f"Submit", key=f"submit_{dataset_num}"):
            st.write(f"You asked: {user_query}")
            # Placeholder for processing the user_query
            st.write("Response: [Placeholder response based on the dataset.]")

# Handle the LLM Insights tab separately
with tab_list[-1]:
    st.subheader("LLM Insights")
    # This is where you would provide functionality or insights related to all datasets or offer an interface for LLM queries
    # Example placeholder interaction
    with st.spinner("Processing... Please wait"):
        for dataset_num, tab in enumerate(tab_list[:-1]):
            dataset_name = list(datasets.keys())[dataset_num]
            st.subheader(dataset_name)
            insight = get_llm_insights(datasets[dataset_name])
            st.write(insight.content)
    # llm_query = st.text_input("Ask a general question or request insights:", key="llm_query")
    # if st.button("Submit LLM Query", key="submit_llm"):
    #     st.write(f"You asked: {llm_query}")
    #     # Placeholder for processing the LLM query
    #     st.write("Response: [Placeholder response or insights.]")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

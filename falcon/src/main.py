import pandas as pd
import openai
import streamlit as st

# import streamlit_nested_layout
from classes import get_primer, format_question, format_response, run_request
import warnings
from llama_api import query_llama_api
import threading
from helpers import get_llm_insights
import csv_to_sql
from csv_to_sql import convert_csv_to_sqlite,get_csv_files_in_folder,read_csv_without_header_and_convert_to_sqlite,csv_to_sqlite,query_sqlite
from dotenv import load_dotenv
load_dotenv('falcon/src/.env')


warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

# import mpld3
# import streamlit.components.v1 as components

# st.markdown(
#     "<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
#             Falcon</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<h2 style='text-align: center;padding-top: 0rem;'>Talk to your data.</h2>",
#     unsafe_allow_html=True,
# )


available_models = {
    # "deepseek":"deepseek-ai/deepseek-coder-33b-instruct",
    "Code Llama": "CodeLlama-34b-Instruct-hf",
}

# Initialize session state for insights if not already done
if "llm_insights" not in st.session_state:
    st.session_state.llm_insights = {}


datasets = {}
# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["GRID"] = pd.read_csv("falcon/data/Real_Estate_Sales_2001-2021_GL.csv")
    #######csv_to_sql#################
    folder_path =  'falcon/data/'
    convert_csv_to_sqlite(folder_path)
    #######csv_to_sql##################
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]
import os

openai_key = ""

hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
            #################csv_to_sql###################33
            read_csv_without_header_and_convert_to_sqlite(file_name)
            #################csv_to_sql###################33
            # We want to default the radio button to the newly added dataset
            index_no = len(datasets) - 1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio("Choose your data:", datasets.keys(), index=index_no)  # ,horizontal=True,)

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


def execute_plot_code(code_str):
    """
    Executes a string of code that is expected to generate a matplotlib plot.

    Args:
    code_str (str): A string containing the code to execute. This code should generate a matplotlib plot.

    Returns:
    The matplotlib figure and axes created by the executed code, if any.
    """
    # Dictionary to capture the local variables created by exec()
    print("-" * 10)
    print(code_str)
    print("-" * 10)
    local_vars = {}

    # Execute the given code string
    exec(code_str, globals(), local_vars)

    # Assuming the code creates a figure, try to return it
    fig = local_vars.get("fig", None)
    ax = local_vars.get("ax", None)

    return fig, ax


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
                    # resp = query_llama_api(
                    #     f"write a prompt query to code llama in natural language with detailed specifaction for question {question}",
                    #     primer1,
                    # )
                    # print(resp)
                    question_to_ask = format_question(primer1, primer2, question, model_type)

                    # Run the question
                    # print(question_to_ask)
                    answer = ""
                    answer = run_request(
                        question_to_ask,
                        available_models[model_type],
                        key=openai_key,
                        alt_key=hf_key,
                    )

                    fig, ax = execute_plot_code(answer)

                    # answer = query_llama_api(question_to_ask)
                    # llm_response = format_response(answer)
                    # the answer is the completed Python script so add to the beginning of the script to it.
                    answer = primer2 + answer
                    # print("Model: " + model_type)
                    # print(answer)
                    plot_area = st.empty()
                    # print("-" * 100)
                    # print(llm_response)
                    # print("-" * 100)
                    # try:
                    #     fig_html = mpld3.fig_to_html(fig)
                    #     components.html(fig_html, height=600)
                    # except Exception as e:
                    # print("-" * 10, e)
                    plot_area.pyplot(fig)

                except Exception as e:

                    st.error(
                        f"Unfortunately the code generated from the model contained errors and was unable to execute. {str(e)}"
                    )


tab_list = st.tabs(list(datasets.keys()) + ["LLM Insights"])


# Thread function simplified for clarity
def calculate_insights_thread(datasets, output):
    # Compute insights and store them in a provided output structure
    for dataset_name, dataset_content in datasets.items():
        insight = get_llm_insights(dataset_content)  # Placeholder function
        output[dataset_name] = insight


# Main script
if "llm_insights" not in st.session_state:
    st.session_state["llm_insights"] = {}


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


output = {}
thread = threading.Thread(target=calculate_insights_thread, args=(datasets, output))
thread.start()
thread.join()  # Wait for the thread to complete; consider async handling for real app

# After thread completion, update session state based on output
for dataset_name, insight in output.items():
    st.session_state.llm_insights[dataset_name] = insight

with tab_list[-1]:
    st.subheader("LLM Insights")

    # Button to start the insight calculation

    # Display pre-calculated LLM insights with a spinner to indicate processing
    with st.spinner("Calculating insights... Please refresh the page after a short while to see the results."):
        if st.session_state.llm_insights:
            for dataset_name, insight in st.session_state.llm_insights.items():
                st.subheader(dataset_name)
                st.write(insight or "No insight generated.")
        else:
            st.write("No insights calculated yet. Click 'Calculate Insights' to start.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

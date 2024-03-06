import os
import re

from dotenv import load_dotenv, find_dotenv
import streamlit as st

from langchain_community.chat_models import ChatOllama

from langchain.agents import AgentType

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema.output_parser import OutputParserException
from langchain_core.messages import HumanMessage

import ollama

# from PlotlyTool import PlotlyPythonAstREPLTool

# if os.environ.get('OPENAI_API_KEY') is not None:
#     openai.api_key = os.environ['OPENAI_API_KEY']
# else:
#     _ = load_dotenv(find_dotenv())  # read local .env file
#     openai.api_key = os.environ['OPENAI_API_KEY']


def chat_api(messages, model="gemma:2b", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    The chat API endpoint of the ChatGPT

    Args:
        messages (str): The input messages to the chat API
        model (str): The model, i.e. the LLM
        temperature (float): The temperature parameter
        max_tokens (int): Max number of tokens parameters
        top_p (float): Top P parameter

    Returns:
        str: The LLM response
    """
    plot_flag = False

    if "plot" in messages[-1]["content"].lower():
        plot_flag = True
        code_prompt = """
        Considering the provided pandas DataFrame, generate Python code using Plotly 
        to create a visualization that aligns with the user's request. The visualization 
        should effectively represent the data's key patterns or insights, 
        leveraging appropriate plot types (e.g., scatter plots, line charts, bar charts, etc.) 
        depending on the data characteristics and user's intent. Ensure the Plotly code includes 
        necessary components such as plot titles, axis labels, and legend (if applicable), 
        making the visualization informative and accessible. Return the Python code for Plotly 
        visualization in the format: ```python <code>```. 
        Assume the DataFrame is available as 'df' in the execution environment.
        """
        messages.append({"role": "assistant", "content": code_prompt})

    response = ollama.chat(
        model=model,
        messages=messages,
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
    )
    print(response)
    if plot_flag:
        code = extract_python_code(response["message"]["content"])
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨",
            )
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"```{code}")
            exec(code)
        print(response)
    return response["message"]


def extract_python_code(text):
    pattern = r"```python\s(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None


def chat_with_data_api(
    df, model="gemma:2b", temperature=0.0, max_tokens=256, top_p=0.5
):
    """
    A function that answers data questions from a dataframe.
    """
    error_msg = ""
    # if "plot" in st.session_state.messages[-1]["content"].lower():
    code_prompt = f"""
    DataFrame Details:

    - Head: {df.head()}
    - Shape: {df.shape}

    Please generate Python code that:
    1. Calculates relevant statistics or aggregates data as needed.
    2. Creates Plotly visualizations based on the analyzed data.
    3. Includes clear labels and titles for the plot.

    the code has to end with 
    ```
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    ```
    """

    st.session_state.messages.append({"role": "assistant", "content": code_prompt})

    #     # Assuming ollama.chat() generates a response
    #     response = ollama.chat(
    #         model=model,
    #         messages=st.session_state.messages,
    #     )

    #     # Extract Python code from the response.
    #     code = extract_python_code(response["message"]["content"])

    #     if code is None:
    #         st.warning(
    #             "Couldn't find data to plot in the chat."
    #             "Check if the number of tokens is too low for the data at hand. "
    #             "I.e. if the generated code is cut off, this might be the case.",
    #             icon="🚨",
    #         )
    #         return "Couldn't plot the data"
    #     elif code.strip() == "":
    #         st.warning("Empty code block received in the response.")
    #         return "Couldn't plot the data"
    #     else:
    #         # Add code to display the plot using Streamlit.
    #         code += (
    #             """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
    #         )

    #         # Display the code snippet.
    #         st.code(code, language="python")

    #         # Execute the code.
    #         try:
    #             exec(code)
    #         except Exception as e:
    #             st.error(f"An error occurred during execution: {e}")

    #         return response["message"]["content"]

    # else:
    # llm = ChatOpenAI(
    #     model=model,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=top_p,
    # )
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        return_intermediate_steps=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        # extra_tools=[PlotlyPythonAstREPLTool],
    )

    try:
        answer = pandas_df_agent(st.session_state.messages)
        if answer.get("intermediate_steps"):
            action = answer["intermediate_steps"][-1][0]["tool_input"]["query"]
            st.write(f"Executed the code: ```{action}```")
        return answer["output"]
    except OutputParserException as e:
        error_msg = "OutputParserException error occurred in LangChain agent. Refine your query."
        print(e)
        # st.error(error_msg)
        # return error_msg
    except Exception as e:
        error_msg = f"Unknown error occurred in LangChain agent. Refine your query. Error: {str(e)}"
        print(e)
        # st.error(error_msg)
        # return error_msg


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import json

from langchain.output_parsers import PandasDataFrameOutputParser

from scipy import stats
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
        outliers_detection = (
            df[numerical_cols].apply(lambda x: np.abs(stats.zscore(x)) > 3).sum()
        )
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
    Given a dataset with {df.shape[0]} rows and {df.shape[1]} columns, 
    covering the following areas: {', '.join(df.columns.tolist())}.

    Here's a statistical summary of the data:
    {summary_stats}

    Additionally, there are {missing_values} missing values in the dataset.

    Data quality checks indicate:
    {data_quality_report}

    Please use this information to provide insights on the following:
    1. Identify any apparent patterns or correlations between variables.
    2. Offer insights into anomalies or outliers.
    3. Provide general predictive insights that the data might suggest.
    4. Give recommendations for further data analysis or additional data collection.

    Your response should be comprehensive and actionable.
    """

    llm = ChatOllama(model=model, temperature=temperature, top_p=top_p)
    messages = [
        HumanMessage(content=prompt_content),
    ]

    # Create a prompt from the messages
    prompt = ChatPromptTemplate.from_messages(messages)

    # Execute the chain
    chain = prompt | llm | StrOutputParser()

    # Since dynamic content injection is directly handled in the messages, no need to pass in a separate context
    result = chain.invoke({})

    # Format the result for Streamlit display
    insights = result if result else "No insights generated."
    print("___________")
    print(insights)
    print("___________")

    return str(insights)

import glob
import json
import os

# Mock functions (for complete examples, implement the logic)
import re  # Import regular expression module for parsing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.llms import HuggingFaceHub

from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
from sqlalchemy import create_engine

import pygwalker as pyg
import streamlit.components.v1 as components
from sqlalchemy.engine import reflection


# from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from llama_api import query_mistral, query_llama_api, time_logger
from loguru import logger
from streamlit_text_rating.st_text_rater import st_text_rater


st.set_page_config(
    page_title="Falcon: Talk to your data",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# This is a very cool app!",
    },
)

init_streamlit_comm()
# logo = "falcon/src/eagle_1f985.gif"  # Change this to the path of your logo file

title = "Falcon: Talk to your data"
subtitle = "An interactive data visualization and analysis tool."


# Display Header
col1, col2 = st.columns([1, 4])  # Adjust the ratio based on your preference

# with col1:
#     st.image("🦅", width=50)  # Adjust the width as needed

with col2:
    st.markdown(f"# {title}")
    st.markdown(f"### {subtitle}")


@time_logger
def query_db_to_dataframe(db_path, sql_query):
    """
    Executes a SQL query on a SQLite database and returns the results as a pandas DataFrame.

    Parameters:
    - db_path: String. The path to the SQLite database file.
    - sql_query: String. The SQL query to execute.

    Returns:
    - A pandas DataFrame containing the results of the query.
    """
    # Create a connection to the SQLite database
    with sqlite3.connect(db_path) as conn:
        # Execute the query and return the results as a DataFrame
        df = pd.read_sql_query(sql_query, conn)
        logger.info(df)

    return df


# Function to get DataFrame from the selected database
def get_df_from_database(db_path: str, table_name: str = "your_table_name") -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{db_path}")
    # Create an inspector interface for the engine
    insp = reflection.Inspector.from_engine(engine)

    # Get list of table names from the database
    table_names = insp.get_table_names()
    if table_names:
        # Assuming the first table is the target table
        first_table_name = table_names[0]
        # Retrieve data from the first table
        with engine.connect() as connection:
            df = pd.read_sql_table(first_table_name, con=connection)
        return df
    else:
        raise ValueError("No tables found in the database.")


# Cache the Pygwalker HTML to prevent re-rendering on every interaction
@st.cache_resource
def get_pyg_html(df: pd.DataFrame) -> str:
    html = get_streamlit_html(
        df, spec="./gw0.json", use_kernel_calc=True, debug=False, dark="light", appearance="light", default_tab="data"
    )
    return html


# @lru_cache(maxsize=None)
# def generate_sql_schema_context(db_path):
#     try:
#         with sqlite3.connect(db_path) as conn:
#             cursor = conn.cursor()

#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             tables = cursor.fetchall()

#             if not tables:
#                 return "No tables found in the database."

#             schema_descriptions = []

#             for table in tables:
#                 table_name = table[0]
#                 schema_descriptions.append(f"Table: {table_name}\n")

#                 # Since PRAGMA table_info does not support parameter substitution, ensure table_name is safe.
#                 cursor.execute(f"PRAGMA table_info({table_name})")
#                 columns_info = cursor.fetchall()

#                 for column_info in columns_info:
#                     col_name, col_type = column_info[1], column_info[2]
#                     schema_descriptions.append(f"Column: {col_name}, Type: {col_type}")

#                     if col_type in ["INTEGER", "REAL"]:
#                         cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table_name}")
#                         min_val, max_val, avg_val = cursor.fetchone()
#                         avg_val_str = f"{avg_val:.2f}" if avg_val is not None else "N/A"
#                         schema_descriptions.append(
#                             f"Stats for {col_name} - Min: {min_val}, Max: {max_val}, Avg: {avg_val_str}"
#                         )

#                     cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10")
#                     unique_vals = cursor.fetchall()
#                     unique_vals_str = ", ".join(str(val[0]) for val in unique_vals) + (
#                         "..." if len(unique_vals) == 10 else ""
#                     )
#                     cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
#                     unique_count = cursor.fetchone()[0]
#                     schema_descriptions.append(
#                         f"Unique values in {col_name}: {unique_count} (Sample: {unique_vals_str})"
#                     )

#                 schema_descriptions.append("")
#             return "\n".join(schema_descriptions)
#     except sqlite3.Error as e:
#         raise e
#         return f"An error occurred: {e}"

import pickle


cache_file_path = "sql_schema_cache.pkl"

# Attempt to load existing cache
if os.path.exists(cache_file_path):
    with open(cache_file_path, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}


def cache_result(func):
    def wrapper(db_path):
        # Use db_path as the cache key
        if db_path in cache:
            return cache[db_path]

        result = func(db_path)
        cache[db_path] = result

        # Save the cache to a file
        with open(cache_file_path, "wb") as f:
            pickle.dump(cache, f)

        return result

    return wrapper


# @cache_result
# def generate_sql_schema_context(db_path):
#     """Generates a concise, yet contextually rich narrative-driven schema context of the SQLite database."""
#     try:
#         with sqlite3.connect(db_path) as conn:
#             cursor = conn.cursor()

#             # Fetch all tables in the database
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             tables = cursor.fetchall()

#             if not tables:
#                 return "No tables were found in the database."

#             schema_narrative = ["The database includes the following tables:"]

#             for table in tables:
#                 table_name = table[0]
#                 schema_narrative.append(f"The '{table_name}' table has columns like:")

#                 # Fetch column details
#                 cursor.execute(f"PRAGMA table_info({table_name})")
#                 columns_info = cursor.fetchall()

#                 for column_info in columns_info:
#                     col_name, col_type = column_info[1], column_info[2]
#                     column_summary = f"'{col_name}' ({col_type})"

#                     # Include brief statistical summary for numerical types
#                     if col_type in ["INTEGER", "REAL"]:
#                         cursor.execute(
#                             f"""SELECT MIN({col_name}), MAX({col_name}), AVG({col_name})
#                                FROM {table_name}"""
#                         )
#                         min_val, max_val, avg_val = cursor.fetchone()
#                         avg_val_str = f"{avg_val:.2f}" if avg_val is not None else "N/A"
#                         column_summary += (
#                             f": values range from {min_val} to {max_val}, with an average of {avg_val_str}."
#                         )

#                     # Mention diversity for columns with significant unique values
#                     cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
#                     unique_count = cursor.fetchone()[0]
#                     if unique_count > 10:  # Arbitrary threshold to indicate significant diversity
#                         column_summary += f" It has {unique_count} unique values, indicating a high level of diversity."

#                     schema_narrative.append(column_summary)

#             return " ".join(schema_narrative)
#     except sqlite3.Error as e:
#         return f"An error occurred: {e}"


@time_logger
@cache_result
def generate_sql_schema_context(db_path):
    """Generates a narrative-driven schema context of the SQLite database for better LLM comprehension."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Fetch all tables in the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                return "No tables were discovered in the database."

            schema_narrative = [
                "Let's explore the database. It comprises several tables, each serving a unique purpose."
            ]

            for table in tables:
                table_name = table[0]
                schema_narrative.append(
                    f"Consider the '{table_name}' table, which plays a key role with its specific columns:"
                )

                # Fetch column details
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()

                for column_info in columns_info:
                    col_name, col_type = column_info[1], column_info[2]
                    schema_narrative.append(f"- '{col_name}' is a {col_type} column that stores crucial information.")

                    # Fetch column statistics for numerical types
                    if col_type in ["INTEGER", "REAL"]:
                        cursor.execute(
                            f"""SELECT MIN({col_name}), MAX({col_name}), AVG({col_name})
                                           FROM {table_name}"""
                        )
                        min_val, max_val, avg_val = cursor.fetchone()
                        avg_val_str = f"{avg_val:.2f}" if avg_val is not None else "Not Available"
                        schema_narrative.append(
                            f"  Specifically, its values range from {min_val} to {max_val}, averaging around {avg_val_str}. This gives us insight into the numerical landscape of '{col_name}'."
                        )

                    # Fetch sample of unique values
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10")
                    unique_vals = cursor.fetchall()
                    unique_vals_str = ", ".join(str(val[0]) for val in unique_vals) + (
                        "..." if len(unique_vals) == 10 else ""
                    )
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}")
                    unique_count = cursor.fetchone()[0]
                    schema_narrative.append(
                        f"  It boasts {unique_count} unique values, for instance: {unique_vals_str}. This diversity underpins the column's variability and richness."
                    )

            return " ".join(schema_narrative)
    except sqlite3.Error as e:
        return f"Encountered an error: {e}"


def better_query(question, db_path):
    """
    Enhances a user's query based on the SQL schema context of the specified database,
    aiming to provide a more precise and contextually relevant version of the query.

    Parameters:
    - question (str): The user's original query.
    - db_path (str): The filesystem path to the database.

    Returns:
    - str: The enhanced query or an error message if the API call fails.
    """
    try:
        # Generate the SQL schema context for the database to provide detailed background.
        schema = generate_sql_schema_context(db_path)
        refined_prompt = (
            f"Given a user query and the context of a specific SQL database schema, "
            f"enhance the query for clarity and specificity.\n\n"
            f'User Query: "{question}"\n'
            f"Database Schema Context:\n{schema}\n\n"
            f"Guidelines:\n"
            f"- Consider the relationships and constraints within the schema.\n"
            f"- Refine the query to be more precise, potentially adding specific table or column references.\n"
            f"- Ensure the enhanced query remains under 100 words.\n\n"
            f"Enhanced Query:"
        )

        # Make the API call with the refined prompt.
        response = query_llama_api(refined_prompt)

        # Assuming the API response is a string directly containing the refined question,
        # you may need to adjust based on the actual API response structure.
        refined_question = response.strip()

        return refined_question

    except Exception as e:
        # Gracefully handle any errors encountered during the API call.
        error_message = f"An error occurred while refining the query: {e}"
        print(error_message)
        return error_message


@time_logger
def generate_sql_query(question, db_path, previous_code=None, error_reason=None):
    """
    Generates an SQL query using the Llama API based on a given question and SQL schema.
    The function now includes a more specific prompt to ensure the SQL query is
    clearly enclosed in backticks and distinctly separated from any explanatory text.

    Parameters:
    - question (str): The question to generate an SQL query for.
    - sql_schema (str): The SQL schema of the database to provide as context.

    Returns:
    - str: The extracted SQL query, or an error message if the extraction fails or the request is unsuccessful.
    """
    schema = generate_sql_schema_context(db_path)
    refined_prompt = f"""\
    [INST] <<SYS>>
    **Task:** As a data analyst agent, your main objective is to build a single, conciese `SQLite query` for retrieving data pertinent to a specific inquiry. Emphasize using the `SQLite3 dialect`, with a strong focus on SQLite’s unique syntax, functions, and performance optimization techniques.

    **Instructions:**

    1. **Understand the Question:** Read the question thoroughly to grasp the exact information needed. Identify the entities involved, their relationships, and any particular details requested.

    2. **Construct the Query:**
    - **Select Columns:** Begin with the `SELECT` statement, specifying the columns needed for your analysis.
    - **Apply Filters:** Use the `WHERE` clause to narrow down your data based on specific conditions.
    - **Sort Results:** Optionally, include `ORDER BY` to organize your results in ascending (`ASC`) or descending (`DESC`) order.
    - **Limit Output:** To manage the size and performance of your query, always conclude with `LIMIT 100` at the end of your query. This ensures that only the top 100 records, based on your sorting criteria, are returned. This step is crucial for focusing on the most relevant data and optimizing query execution speed.
    - **Utilize SQLite Functions:** Employ SQLite-specific functions (like `strftime()` for date manipulations) to enhance your query’s capabilities.
    - **Index for Performance:** Utilize indexes on columns in your `WHERE` or `ORDER BY` clauses to speed up query execution.
    - **Type Affinity:** Keep SQLite’s dynamic type system in mind, especially when dealing with data types, to ensure compatibility.


    3. **Reference the Schema:** Refer to the provided schema to understand the tables, fields, and their relationships. This insight will guide you in selecting the appropriate tables or columns for your query.

    4. **Formatting:**
    - Clearly demarcate your query within triple backticks (```) to signify it as a code snippet.
    - Maintain simplicity and clarity in your query, focusing directly on the given task.

    **Leveraging SQLite's Strengths in Your Queries:**

    - Explore SQLite's embedded functions, subqueries, CTEs, and window functions for complex data manipulation.
    - Adapt to SQLite's type affinity system to ensure data consistency and predictability.

    **Review of Previous Attempt:** If there was a previous attempt that failed, reflect on the provided feedback focusing on SQLite's syntax and best practices to guide your corrections.

    <</SYS>>[INST]
    [INST]
    SCHEMA: `{schema}`
    QUESTION: ```{question}```.
    [/INST]
    """

    # Query the Llama API
    refined_prompt = """
    # Generic SQLite Data Analysis Task
    Your task is to conduct an analysis using SQL queries within SQLite3, tailored to a dataset of your choosing. This involves crafting a SQL query that efficiently retrieves, filters, and analyzes data based on specific criteria or questions. Follow the structured approach below to ensure your query provides insightful and relevant results.
    ## Step 1: Schema Exploration
    - **Familiarize Yourself with the Dataset**: Begin by examining the dataset's schema. Understand the structure of the tables, the relationships between them, and the types of data each column holds. This foundational knowledge is crucial for identifying which parts of the dataset will be most relevant to your analysis.
    ## Step 2: Query Development
    - **Column and Data Selection**: Decide on the columns that are most relevant to your analysis. Consider what data is necessary to answer your analytical question.
    - **Applying Filters**: Determine what filters, if any, should be applied to narrow down the data. Filters can be based on specific values, ranges, or conditions relevant to your analytical goals.
    - **Data Aggregation and Grouping**: Use SQL aggregation functions (`COUNT()`, `SUM()`, `AVG()`, etc.) and `GROUP BY` clauses to summarize and analyze the data. This is particularly useful for identifying trends or making comparisons across different data segments.
    - **Sorting and Limiting Results**: Sort your query results in a meaningful order (e.g., ascending or descending) to highlight the most important data points. Limit the number of results returned to focus on the most relevant information.
    ## Step 3: Leveraging SQLite3 Features
    - Utilize SQLite-specific functions and features to enhance your query. This includes functions for date and time manipulation (`strftime()`), string manipulation, and using indexes to optimize query performance.
    - Pay attention to SQLite’s dynamic type system, which may affect how data comparisons and calculations are performed.
    ## Step 4: Preparation for Analysis
    - Design your query to output data in a format that is immediately useful for your intended analysis. Whether your goal involves statistical analysis, trend observation, or data visualization, ensure the query output is conducive to these activities.
    Here's a basic template to guide the construction of your SQL query for data analysis. Customize this template based on the schema of your dataset and the specific questions you aim to answer through your analysis.
    SCHEMA: {}
    QUESTION: {}
    # only generate a single SQLite query, no explanation needed.
    # Your code goes here in enclosed triple backticks:
    ```sql
    """.format(
        schema, question
    ).strip()

    if previous_code and error_reason:
        refined_prompt = f"""**Review of Previous Attempt:** Reflecting on your last query can offer valuable insights. Below is your previous attempt and the reason it failed. Use this feedback to guide your corrections, focusing on SQLite's specific syntax and best practices.\n\nPrevious Attempt: \n{previous_code}\n \n**Error Reason:** {error_reason}.
        # Generic SQLite Data Analysis Task
        Your task is to conduct an analysis using SQL queries within SQLite3, tailored to a dataset of your choosing. This involves crafting a SQL query that efficiently retrieves, filters, and analyzes data based on specific criteria or questions. Follow the structured approach below to ensure your query provides insightful and relevant results.
        SCHEMA: {schema}
        QUESTION: `{question}`
        # only generate a single SQLite query, no explanation needed.
        # Your code goes here in enclosed triple backticks:
        
        ```sql\n"""
    # response = query_llama_api(refined_prompt)
    response = run_request(refined_prompt)
    logger.info(response)  # Logging the response
    # Attempt to extract the SQL query enclosed in backticks from the response
    # match = re.search(r"```(?:sql)?\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", response, re.DOTALL)

    if match:
        sql_query = match.group(1).strip()  # Stripping whitespace for clean extraction
        logger.info(sql_query)
        return sql_query  # Extract the SQL query
    else:
        return "No SQL query generated or query not enclosed in backticks."


@time_logger
def extract_python_code(response):
    """
    Extracts Python code from a response string that might contain the code
    block enclosed in ``` or ```python, removes any line containing `fig.show()`,
    and also removes any line containing `pd.read_csv()`.
    """
    # Try extracting with ```python first
    start_marker = "```python"
    end_marker = "```"
    start_idx = response.find(start_marker)
    if start_idx == -1:
        # If not found, try just ```
        start_marker = "```"
        start_idx = response.find(start_marker)

    # Adjust the start index to account for the marker length
    start_idx += len(start_marker)

    # Find the end of the code block
    end_idx = response.find(end_marker, start_idx)

    # Extract the code
    if start_idx != -1 and end_idx != -1:
        code_block = response[start_idx:end_idx].strip()
        # Split into lines and filter out any line containing `fig.show()` or `pd.read_csv()`
        filtered_lines = [
            line for line in code_block.split("\n") if "fig.show()" not in line and "pd.read_" not in line
        ]
        # Join the lines back into a single string
        return "\n".join(filtered_lines)
    else:
        # If no code block is found, return an empty string or handle as needed
        return ""


def is_code_safe(code):
    # Implement safety checks on the generated code here
    return "import os" not in code and "open(" not in code


@time_logger
def generate_plotly_chart(df, question, attempt=1, max_attempts=3, last_error=""):
    if attempt > max_attempts:
        logger.info("Maximum attempts reached. Unable to generate executable code.")
        return None
    # Details about the DataFrame 'df', including column names and types, and a brief on what we're trying to visualize.
    primer_desc = ""
    for i in df.columns:
        if len(df[i].drop_duplicates()) < 20 and df.dtypes[i] == "O":
            primer_desc = (
                primer_desc
                + "\nThe column '"
                + i
                + "' has categorical values '"
                + "','".join(str(x) for x in df[i].drop_duplicates())
                + "'. "
            )
        elif df.dtypes[i] == "int64" or df.dtypes[i] == "float64":
            primer_desc = (
                primer_desc + "\nThe column '" + i + "' is type " + str(df.dtypes[i]) + " and contains numeric values. "
            )
    primer_desc = (
        primer_desc + "PERFORM filtering and aggregating on the dataframe. If required by the `QUESTION TO PLOT:`.\n"
    )
    primer_desc = primer_desc + "\nUpdate layout with labels appropriately.\n"
    df_info = f"DataFrame 'df' contains columns: {list(df.columns)}. DataFrame shape: {df.shape}. data types: {df.dtypes} infer data context using head: {df.head(5)}"
    question_info = f"QUESTION TO PLOT: {question}"

    # Adjust the prompt to be more specific about the expected output and guidelines for code generation.
    prompt = f"""
    <s>[INST] <<SYS>> This instruction block is designed to guide the model in generating a Python code snippet for visualizing data from a pandas DataFrame using Plotly Express. The model should adhere to the following guidelines:
    - Tailor the Python code snippet to the specifics of the DataFrame structure (`{df_info}`) (`{primer_desc}`)and the visualization question (`{question_info}`).
    - Generate Python code that is executable in a standard Python environment, focusing on creating a straightforward and interactive graph design.
    - Provide the Python code snippet within triple backtick markers to ensure clarity and immediate usability.
    - Dont plot on any geo based data.

    The code snippet should be simple yet effective in visualizing the dataset's insights, making it interactive where possible. The model's output should directly follow the instruction block, encapsulated within a code block formatted as shown below. <</SYS>>

        ```
        import plotly.express as px
        fig = px.some_chart_type(df, x='column_x', y='column_y')
        fig.update_layout(
                font_family="desired_font_family", # Placeholder for the font family.
                legend=dict(
                    title="legend_title", # Use None if no title is desired.
                    orientation="horizontal_or_vertical", # Placeholder for the legend's orientation.
                    y=numeric_value_for_y_position, # Placeholder for the legend's y position.
                    yanchor="anchor_point", # Placeholder for the y anchor.
                    x=numeric_value_for_x_position, # Placeholder for the legend's x position.
                    xanchor="anchor_point" # Placeholder for the x anchor.
                )
            )
        fig.add_annotation(
            text="annotation_text", # Placeholder for the annotation text.
            x=x_position_for_annotation, # Placeholder for the x position of the annotation.
            y=y_position_for_annotation, # Placeholder for the y position of the annotation.
            arrowhead=numeric_value_for_arrowhead_style, # Placeholder for the arrowhead style.
            showarrow=True # Indicates whether to show the arrow.
        )
        ```
    Adapt this structure to fit the visualization requirements mentioned, ensuring all instructions are contained within a single triple backtick code block for immediate usability.

    - Only one code block is required.
    """
    fig = None
    if last_error:
        prompt += "GENERATE NEW CODE BASED ON: " + last_error

    resp = query_llama_api(prompt)
    python_code = extract_python_code(resp)
    logger.info(f"Attempt {attempt}: Generated code:\n{python_code}")

    if not is_code_safe(python_code):
        logger.info("Generated code failed safety checks. Trying again...")
        return generate_plotly_chart(df, question, attempt + 1, max_attempts, "Code failed safety checks.")

    local_namespace = {"df": df, "pd": pd, "px": px}

    try:
        exec(python_code, local_namespace)
        fig = local_namespace.get("fig", None)
        # if fig is None:
        #     raise ValueError("Figure object 'fig' was not created.")
    except Exception as e:
        exception_msg = str(e)
        logger.error(f"generated code:{python_code} Error: {exception_msg}")
        return generate_plotly_chart(df, question, attempt + 1, max_attempts, "")

    return fig


def generate_matplotlib_seaborn_chart(df, attempt=1, max_attempts=3, last_error=""):
    if attempt > max_attempts:
        logger.info("Maximum attempts reached. Unable to generate executable code.")
        return None

    primer_code = """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Your DataFrame 'df' will be directly used here.
    # Please proceed to analyze 'df' and plot the data in a way that is most appropriate for its structure.
    # The plot should be created using Matplotlib without calling 'plt.show()'.
    """.strip()

    error_feedback = f"\n\nFeedback from last attempt: {last_error}" if last_error else ""
    prompt = (
        f"""{primer_code}
    Given a pandas DataFrame 'df' with the sample data as follows: {df.head()}, and its shape: {df.shape}, 
    write Python code to plot this data using Matplotlib. Ensure the code is compatible for 
    execution with the exec() function in Python, including all necessary preparations and considerations for such execution.
    """.strip()
        + error_feedback
    )

    resp = query_llama_api(prompt)
    python_code = extract_python_code(resp)
    logger.info(f"Attempt {attempt}: Generated code:\n{python_code}")

    if not is_code_safe(python_code):
        logger.info("Generated code failed safety checks. Trying again...")
        return generate_matplotlib_seaborn_chart(df, attempt + 1, max_attempts, "Code failed safety checks.")

    local_namespace = {"df": df, "pd": pd, "plt": plt, "sns": sns}

    try:
        exec(python_code, local_namespace)
        fig = local_namespace.get("plt", None)
        if fig is None:
            raise ValueError("Matplotlib figure 'plt' was not properly configured.")
    except Exception as e:
        logger.info(f"Error executing generated code: {e}. Trying again...")
        return generate_matplotlib_seaborn_chart(df, attempt + 1, max_attempts, f"Error: {e}")

    # Instead of returning fig, since Matplotlib does not use a figure object in the same way as Plotly,
    # we ensure the current figure is returned or properly saved within the exec scope.
    return fig


def format_response(res):
    # res = extract_code(res)
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    start_index = res.find("fig.show()")
    if start_index != -1:
        # Find the start of the line
        line_start = res.rfind("\n", 0, start_index)
        # Find the end of the line
        line_end = res.find("\n", start_index)
        # Remove the line
        res = res[:line_start] + res[line_end:]
    return res


@time_logger
def run_request(question_to_ask):
    # print(question_to_ask)

    # Hugging Face model
    llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        repo_id="codellama/" + "CodeLlama-34b-Instruct-hf",
        # repo_id= "mistralai/Mistral-7B-Instruct-v0.2",
        # repo_id="NousResearch/Hermes-2-Pro-Mistral-7B",
        model_kwargs={"temperature": 0.01, "max_new_tokens": 512, "top_p": 0.95},
    )
    llm_prompt = PromptTemplate.from_template(question_to_ask)
    # print(llm_prompt)
    llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
    llm_response = llm_chain.predict()
    # rejig the response
    return llm_response


@time_logger
def execute_plot_code(code: str):
    """
    Executes a string of code that generates a matplotlib plot and returns the figure and axes.

    Args:
        code (str): The code to execute.

    Returns:
        A tuple containing the matplotlib figure and axes, or None if they were not created.
    """
    local_vars = {}
    exec(code, globals(), local_vars)
    fig = local_vars.get("fig")
    ax = local_vars.get("ax")
    return fig, ax


def get_primer_plotly(df_dataset):
    # Primer function to take a dataframe and its name,
    # analyze the columns to add descriptions for columns with less than 20 unique values,
    # and prepare a code snippet for plotting with Plotly Express.
    primer_desc = (
        "Use a dataframe called df from data_file.csv with columns '"
        + "','".join(str(x) for x in df_dataset.columns)
        + "'. "
    )
    primer_desc += df_dataset.head().to_string()
    # primer_desc += f"\n Memory usage: {df_dataset.memory_usage(deep=True).sum()} bytes"
    # primer_desc += "\n change datatype of any time / date columns."
    for i in df_dataset.columns:
        data_type = df_dataset.dtypes[i]
        if len(df_dataset[i].drop_duplicates()) < 20:
            primer_desc += (
                "\nThe column '"
                + i
                + f"' has datatype {data_type} and contains unique values: '"
                + "','".join(str(x) for x in df_dataset[i].drop_duplicates())
                + "'. "
            )
        # elif df_dataset.dtypes[i] == "int64" or df_dataset.dtypes[i] == "float64":
        #     primer_desc += (
        #         "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "
        #     )
    # primer_desc = (
    #     primer_desc
    #     + "if needed, Extract latitude and longitude coordinates from a column containing strings in the format 'POINT (longitude latitude)', and plot the points on a geographical map using Plotly"
    # )
    # primer_desc += (
    #     "PERFORM filtering and aggregating on the dataframe using pandas If required by the `QUESTION TO PLOT:`.\n"
    # )
    primer_desc += "\nLabel the x and y axes appropriately."
    primer_desc += "\nCreate dynamic and interactive visualizations that allow for exploring the dataset visually."
    primer_desc += "\n If multiple plots are required, create subplots using Plotly Express.\nThe script should only include code, no comments.\n"

    primer_desc += "QUESTION TO PLOT: `{}`"

    # Primer code for Plotly Express
    primer_code = "import pandas as pd\nimport plotly.express as px\n"

    # primer_code += "# Plotly graph_objects plot.\n"
    primer_code += "# code here\n"
    # primer_code += "# expose fig object dont include `fig.show()` in the code snippet\n"
    # Example Plotly Express plot (generic placeholder, adjust as needed)

    # primer_code += "fig = px.bar(df, x='[X_COLUMN]', y='[Y_COLUMN]', color='[CATEGORY_COLUMN]', barmode='group')\n"
    # primer_code += "fig.update_layout(xaxis_title='X Axis Title', yaxis_title='Y Axis Title', title='Plot Title')\n"
    # primer_code += "fig\n"

    return primer_desc, primer_code


def format_question(primer_desc, primer_code, question):
    primer_desc = primer_desc.format(" ")
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return '"""\n' + primer_desc + question + '\n"""\n' + primer_code


def generate_plotly(df, prompt):
    primer1, primer2 = get_primer_plotly(df)
    question_to_ask = format_question(primer1, primer2, prompt)
    answer = run_request(question_to_ask)
    answer = format_response(answer)
    print(answer)
    try:
        fig, ax = execute_plot_code(answer)
        return fig
    except Exception as e:
        logger.error(e)
        return None


def get_table_names(db_path):
    """Retrieve a list of all table names in the specified SQLite database."""
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    df_tables = query_db_to_dataframe(db_path, query)
    return df_tables["name"].tolist()


def preprocess_column_names(df):
    """Preprocess DataFrame column names: lowercase, remove special characters, replace spaces with underscores."""
    df.columns = [re.sub(r"\W+", "", column).lower().replace(" ", "_") for column in df.columns]
    return df


def sanitize_table_name(table_name):
    """Sanitize the table name to ensure it's a valid SQL table name."""
    # Replace invalid characters with an underscore
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
    # Ensure the table name does not start with a digit
    if re.match(r"^\d", sanitized_name):
        sanitized_name = "_" + sanitized_name
    return sanitized_name


def csv_to_sqlite(csv_file, db_path, table_name=None):
    """
    Convert a CSV file to a SQLite database with preprocessed column names.

    :param csv_file: Streamlit UploadedFile object or file path as str.
    :param db_path: Path to the SQLite database file where the table will be created.
    :param table_name: Optional; name of the table to create/replace in the database.
    """
    try:
        # Check if csv_file is a filepath (str) or an UploadedFile object
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            # Read directly from the UploadedFile object
            df = pd.read_csv(csv_file)

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            return

        # Preprocess column names
        df = preprocess_column_names(df)

        # Determine the table name if not provided
        if table_name is None:
            base_name = os.path.basename(csv_file.name) if not isinstance(csv_file, str) else os.path.basename(csv_file)
            table_name = os.path.splitext(base_name)[0]
            table_name = sanitize_table_name(table_name)

        # Save the DataFrame to an SQLite table
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        st.success(f"Data uploaded successfully to table {table_name} in the database.")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or not in the expected format.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


class DatabaseManager:
    def __init__(self, upload_directory="uploaded_databases", predefined_db_paths=None):
        self.upload_directory = upload_directory
        self.predefined_db_paths = predefined_db_paths or []
        self.ensure_directory_exists(self.upload_directory)

    @staticmethod
    def ensure_directory_exists(directory):
        """Ensure the target directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_uploaded_databases(self):
        """Retrieve filenames of databases uploaded by the user."""
        search_pattern = os.path.join(self.upload_directory, "*.db")
        return [os.path.basename(path) for path in glob.glob(search_pattern)]

    def get_available_databases(self):
        """Combine predefined databases with uploaded ones, returning only filenames."""
        predefined_databases = [os.path.basename(path) for path in self.predefined_db_paths]
        uploaded_databases = self.get_uploaded_databases()
        return predefined_databases + uploaded_databases

    def get_full_path(self, filename):
        """Map a selected filename back to its full path."""
        if filename in [os.path.basename(path) for path in self.predefined_db_paths]:
            return self.predefined_db_paths[
                [os.path.basename(path) for path in self.predefined_db_paths].index(filename)
            ]
        else:
            return os.path.join(self.upload_directory, filename)

    def handle_csv_upload(self, uploaded_file):
        """Process an uploaded CSV file, converting it to a SQLite database."""
        if uploaded_file is not None:
            db_path = os.path.join(self.upload_directory, uploaded_file.name.replace(".csv", ".db"))
            # Assuming csv_to_sqlite is a predefined function
            csv_to_sqlite(uploaded_file, db_path)
            st.sidebar.success(f"Database created successfully at {db_path}")

    def get_first_table_name(self, filename):
        """Get the name of the first table in the database."""
        full_path = self.get_full_path(filename)
        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        first_table = cursor.fetchone()
        conn.close()
        if first_table:
            return first_table[0]
        else:
            return None

    def get_top_rows(self, filename, rows=10):
        """Get the top rows from the first table in the database."""
        table_name = self.get_first_table_name(filename)
        if table_name is None:
            print("No tables found in the database.")
            return None
        full_path = self.get_full_path(filename)
        conn = sqlite3.connect(full_path)
        query = f"SELECT * FROM {table_name} LIMIT {rows}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df


# Initialize the DatabaseManager with predefined databases
predefined_databases_paths = ["falcon/real_estate_transactions_pandas.db"]
db_manager = DatabaseManager(predefined_db_paths=predefined_databases_paths)

# Streamlit UI for uploading CSV
uploaded_file = st.sidebar.file_uploader("Upload a CSV file to convert to SQLite", type=["csv"])
db_manager.handle_csv_upload(uploaded_file)

# Refresh and display the available databases for selection
available_databases = db_manager.get_available_databases()
selected_db_filename = st.sidebar.selectbox("Select a database:", available_databases)

# Get the full path for the selected database
selected_db_path = db_manager.get_full_path(selected_db_filename)

st.dataframe(db_manager.get_top_rows(selected_db_filename))

tab1, tab2, tab3 = st.tabs(["Chat", "EDA", "Summary Insights"])


with tab1:
    # Check if a database has been selected before proceeding
    if not selected_db_path:
        st.warning("Please select a database from the sidebar before proceeding.")
    else:
        # dfx = query_db_to_dataframe(db_path, f"SELECT * FROM {get_table_names(db_path)} LIMIT 5")
        # print("---------------------------")
        # print(dfx)
        # print("---------------------------")
        # st.dataframe(dfx.head())

        if "messages" not in st.session_state.keys():  # Initialize the chat message history
            st.session_state.messages = []

        if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                try:
                    if "df" in message:
                        with st.expander("View DataFrame"):
                            st.dataframe(message["df"])
                    if "chart" in message:
                        with st.expander("View Chart"):
                            st.plotly_chart(message["chart"])
                except Exception as e:
                    pass
        # if st.session_state.messages[-1]["role"] != "assistant":

        if prompt:
            with st.chat_message("assistant"):
                text_write = (
                    "I am unable to provide an answer to your question. Sorry for the inconvenience. Please try again."
                )
                df = ""
                chart = ""
                sql_query = ""
                with st.spinner("Thinking..."):
                    logger.info(selected_db_path)
                    success = False
                    # Initial SQL query generation based on user input
                    max_attempts = 5
                    attempt = 0
                    with st.status("Querying data...") as status:
                        while attempt < max_attempts and not success:
                            attempt += 1  # Increment attempt at the start to reflect the actual attempt number
                            try:
                                sql_query = generate_sql_query(prompt, selected_db_path)
                                # Informative feedback about the current attempt and action
                                # st.text(f"Attempt {attempt}")
                                st.text("Current SQL query being executed:")
                                st.code(sql_query, language="sql")

                                # Attempt to query the database and store results in a DataFrame
                                df = query_db_to_dataframe(selected_db_path, sql_query)
                                # pyg_html = pyg.to_html(df)

                                # # Embed the HTML into the Streamlit app
                                # components.html(pyg_html, height=600, scrolling=True)
                                st.dataframe(df)
                                success = True  # Query was successful

                                # Update status message based on success or continuation
                                status_msg = "Download complete!" if success else "Adjusting query for next attempt..."
                                status.text(status_msg)

                                if success:
                                    st.success(f"Data successfully retrieved on attempt {attempt}!")
                                else:
                                    st.info("Trying again with an adjusted query...")

                            except Exception as e:
                                logger.error(f"Attempt {attempt} failed: {e}")  # Log the error

                                if attempt < max_attempts:
                                    # Generate a new SQL query for the next attempt, adjusting based on the error encountered
                                    sql_query = generate_sql_query(
                                        prompt,
                                        previous_code=sql_query,
                                        error_reason=f"Attempt {attempt} failed: {e}",
                                        db_path=selected_db_path,
                                    )
                                    st.warning(
                                        f"Encountered an issue. Trying again with a different query (Attempt {attempt + 1}/{max_attempts})..."
                                    )
                                else:
                                    # Final attempt has failed
                                    st.error(
                                        "All attempts to query the database have failed. Please check the query and database connection."
                                    )

                        # st.write("Here are the top 10 most expensive properties based on your query:")
                    api_response = ""
                    if success:
                        if not df.empty:
                            # Limit the data to the top 50 records for analysis if the DataFrame has more than 50 rows.
                            dfx = df.head(100) if len(df) > 100 else df

                            # Convert the limited DataFrame to a JSON string for structured data representation.
                            # df_json = dfx.to_json(orient="records", lines=True)

                            # Generate descriptive statistics for the entire DataFrame to summarize the data.
                            # descriptive_stats = df.describe().to_string()

                            # Prepare the data summary and descriptive statistics for presentation.
                            data_summary = f"""
                            Data: {dfx}.
                            """

                        else:
                            data_summary = "No data found."

                        # Format the prompt to summarize the query results in a professional tone, excluding raw data tables.
                        prompt_summary = f"""
                        ## Summary of Findings from the Query: "{sql_query}"

                        ### Objective
                        To answer the question: "{prompt}", this summary breaks down the most important points from the data. As a data analyst, my job is to focus on what's really important for this question.

                        ### Approach
                        - **Simplify**: Only include information that answers our main question.
                        - **Clear Insights**: Make sure every point is easy to understand and directly from the data.
                        - **No Guesses**: Stick to what the data shows us, without adding extra thoughts that aren't based on the data.

                        ### Key Insights
                        {data_summary}

                        ### Conclusion
                        This summary keeps things simple and to the point, making sure you get the key insights from the data that directly relate to the question, Don't include any raw data in the response.

                        # Respond in a professional conversation tone, No pre-amble.
                        """

                        prompt_summary = prompt_summary.strip()
                        future_chart = None
                        # xy = st.container()
                        summary = st.empty()
                        # with xy:
                        with summary:
                            text_write = query_mistral(
                                prompt_summary, callbacks=[StreamlitCallbackHandler(summary, expand_new_thoughts=True)]
                            )["text"]
                            st.markdown(text_write)
                        try:
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                # Submitting task for API call
                                # future_api_call = executor.submit(query_mistral, prompt_summary, [StreamlitCallbackHandler(xy)])

                                # Submitting task for chart generation if DataFrame is not empty
                                if not df.empty:
                                    future_chart = executor.submit(generate_plotly, df, prompt)

                                # api_response = future_api_call.result()
                                # st.write(api_response["text"])
                                # Display status updates while waiting for tasks to complete
                                with st.container():
                                    with st.spinner("Generating visulations, please wait..."):
                                        # Waiting for API response task to complete and displaying results

                                        # Conditional execution of chart rendering based on future_chart's state
                                        if future_chart is not None:
                                            chart = future_chart.result()
                                            if chart:
                                                st.plotly_chart(chart)

                        except Exception as e:
                            logger.error(f"Error plotting plotly chart: {e}.")
                            # st.write(df.plot())
                        # for text in ["Is this response helpful?", "Do you like this text?"]:
                response = st_text_rater(text="Is this response helpful?")
                message = {"role": "assistant", "content": text_write, "df": df, "chart": chart, "response": response}
                st.session_state.messages.append(message)

    # if __name__ == "__main__":
    #     main()
with tab2:
    # Main content area for data visualization
    if selected_db_path:
        # st.subheader(f"Data from {selected_db_filename}")
        df = get_df_from_database(selected_db_path)
        pyg_html = get_pyg_html(df)
        components.html(pyg_html, width=1300, height=1000, scrolling=True)
    else:
        st.warning("Please select a database from the sidebar before proceeding.")


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

        numeric_data = df.select_dtypes(include=[np.number])  # Select only numeric columns
        correlation_matrix = numeric_data.corr()
        report_parts.append(f"correlation_matrix:\n{correlation_matrix}\n")
        # Compile the Data Quality Report
        data_quality_report = "Data Quality Report:\n" + "\n".join(report_parts)
    except Exception as e:
        data_quality_report = f"An error occurred while generating the data quality report: {e}"
        print(data_quality_report)
        data_quality_report = ""
    return data_quality_report


import pandas as pd
import hashlib


def hash_dataframe(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def get_llm_insights(df, selected_db_path, cache_path="llm_insights_cache.txt"):
    """
    Generate insights based on the DataFrame using an LLM (Ollama in this case).
    This function sends a detailed prompt based on df to the LLM, caches the result in a local file,
    and returns the LLM's response as a string formatted for Streamlit.
    """
    content =None
    cache_path = selected_db_path + "_" + cache_path
    # Check for existing cache
    if os.path.exists(cache_path):
        with open(cache_path, "r") as file:
            content = file.read()

    if not content:
        summary_stats = df.describe(include="all").to_string()

        missing_values = (df.isna().sum() / df.shape[0]) * 100
        missing_percentage = missing_values.round(2).sort_values(ascending=False)

        # Assume generate_data_quality_report is defined elsewhere
        data_quality_report = generate_data_quality_report(df)

        content = f"""
        Given a dataset with {df.shape[0]} rows and {df.shape[1]} columns, covering the following areas: {', '.join(df.columns.tolist())}.

        **Statistical Summary of the Data:**
        {summary_stats}

        **Missing Percentage:**
        {missing_percentage}

        **Data Quality Report:**
        {data_quality_report}
        """

    # Assuming 'query_mistral' sends the query to the LLM and returns a string response
    # response = query_mistral(q + prompt_content)

    # Cache the response
    with open(cache_path, "w") as file:
        file.write(content)

    prompt_content = f"""
        Act as a Business ANALYST and create a REPORT for Stakeholders.

        1. **Patterns and Correlations:**
        2. **Anomalies and Outliers:**
        3. **Predictive Insights:**
        4. **Recommendations for plot KPI's:**

        using DATA:
        {content}
        """
    return prompt_content


with tab3:
    with st.spinner("Generating Insights..."):
        prompt_summary = get_llm_insights(df, selected_db_path)
        summary_1 = st.empty()
        # with xy:
        with summary_1:
            text_write = query_mistral(
                prompt_summary,
                callbacks=[StreamlitCallbackHandler(summary_1, expand_new_thoughts=True)],
                max_tokens=1024,
            )["text"]
            st.write(text_write)

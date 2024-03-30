import streamlit as st
from llama_api import query_llama_api

# Mock functions (for complete examples, implement the logic)
import re  # Import regular expression module for parsing

import sqlite3

import pandas as pd
import sqlite3

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")


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


import sqlite3

from functools import lru_cache


@lru_cache(maxsize=None)
def generate_sql_schema_context(db_path):
    """
    Generates a textual SQL schema description from an SQLite database, including
    the top 5 rows of data from each table and additional statistical context
    like minimum, maximum, and average values for numerical columns.

    Parameters:
    - db_path (str): The path to the SQLite database file.

    Returns:
    - str: A formatted string describing the database schema and sample data.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            if not tables:
                return "No tables found in the database."

            schema_descriptions = []

            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                column_descriptions = []
                for column in columns:
                    column_name = column[1]
                    data_type = column[2]
                    if data_type in ["INTEGER", "REAL"]:  # Add more numeric types as needed
                        # Fetch statistical data for numerical columns
                        cursor.execute(
                            f"SELECT MIN({column_name}), MAX({column_name}), AVG({column_name}) FROM {table_name}"
                        )
                        min_val, max_val, avg_val = cursor.fetchone()
                        column_descriptions.append(
                            f"'{column_name}' ({data_type}, min: {min_val}, max: {max_val}, avg: {avg_val:.2f})"
                        )
                    else:
                        column_descriptions.append(f"'{column_name}' ({data_type})")

                table_description = f"Table '{table_name}': columns {', '.join(column_descriptions)}."
                schema_descriptions.append(table_description)

                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()

                if rows:
                    rows_descriptions = ["Top 5 rows:"]
                    for row in rows:
                        row_description = ", ".join([str(cell) for cell in row])
                        rows_descriptions.append(f"({row_description})")
                    schema_descriptions.extend(rows_descriptions)
                else:
                    schema_descriptions.append("No data available.")

            schema_context = "\n".join(schema_descriptions)

            logger.info("-" * 100)
            logger.info(schema_context)
            logger.info("-" * 100)

            return schema_context
    except sqlite3.Error as e:
        return f"An error occurred: {e}"


def generate_sql_query(question, db_path):
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
    # Refine the prompt for more clarity and specificity
    refined_prompt = f"""
    Given a specific question that requires analysis, generate an SQL query that adheres to the following guidelines:
    - Directly answers the question: '{question}'
    - Use schema and details for better context and relevance.
    - Limits the result to 100 rows to ensure efficiency
    - Selects only the necessary columns for answering the question, avoiding unnecessary data
    - Utilizes SQL features such as constraints, aggregations, or conditions to optimize query performance and relevance
    The query should be concise and focused, enclosed within backticks for clarity. No explanatory text is required; please provide only the SQL query.

    Enclose query in backticks.

    SCHEMA: {schema}
    """
    # Query the Llama API
    response = query_llama_api(refined_prompt)

    # Attempt to extract the SQL query enclosed in backticks from the response
    match = re.search(r"`{1,3}\s*(SELECT.*?);?\s*`{1,3}", response, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()  # Stripping whitespace for clean extraction
        logger.info(sql_query)
        return sql_query  # Extract the SQL query
    else:
        return "No SQL query generated or query not enclosed in backticks."


import plotly.express as px


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
            line for line in code_block.split("\n") if "fig.show()" not in line and "pd.read_csv(" not in line
        ]
        # Join the lines back into a single string
        return "\n".join(filtered_lines)
    else:
        # If no code block is found, return an empty string or handle as needed
        return ""


def is_code_safe(code):
    # Implement safety checks on the generated code here
    return "import os" not in code and "open(" not in code


def generate_plotly_chart(df, question, attempt=1, max_attempts=3, last_error=""):
    if attempt > max_attempts:
        logger.info("Maximum attempts reached. Unable to generate executable code.")
        return None
    # Details about the DataFrame 'df', including column names and types, and a brief on what we're trying to visualize.
    df_info = f"DataFrame 'df' contains columns: {list(df.columns)}. DataFrame shape: {df.shape}. data types: {df.dtypes} infer data context using head: {df.head(5)}"
    question_info = f"QUESTION TO PLOT: {question}"

    # Adjust the prompt to be more specific about the expected output and guidelines for code generation.
    prompt = f"""
    Using the pandas DataFrame structure provided: `{df_info}` use this context to create a Python code snippet with Plotly Express to generate a figure object `fig` visualizing the data as per the specific question: `{question_info}`. The visualization should be straightforward and interactive, focusing directly on the dataset's insights.

    The code must:
    - Be executable in a standard Python environment.
    - Follow data visualization best practices, ensuring the x and y axes are accurately labeled to reflect the dataset's content.
    - Maintain simplicity in the graph's design, avoiding overcomplication.
    - Use Plotly Express for an interactive visualization experience, as shown in the reference code structure:
        ```
        import plotly.express as px
        fig = px.some_chart_type(df, x='column_x', y='column_y')
        ```
    Adapt this structure to fit the visualization requirements mentioned, ensuring all instructions are contained within a single triple backtick code block for immediate usability.
    """

    if last_error:
        prompt += f"\nPlease correct the code based on this feedback: {last_error}"

    resp = query_llama_api(prompt)
    python_code = extract_python_code(resp)
    logger.info(f"Attempt {attempt}: Generated code:\n{python_code}")

    if not is_code_safe(python_code):
        logger.info("Generated code failed safety checks. Trying again...")
        return generate_plotly_chart(df, attempt + 1, max_attempts, "Code failed safety checks.")

    local_namespace = {"df": df, "pd": pd, "px": px}

    try:
        exec(python_code, local_namespace)
        fig = local_namespace.get("fig", None)
        if fig is None:
            raise ValueError("Figure object 'fig' was not created.")
    except Exception as e:
        logger.info(f"Error executing generated code: {e}. Trying again...")
        return generate_plotly_chart(df, attempt + 1, max_attempts, f"Error: {e}")

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


from concurrent.futures import ThreadPoolExecutor


# Streamlit app
def main():
    st.title("Real Estate Analysis Chatbot")

    db_path = "falcon/property_data_1.db"
    user_question = st.text_input("Ask me a question about property data:")
    logger.info(user_question)
    if user_question:
        sql_query = generate_sql_query(user_question, db_path)
        df = query_db_to_dataframe(db_path, sql_query)
        # st.write("Here are the top 10 most expensive properties based on your query:")
        AgGrid(df)
        # st.plotly_chart(df.plot(), use_container_width=True)
        # st.pyplot(df.plot())
        if len(df) > 100:
            dfx = df.head(100)
        else:
            dfx = df
        df_json = dfx.to_json(orient="records", lines=True)

        prompt = f"""
        I have compiled the sales data into the following structured format, where each line represents a record in JSON format:

        {df_json}

        Using this data, could you summarize, as a data analyst, in natural language, the answer to the following question: {user_question}

        Respond in markdown format.
        """

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_api_call = executor.submit(query_llama_api, prompt)
            future_chart = executor.submit(generate_plotly_chart, df, user_question)

            # Waiting for tasks to complete and capturing results
            api_response = future_api_call.result()
            chart = future_chart.result()

        # Display results in Streamlit
        st.markdown(api_response)
        st.plotly_chart(chart)


db_path = "falcon/property_data_1.db"


if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = []

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# if st.session_state.messages[-1]["role"] != "assistant":
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        sql_query = generate_sql_query(prompt, db_path)
        df = query_db_to_dataframe(db_path, sql_query)
        # st.write("Here are the top 10 most expensive properties based on your query:")

        if not df.empty:
            if len(df) > 10:
                dfx = df.head(10)
            else:
                dfx = df
            df_json = dfx.to_json(orient="records", lines=True)

            prompt = f"""
            I have compiled the sales data into the following structured format, where each line represents a record in JSON format:

            {df_json}

            Using this data, could you summarize, as a data analyst, in natural language, the answer to the following question: {prompt}

            Respond in markdown format.
            """

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_api_call = executor.submit(query_llama_api, prompt)
                future_chart = executor.submit(generate_plotly_chart, df, prompt)

                # Waiting for tasks to complete and capturing results
                api_response = future_api_call.result()
                st.markdown(api_response)

                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_side_bar()
                gridoptions = gb.build()

                # Mostar AgGrid
                AgGrid(df, height=400, gridOptions=gridoptions)
                chart = future_chart.result()

            # Display results in Streamlit

            try:
                st.plotly_chart(chart)
            except Exception as e:
                logger.error(f"Error generating plotly chart: {e}.")
                st.write(df.plot())

        # message = {"role": "assistant", "content": df}
        # st.session_state.messages.append(message)

# if __name__ == "__main__":
#     main()


# echo "https://akuttamath:glpat-G9scD1xR4wyn_3iQFxq4@gitlab.com/sorcero/ai/retrieval-augmented-generation.git" >> ~/.git-credentials


# https://gitlab.com/sorcero/ai/retrieval-augmented-generation.git

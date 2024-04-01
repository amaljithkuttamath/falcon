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

# from st_aggrid import AgGrid, GridOptionsBuilder
# from code_editor import code_editor
# from streamlit_extras.app_logo import add_logo

# add_logo("http://placekitten.com/120/120")

st.set_page_config(
    page_title="Falcon: Talk to your data",
    page_icon="falcon/src/eagle_1f985.gif",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# This is a very cool app!",
    },
)

# logo = "falcon/src/eagle_1f985.gif"  # Change this to the path of your logo file

# # If you want to add some text next to the logo
# col1, col2 = st.columns([1, 4])  # Adjust the ratio based on your preference

# with col1:
#     st.image(logo, width=100)  # Adjust the width as needed

# with col2:
#     st.markdown("# Falcon: Talk to your data")  # Adjust the title as needed


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
    Generates a textual SQL schema description from an SQLite database, including a list of tables,
    a list of columns for each table, the top 5 rows of data from each table, additional statistical context
    like minimum, maximum, and average values for numerical columns, a list of up to 10 unique values for all columns
    (indicating if there are more), and counts of unique values.

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

            schema_list = [f"- {table[0]}" for table in tables]
            schema_list_str = "Tables in the database:\n" + "\n".join(schema_list) + "\n\n"

            schema_descriptions = [schema_list_str]

            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()

                columns_list = [column[1] for column in columns_info]
                columns_list_str = "`, `".join([f"`{column}`" for column in columns_list])
                table_header = f"Table '{table_name}' - Columns: {columns_list_str}"
                schema_descriptions.append(table_header)

                column_descriptions = []
                for column in columns_info:
                    column_name = column[1]
                    data_type = column[2]

                    # Fetch unique count for all columns
                    cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}")
                    unique_count = cursor.fetchone()[0]

                    # Fetch up to 10 unique values for all columns
                    cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT 10")
                    unique_values = cursor.fetchall()
                    unique_values_str = ", ".join([str(val[0]) for val in unique_values])

                    if data_type in ["INTEGER", "REAL"]:
                        # Fetch statistical data for numerical columns
                        cursor.execute(
                            f"SELECT MIN({column_name}), MAX({column_name}), AVG({column_name}) FROM {table_name}"
                        )
                        min_val, max_val, avg_val = cursor.fetchone()
                        more_indicator = ", more..." if unique_count > 10 else ""
                        column_descriptions.append(
                            f"'{column_name}' ({data_type}, min: {min_val}, max: {max_val}, avg: {avg_val:.2f}, unique: {unique_count}, sample values: [{unique_values_str}{more_indicator}])"
                        )
                    else:
                        more_indicator = ", more..." if unique_count > 10 else ""
                        column_descriptions.append(
                            f"'{column_name}' ({data_type}, unique: {unique_count}, sample values: [{unique_values_str}{more_indicator}])"
                        )

                columns_description = "Columns details: " + ", ".join(column_descriptions)
                schema_descriptions.append(columns_description)

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

                schema_descriptions.append("")  # For better separation in the output

            schema_context = "SCHEMA DESCRIPTION:\n" + "\n".join(schema_descriptions)

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
    # refined_prompt = f"""
    # Given a specific question that requires analysis, generate an SQLite3 query that adheres to the following guidelines:
    # - Use schema and details for better context and relevance.
    # - Directly answers the question: '{question}'
    # - Limits the result to 100 rows to ensure efficiency
    # - Write simple queries
    # - Selects only the necessary columns for answering the question, avoiding unnecessary data
    # - Utilizes SQL features such as constraints, aggregations, or conditions to optimize query performance and relevance
    # The query should be concise and focused, enclosed within backticks for clarity. No explanatory text is required; please provide only the SQL query.

    # Enclose SQL query in backticks and separate it from any explanatory text.

    # SCHEMA: {schema}
    # """
    refined_prompt = f"""<s>[INST] Given `SCHEMA DESCRIPTION` and  `SCHEMA`: {schema}\n  Task: You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, provided database `SCHEMA`, aimed at answering a specified question. This query should adhere to principles of clarity, efficiency, and relevance, avoiding unnecessary complexity and focusing on extracting the essential data needed for the answer. If the question or schema details are unclear or if pertinent information is missing, prioritize addressing these gaps over formulating an imprecise response. The final query should be concise, optimized for performance, and directly applicable to the question at hand. Responses that do not align with these guidelines or that speculate beyond the available information are not acceptable. <</SYS>>

    [INST] Your specific instructions are:
    0. USE columns that are available in the `SCHEMA`.
    1. Construct an `SQLite3` query that directly answers the question: "{question}" with context from `SCHEMA DESCRIPTION`.
    2. Focus on selecting only the columns necessary to answer the question, also include relavent columns.
    3. Write simple queries. Aggregate data if required.
    4. Query small amount of data to ensure efficiency.
    5. The queried data would be used for further analysis and plotting.
    6. Dont perform complex queries.

    The query must be presented within backticks (``` ```). [/INST]\n """
    # Query the Llama API
    response = query_llama_api(refined_prompt)

    # Attempt to extract the SQL query enclosed in backticks from the response
    match = re.search(r"```(?:sql)?\s*\n([\s\S]*?)\n```", response, re.DOTALL)

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
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
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


from concurrent.futures import ThreadPoolExecutor

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import os

# Streamlit app


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


def run_request(question_to_ask):
    print(question_to_ask)

    # Hugging Face model
    llm = HuggingFaceHub(
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        repo_id="codellama/" + "CodeLlama-34b-Instruct-hf",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 500},
    )
    llm_prompt = PromptTemplate.from_template(question_to_ask)
    print(llm_prompt)
    llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
    llm_response = llm_chain.predict()
    # rejig the response
    llm_response = format_response(llm_response)
    return llm_response


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
    primer_desc += " The script should only include code, no comments.\n"

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
    try:
        fig, ax = execute_plot_code(answer)
        return fig
    except Exception as e:
        logger.error(e)
        return None


db_path = "falcon/property_data_1.db"


if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = []

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
# if st.session_state.messages[-1]["role"] != "assistant":
if prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with st.status("Querying data", expanded=False) as status:
                sql_query = generate_sql_query(prompt, db_path)
                st.code(sql_query, language="sql")
                # code_d = code_editor(sql_query, lang="sql")
                # logger.info(code_d)
                # st.code(sql_query, language="sql")
                df = query_db_to_dataframe(db_path, sql_query)
                status.update(label="Download complete!", state="complete", expanded=False)
                # st.write("Here are the top 10 most expensive properties based on your query:")
            api_response = ""

            if not df.empty:
                if len(df) > 50:
                    dfx = df.head(50)
                else:
                    dfx = df
                df_json = dfx.to_json(orient="records", lines=True)
                descriptive_stats = df.describe()
                prompt_summary = f"""
                I have compiled the top 50 points of data into the following structured format:

                {df_json}

                Descriptive statistics:
                {descriptive_stats}

                Using this data, could you summarize, as a data analyst, in natural language, the answer to the following question: {prompt}

                Respond in markdown format.
                """

                with ThreadPoolExecutor(max_workers=2) as executor:

                    future_api_call = executor.submit(query_llama_api, prompt_summary)
                    future_chart = executor.submit(generate_plotly, df, prompt)

                    # Waiting for tasks to complete and capturing results
                    # with st.status("Response Summary", expanded=True) as status:
                    api_response = future_api_call.result()
                    st.markdown(api_response)
                    st.dataframe(df)
                    # gb = GridOptionsBuilder.from_dataframe(df)
                    # gb.configure_side_bar()
                    # gridoptions = gb.build()

                    # # Mostar AgGrid
                    # AgGrid(df, height=200, gridOptions=gridoptions)
                    chart = future_chart.result()

                # Display results in Streamlit

                try:
                    if chart:
                        st.plotly_chart(chart)
                except Exception as e:
                    logger.error(f"Error plotting plotly chart: {e}.")
                    st.write(df.plot())

            message = {"role": "assistant", "content": api_response}
            st.session_state.messages.append(message)

# if __name__ == "__main__":
#     main()

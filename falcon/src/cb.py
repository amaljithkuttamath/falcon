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
from llama_api import query_llama_api
from loguru import logger


st.set_page_config(
    page_title="Falcon: Talk to your data",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# This is a very cool app!",
    },
)

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


@lru_cache(maxsize=None)
def generate_sql_schema_context(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                return "No tables found in the database."

            schema_descriptions = []

            for table in tables:
                table_name = table[0]
                schema_descriptions.append(f"Table: {table_name}\n")

                # Since PRAGMA table_info does not support parameter substitution, ensure table_name is safe.
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()

                for column_info in columns_info:
                    col_name, col_type = column_info[1], column_info[2]
                    schema_descriptions.append(f"Column: {col_name}, Type: {col_type}")

                    if col_type in ["INTEGER", "REAL"]:
                        cursor.execute(
                            f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table_name}"
                        )
                        min_val, max_val, avg_val = cursor.fetchone()
                        avg_val_str = f"{avg_val:.2f}" if avg_val is not None else "N/A"
                        schema_descriptions.append(
                            f"Stats for {col_name} - Min: {min_val}, Max: {max_val}, Avg: {avg_val_str}"
                        )

                    cursor.execute(
                        f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10"
                    )
                    unique_vals = cursor.fetchall()
                    unique_vals_str = ", ".join(str(val[0]) for val in unique_vals) + (
                        "..." if len(unique_vals) == 10 else ""
                    )
                    cursor.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name}"
                    )
                    unique_count = cursor.fetchone()[0]
                    schema_descriptions.append(
                        f"Unique values in {col_name}: {unique_count} (Sample: {unique_vals_str})"
                    )

                schema_descriptions.append("")
            return "\n".join(schema_descriptions)
    except sqlite3.Error as e:
        raise e
        return f"An error occurred: {e}"

def better_query(question, db_path):
    schema = generate_sql_schema_context(db_path)
    refined_prompt =f""" Question: ```{question}``` Enhance the users query with the context: `{schema}`
    Respond only with the modified or enhanced question in under 100 words."""
    response = query_llama_api(refined_prompt)
    print(response)



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
    question_m = better_query(question, db_path)
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

    #     1. **Use Schema Columns**: Only use columns from the available `Column_Name`.
    # 2. **Answer the QUESTION**: Your query should directly answer the specified QUESTION.
    # 3. **Ready for Analysis**: The data you query will be used for further analysis and plotting, ensure you inlcude all necessary columns.
    # 4. **Avoid Complex Queries**: Do not craft overly complex queries.
    # 5. **Small Data Queries**: Ensure your query pulls a small amount of data [less than 100 rows] for efficiency.
    # Sample values are just for you to have context.
    # Dont perform unnecessary aggregations.

    # """
    ## Works
    # refined_prompt = f"""[INST] Task: You're an expert in `SQLite3` queries. Your objective is to create a precise `SQLite3` query, based on the provided database `SCHEMA`, to effectively address a given question.[/INST]

    # [INST] Instructions:

    # -Use `SQLite3` dialect for query generation.
    # - Use schema and details for better context and relevance.
    # -Direct Answer: Ensure your query directly addresses the QUESTION.
    # -Analysis Ready: Include all necessary columns for analysis.
    # -Keep it Simple: Craft straightforward queries for clarity.
    # -Efficient Retrieval: Retrieve fewer than 100 rows.

    # GIVEN: {schema}

    # QUESTION: ```{question}```

    # Please present your `SQLite3` QUERY within backticks (``` ```) to clearly indicate it as a code snippet. [/INST]
    # """
    logger.info(schema)
    refined_prompt = f"""[INST] Task: You're a data analyst and an expert in `SQLite3` queries. 
    Your objective is to create a `SQLite3` query, based on the provided `SCHEMA`, 
    to effectively query related data to a given question.[/INST]

    [INST] Instructions:
    - Use `SQLite3` dialect for query generation.
 
    SCHEMA: ```{schema}```

    QUESTION: ```{question_m}```

    Please present your `SQLite3` QUERY within backticks (``` ```) to clearly indicate it as a code snippet. [/INST]
    """

    if previous_code and error_reason:
        refined_prompt += f"\n\nPrevious Attempt:\n```sql\n{previous_code}\n```\nError Reason: {error_reason}. Please correct the query based on this feedback."

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
            line
            for line in code_block.split("\n")
            if "fig.show()" not in line and "pd.read_" not in line
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
                primer_desc
                + "\nThe column '"
                + i
                + "' is type "
                + str(df.dtypes[i])
                + " and contains numeric values. "
            )
    primer_desc = (
        primer_desc
        + "PERFORM filtering and aggregating on the dataframe. If required by the `QUESTION TO PLOT:`.\n"
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
        return generate_plotly_chart(
            df, question, attempt + 1, max_attempts, "Code failed safety checks."
        )

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

    error_feedback = (
        f"\n\nFeedback from last attempt: {last_error}" if last_error else ""
    )
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
        return generate_matplotlib_seaborn_chart(
            df, attempt + 1, max_attempts, "Code failed safety checks."
        )

    local_namespace = {"df": df, "pd": pd, "plt": plt, "sns": sns}

    try:
        exec(python_code, local_namespace)
        fig = local_namespace.get("plt", None)
        if fig is None:
            raise ValueError("Matplotlib figure 'plt' was not properly configured.")
    except Exception as e:
        logger.info(f"Error executing generated code: {e}. Trying again...")
        return generate_matplotlib_seaborn_chart(
            df, attempt + 1, max_attempts, f"Error: {e}"
        )

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


def get_table_names(db_path):
    """Retrieve a list of all table names in the specified SQLite database."""
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    df_tables = query_db_to_dataframe(db_path, query)
    return df_tables["name"].tolist()


def preprocess_column_names(df):
    """Preprocess DataFrame column names: lowercase, remove special characters, replace spaces with underscores."""
    df.columns = [
        re.sub(r"\W+", "", column).lower().replace(" ", "_") for column in df.columns
    ]
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
            df.to_sql(table_name, conn, if_exists='replace', index=False)

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
            return self.predefined_db_paths[[os.path.basename(path) for path in self.predefined_db_paths].index(filename)]
        else:
            return os.path.join(self.upload_directory, filename)

    def handle_csv_upload(self, uploaded_file):
        """Process an uploaded CSV file, converting it to a SQLite database."""
        if uploaded_file is not None:
            db_path = os.path.join(self.upload_directory, uploaded_file.name.replace(".csv", ".db"))
            # Assuming csv_to_sqlite is a predefined function
            csv_to_sqlite(uploaded_file, db_path)
            st.sidebar.success(f"Database created successfully at {db_path}")

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

    if prompt := st.chat_input(
        "Your question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # if st.session_state.messages[-1]["role"] != "assistant":

    if prompt:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                logger.info(selected_db_path)
                # with st.status("Querying data", expanded=False) as status:
                #     sql_query = generate_sql_query(prompt, db_path)
                #     st.code(sql_query, language="sql")
                #     # code_d = code_editor(sql_query, lang="sql")
                #     # logger.info(code_d)
                #     # st.code(sql_query, language="sql")
                #     df = query_db_to_dataframe(db_path, sql_query)
                #     status.update(label="Download complete!", state="complete", expanded=False)
                success = False
                with st.status("Querying data", expanded=False) as status:
                    # Initialize variables for attempts and success flag.
                    max_attempts = 3
                    attempt = 0

                    # The initial SQL query is generated based on the user prompt and database path.
                    sql_query = generate_sql_query(prompt, selected_db_path)

                    while attempt < max_attempts and not success:
                        try:
                            # Display the SQL query in the Streamlit app for each attempt.
                            st.code(sql_query, language="sql")

                            # Attempt to query the database and store results in a DataFrame.
                            df = query_db_to_dataframe(selected_db_path, sql_query)
                            success = (
                                True  # Update success flag if query is successful.
                            )
                            status.update(
                                label="Download complete!",
                                state="complete",
                                expanded=False,
                            )
                        except Exception as e:
                            attempt += 1  # Increment attempt counter.
                            logger.error(
                                f"Attempt {attempt} with query failed: {e}"
                            )  # Log the error.

                            if attempt < max_attempts:
                                sql_query = generate_sql_query(
                                    prompt,
                                    previous_code=sql_query,
                                    error_reason=f"Attempt {attempt} with query failed: {e}",
                                    db_path=selected_db_path,
                                )
                                st.warning("Trying again with a different query...")
                                try:
                                    # Attempt the query with the new SQL query.
                                    st.code(sql_query, language="sql")
                                    df = query_db_to_dataframe(
                                        selected_db_path, sql_query
                                    )
                                    success = True  # Update success flag if new query is successful.
                                    status.update(
                                        label="Download complete with alternative query!",
                                        state="complete",
                                        expanded=False,
                                    )
                                except Exception as e:
                                    # Log the error if the new query also fails.
                                    logger.error(
                                        f"Alternative query attempt failed: {e}"
                                    )
                                    break

                    # st.write("Here are the top 10 most expensive properties based on your query:")
                api_response = ""
                if success:
                    if not df.empty:
                        # Limit the data to the top 50 records for analysis if the DataFrame has more than 50 rows.
                        dfx = df.head(50) if len(df) > 50 else df

                        # Convert the limited DataFrame to a JSON string for structured data representation.
                        df_json = dfx.to_json(orient="records", lines=True)

                        # Generate descriptive statistics for the entire DataFrame to summarize the data.
                        descriptive_stats = df.describe().to_string()

                        # Prepare the data summary and descriptive statistics for presentation.
                        data_summary = f"""
                        The query resulted in a selection of the top 50 data points, which are structured as follows: {df_json}.
                        Here are the descriptive statistics of the entire dataset:
                        {descriptive_stats}
                        """

                    else:
                        data_summary = "No data found."

                    # Format the prompt to summarize the query results in a professional tone, excluding raw data tables.
                    prompt_summary = f"""
                    Based on the data obtained from the query "{sql_query}", here is a summary:

                    {data_summary}

                    Please provide a natural language summary answering the following question: {prompt}
                    This summary is prepared in the tone of a professional data analyst, focusing on key insights and findings without delving into raw data details.
                    Use only data that is relevant to the question. and dont make up any irrelevant information.
                    """

                    prompt_summary = prompt_summary.strip()
                    future_chart = None

                    try:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            # Submitting task for API call
                            future_api_call = executor.submit(
                                query_llama_api, prompt_summary
                            )

                            # Submitting task for chart generation if DataFrame is not empty
                            if not df.empty:
                                future_chart = executor.submit(
                                    generate_plotly, df, prompt
                                )

                            # Display status updates while waiting for tasks to complete
                            with st.container():
                                with st.spinner("Generating summary, please wait..."):
                                    # Waiting for API response task to complete and displaying results
                                    api_response = future_api_call.result()
                                    st.markdown(api_response)
                                    st.dataframe(df)

                                    # Conditional execution of chart rendering based on future_chart's state
                                    if future_chart is not None:
                                        chart = future_chart.result()
                                        if chart:
                                            st.plotly_chart(chart)

                    except Exception as e:
                        logger.error(f"Error plotting plotly chart: {e}.")
                        st.write(df.plot())

                message = {"role": "assistant", "content": api_response}
                st.session_state.messages.append(message)

# if __name__ == "__main__":
#     main()

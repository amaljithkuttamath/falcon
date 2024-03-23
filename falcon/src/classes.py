import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


def run_request(question_to_ask, model_type, key, alt_key):
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo":
        # Run OpenAI ChatCompletion API
        task = "Generate Python Code Script."
        if model_type == "gpt-4":
            # Ensure GPT-4 does not include additional comments
            task = task + " The script should only include code, no comments."
        openai.api_key = key
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {"role": "system", "content": task},
                {"role": "user", "content": question_to_ask},
            ],
        )
        llm_response = response["choices"][0]["message"]["content"]
    elif model_type == "text-davinci-003" or model_type == "gpt-3.5-turbo-instruct":
        # Run OpenAI Completion API
        openai.api_key = key
        response = openai.Completion.create(
            engine=model_type,
            prompt=question_to_ask,
            temperature=0,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["plt.show()"],
        )
        llm_response = response["choices"][0]["text"]
    else:
        # Hugging Face model
        llm = HuggingFaceHub(
            huggingfacehub_api_token=alt_key,
            repo_id="codellama/" + model_type,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 500},
        )
        llm_prompt = PromptTemplate.from_template(question_to_ask)
        llm_chain = LLMChain(llm=llm, prompt=llm_prompt)
        llm_response = llm_chain.predict()
    # rejig the response
    llm_response = format_response(llm_response)
    return llm_response


def extract_code(text):
    """
    Extract and rewrite the provided code block from text, removing the initial keyword 'python' if present.
    Assumes code blocks are wrapped in triple backticks.
    """
    # Splitting by triple backticks to find code blocks
    code_blocks = text.split("```")[1::2]  # Extracting only the code blocks
    rewritten_blocks = []

    # Iterate over each code block to process
    for block in code_blocks:
        # Splitting by newlines and removing 'python' if it's the first word
        lines = block.split("\n")
        if lines[0].strip().lower() == "python":
            lines = lines[1:]  # Remove the first line if it's just 'python'

        # Re-joining the lines and appending to the results
        rewritten_blocks.append("\n".join(lines).strip())

    return rewritten_blocks[0]


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
    return res


def format_question(primer_desc, primer_code, question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        # Code llama tends to misuse the "c" argument when creating scatter plots
        instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."
    primer_desc = primer_desc.format(instructions)
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return '"""\n' + primer_desc + question + '\n"""\n' + primer_code


def get_primer(df_dataset, df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = (
        "Use a dataframe called df from data_file.csv with columns '"
        + "','".join(str(x) for x in df_dataset.columns)
        + "'. "
    )
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i] == "O":
            primer_desc = (
                primer_desc
                + "\nThe column '"
                + i
                + "' has categorical values '"
                + "','".join(str(x) for x in df_dataset[i].drop_duplicates())
                + "'. "
            )
        elif df_dataset.dtypes[i] == "int64" or df_dataset.dtypes[i] == "float64":
            primer_desc = (
                primer_desc
                + "\nThe column '"
                + i
                + "' is type "
                + str(df_dataset.dtypes[i])
                + " and contains numeric values. "
            )
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    # primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "{}"  # Space for additional instructions if needed
    primer_desc = (
        primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    )
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc, pimer_code


# def get_primer(df, df_name):
#     """
#     Generate a primer for plotting given a DataFrame and its name.

#     Parameters:
#     - df: Pandas DataFrame to be analyzed.
#     - df_name: String name of the DataFrame variable.

#     Returns:
#     - A tuple of (primer_description, primer_code) for guiding the plotting process.
#     """

#     primer_desc = (
#         f"Use a DataFrame called 'df' from '{df_name}.csv' with columns: '"
#         + "', '".join(str(x) for x in df.columns)
#         + "'."
#     )

#     # Loop through each column to add specific details based on the data type and unique values
#     for column in df.columns:
#         unique_values = df[column].drop_duplicates()
#         num_unique_values = len(unique_values)

#         # Handle categorical data with fewer than 20 unique values
#         if num_unique_values < 20 and df.dtypes[column] == "O":
#             primer_desc += f"\nThe column '{column}' is categorical with {num_unique_values} unique values. "

#         # For numerical data, mention it's a numerical column without specifying the number of unique values
#         elif df.dtypes[column] in ["int64", "float64"]:
#             primer_desc += f"\nThe column '{column}' is numerical, representing quantitative data."

#         # For columns with more than 20 unique values, provide a general statement
#         else:
#             primer_desc += f"\nThe column '{column}' has a wide range of data with {num_unique_values} unique values."

#     # Append a note about labeling axes appropriately to the description
#     primer_desc += "\nLabel the x and y axes appropriately."

#     # Introduction to the Python script for plotting
#     primer_desc += "\nUsing Python, create a script using the DataFrame 'df' to graph the following insights:"

#     # Starter code for data visualization
#     primer_code = (
#         "import pandas as pd\n"
#         "import matplotlib.pyplot as plt\n\n"
#         "# Copy the DataFrame for manipulation\n"
#         f"df = {df_name}.copy()\n\n"
#         "# Set up the figure and axis for a plot\n"
#         "fig, ax = plt.subplots(figsize=(10, 6))\n"
#         "# Remove the top and right spines to clean up the plot appearance\n"
#         "ax.spines['top'].set_visible(False)\n"
#         "ax.spines['right'].set_visible(False)\n\n"
#         "# Your plotting code goes here...\n\n"
#         "# Example of labeling the plot\n"
#         "plt.xlabel('Your X-axis Label')\n"
#         "plt.ylabel('Your Y-axis Label')\n"
#         "plt.title('Your Plot Title')\n"
#         "plt.show()"
#     )

#     return primer_desc, primer_code

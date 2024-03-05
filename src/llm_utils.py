import os
import re
# import openai

from dotenv import load_dotenv, find_dotenv
import streamlit as st

from  langchain_community.chat_models import ChatOllama

from langchain.agents import AgentType
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema.output_parser import OutputParserException

import ollama


# if os.environ.get('OPENAI_API_KEY') is not None:
#     openai.api_key = os.environ['OPENAI_API_KEY']
# else:
#     _ = load_dotenv(find_dotenv())  # read local .env file
#     openai.api_key = os.environ['OPENAI_API_KEY']


def chat_api(
    messages, model= "gemma:2b", temperature=0.0, max_tokens=256, top_p=0.5
):
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
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        messages.append({
            "role": "assistant",
            "content": code_prompt
        })

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
        code = extract_python_code(
            response["message"]["content"])
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"```{code}")
            exec(code)
        print(response)
    return response["message"]


def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]


def chat_with_data_api(df, model= "gemma:2b", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    A function that answers data questions from a dataframe.
    """
    error_msg = ""
    if "plot" in st.session_state.messages[-1]["content"].lower():
        code_prompt = """
        Given a dataframe with specific columns, generate Plotly code to plot the data according to the user's request. 
        Use Plotly for creating the visualization and ensure the code is compatible for execution in a Python environment.
        The plotting code should be returned in the following format: ```python <code>```.
        Note: Replace "<code>" with the actual Python code.
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": code_prompt
        })
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=st.session_state.messages,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     top_p=top_p,
        # )
        response = ollama.chat(
        model=model,
        messages=st.session_state.messages,
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
    )
        print(response)
        code = extract_python_code(response["message"]["content"])
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat."
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
            return "Couldn't plot the data"
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"```{code}")
            print(code)
            exec(code)
            return response["message"]["content"]
    else:
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
        )

        try:
            answer = pandas_df_agent(st.session_state.messages)
            if answer["intermediate_steps"]:
                action = answer["intermediate_steps"][-1][0].tool_input["query"]
                st.write(f"Executed the code ```{action}```")
            return answer["output"]
        except OutputParserException:
            error_msg = """OutputParserException error occured in LangChain agent.
                Refine your query."""
            return error_msg
        except Exception as e:  # noqa: E722
            print(e)
            answer = "Unknown error occured in LangChain agent. Refine your query"
            return error_msg

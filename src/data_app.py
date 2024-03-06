import streamlit as st
from streamlit_chat import message
import pandas as pd
from main import get_text, sidebar
from llm_utils import chat_with_data_api, get_llm_insights


def chat_with_data():
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "generated" not in st.session_state:
        st.session_state.generated = []

    if "past" not in st.session_state:
        st.session_state.past = []

    st.title("Falcon:")

    with st.sidebar:
        model_params = sidebar()
        memory_window = st.slider(
            "Memory Window",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            help="""The size of history chats that is kept for context. A value of, say,
                    3, keeps the last three pairs of prompts and responses, i.e., the last
                    6 messages in the history.""",
        )

    uploaded_file = st.file_uploader("Choose file", type=["csv"])
    df = None  # Initialize df
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(float)

        if (
            "data_analysis_performed" not in st.session_state
            or not st.session_state.data_analysis_performed
        ):
            perform_basic_analysis(df, model_params)
            st.session_state.data_analysis_performed = True  # Set the flag
        prompt = """You are a Python expert. Given a pandas DataFrame loaded from the uploaded file,
                    manipulate it or generate visualizations based on the user's queries. Assume all necessary
                    libraries are imported and use 'df' as the DataFrame variable."""
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": prompt}]
    else:
        df = None
        st.session_state.data_analysis_performed = (
            False  # Reset the flag when no file is uploaded
        )

    user_input = get_text()

    if user_input and df is not None:
        process_user_input(df, user_input, model_params)


def perform_basic_analysis(df, model_params):
    """Perform and display basic analysis of the uploaded DataFrame in a new tab, along with LLM insights."""
    tab1, tab2 = st.tabs(["Chat", "Data Analysis"])

    with tab2:
        st.write("Basic Data Analysis")
        with st.spinner("Generating LLM Insights..."):
            # llm_insights = get_llm_insights(df, **model_params)
            # st.write("LLM Insights:")
            # st.write(llm_insights, unsafe_allow_html=True)

            st.write("DataFrame Shape:", df.shape)
            st.write("Data Types:", df.dtypes)
            st.write("Summary Statistics:")
            st.write(df.describe())
            st.write("Missing Values Count:")
            st.write(df.isnull().sum())


def process_user_input(df, user_input, model_params):
    """Process user input and display response in the chat tab."""
    tab1, tab2 = st.tabs(["Chat", "Data Analysis"])
    with tab1:
        if df.empty:
            st.warning("Dataframe is empty, upload a valid file", icon="⚠️")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display the loading spinner while processing the query
            with st.spinner("Processing... Please wait"):
                response = chat_with_data_api(df, **model_params)

            if response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                # Update the chat history for display
                st.session_state.generated.append(response)
                st.session_state.past.append(user_input)

        # Display the chat history
        if st.session_state.generated:
            for i in range(len(st.session_state.generated) - 1, -1, -1):
                message(st.session_state.generated[i], key=str(i))
                if i - 1 >= 0:
                    message(
                        st.session_state.past[i - 1], is_user=True, key=str(i) + "_user"
                    )


if __name__ == "__main__":
    chat_with_data()

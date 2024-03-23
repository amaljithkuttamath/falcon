from langchain_community.chat_models.openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="tgi",
    openai_api_key="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",
    openai_api_base="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1" + "/v1/",
)
# print(llm.invoke("Why is open-source software important?"))

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

agent = create_csv_agent(
    llm,
    "falcon/20231223-GRID_INCIDENTS.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.invoke("how many rows are there?")

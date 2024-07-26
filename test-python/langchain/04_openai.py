from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from my_config import set_environment
import os

# set_environment()

print(os.environ["OPENAI_API_BASE"])

tools = load_tools(tool_names=["python_repl"])
llm = OpenAI(model="gpt-3.5-turbo-instruct")

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("whats 4 + 4")
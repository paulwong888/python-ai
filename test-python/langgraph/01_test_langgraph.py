from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from typing import TypedDict, Union, Annotated
from langchain_core.messages import BaseMessage
from langchain.schema.agent import AgentAction, AgentFinish

from my_config import set_environment

set_environment()

prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt)

llm = ChatOpenAI(
    streaming=True
)

#Tool
search = TavilySearchResults(max_results=1)
tools = [search]

#AGENT
agent = create_openai_functions_agent(llm, tools, prompt)
# result = agent.invoke({"input" : "what is the weather in Taiwan?", "intermediate_steps": []})
# print(result)

agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke(input={"input" : "what is the weather in Taiwan?"})
print(result)

# TODO: Define the graph state/agent state
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[]
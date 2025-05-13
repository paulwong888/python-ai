from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from my_config import set_environment

set_environment()

prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt = hub.pull("cpatrickalves/react-chat-agent")
print(prompt)

llm = ChatOpenAI(
    # model="gpt-3.5-turbo"
)

#Tool
search = TavilySearchResults(max_results=1)
# tools = [search]

@tool
def search_tool(query: str):
    """get real-time data from internet"""
    
    search = TavilySearchResults(max_results=1)
    result = search.invoke(query)
    print(result)
    return result[0]["content"]

tools = [search_tool]
# open_ai_tools = [convert_to_openai_tool(f) for f in tools]

# result = search.invoke("目前市场上苹果手机15的平均售价是多少？")
# print(result[0]["content"])

#AGENT
agent = create_openai_functions_agent(llm, tools, prompt)
# result = agent.invoke({"input" : "what is the weather in Taiwan?", "intermediate_steps": []})
# print(result)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke(input={"input" : "what is the weather in Taiwan?"})
# result = agent_executor.invoke(input={"input" : "目前市场上苹果手机15的平均售价是多少？"})
print(result)

# for step in agent_executor.stream({"input" : "what is the weather in Taiwan?"}):
#     print(step)
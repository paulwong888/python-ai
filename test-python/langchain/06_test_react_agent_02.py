from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
import datetime
from dotenv import load_dotenv

load_dotenv()

prompt_template = hub.pull("aixgeek/react4chinese")

llm = ChatOpenAI()

@tool
def check_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """获取当前时间"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = load_tools(tool_names=["wikipedia"])
tools.append(check_time)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    llm=llm,
    agent=agent,
    tools=tools,
    # max_iterations=3,
    verbose=True
)

# result = agent_executor.invoke({"input":"现在几点？"})
result = agent_executor.invoke({"input":"香港现任特首是谁？"})
print(result)
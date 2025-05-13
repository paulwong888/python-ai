from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI()

researcher = Agent(
    llm=llm,
    role="市场搜索官",
    goal="搜索关于美食的信息",
    backstory="你是一个经验丰富的市场调研助理",
    allow_delegation=True,
    verbose=True
)

analyst = Agent(
    llm=llm,
    role="市场分析员",
    goal="分析市面上哪些美食最受欢迎，用中文回答",
    backstory="你是一个专业的市场分析师",
    allow_delegation=False,
    verbose=True
)

task1 = Task(
    description="分析最新市场上最受欢迎的美食",
    agent=researcher,
    expected_output=""
)

task2 = Task(
    description="写一份完美的市场分析，分析哪些美食最受欢迎。",
    agent=analyst,
    expected_output=""
)

crew = Crew(
    agents=[researcher, analyst],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print(result)
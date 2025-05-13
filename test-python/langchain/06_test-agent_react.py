from langchain_community.llms import FakeListLLM
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentType, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

import os
from my_config import set_environment

set_environment()

print(os.environ["OPENAI_API_BASE"])

llm = OpenAI(
    temperature=0, 
    # model="gpt-3.5-turbo"
)

tools = load_tools(tool_names=["wikipedia", "llm-math"], llm=llm)

template='''
Answer the following questions as best you can. 
Try to answer the question by yourself.
If not necessary, don't use the tools.
You have access to the following tools:{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Previous Conversation:{chat_history}
Question: {input}
Thought:{agent_scratchpad}
'''


prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(
    tools=tools, llm=llm, prompt=prompt
    #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    #verbose=True
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory = memory, max_iterations = 3,handle_parsing_errors=True,verbose= True)

question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""

agent_executor.invoke({"input":question})

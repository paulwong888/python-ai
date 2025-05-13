from langchain_community.llms import FakeListLLM
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()


llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125")

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

conversation_buffer_window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

agent = initialize_agent(
    tools=tools, llm=llm, prompt=prompt,
    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory = conversation_buffer_window_memory,
    verbose=True,
)

# agent_executor = AgentExecutor(agent=agent, tools=tools, memory = memory, max_iterations = 3,handle_parsing_errors=True,verbose= True)

question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""

print(agent.agent.llm_chain.prompt.template)

agent.invoke({"input":question})

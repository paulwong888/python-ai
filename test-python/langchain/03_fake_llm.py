from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools, initialize_agent, AgentType

tools = load_tools(tool_names=["python_repl"])

responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 5"]
llm = FakeListLLM(responses=responses)

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("whats 2 + 2")
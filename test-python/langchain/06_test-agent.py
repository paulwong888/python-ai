from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4", temperature=0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

zero_shot_agent = initialize_agent(
    tools=tools, llm=llm, zero_shot_agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    handle_parsing_errors="Check your output and make sure it conforms! Do not output an action and a final answer at the same time.",
    verbose=True
)

question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""

print(zero_shot_agent.agent.llm_chain.prompt.template)

zero_shot_agent.run(question)
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from agents import AnalysisAgents
from tasks import AnalysisTasks

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

class FinancialCrew:

    def __init__(self, company: str) -> None:
        self.company = company

    def run(self):

        agents = AnalysisAgents(llm)
        tasks = AnalysisTasks()

        mr_analyst = agents.market_research_analyst()
        cfa = agents.cfa()

        research_task = tasks.research_task(mr_analyst, self.company)
        analysis_task = tasks.analysis_task(cfa)


        crew = Crew(
            agents=[mr_analyst, cfa],
            tasks=[research_task, analysis_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("\n\n## 欢迎来到投资顾问团队")
    print("--------------------------")
    company = input("请输入你想分析的公司名称\n")

    finacial_crew = FinancialCrew(company)
    result = finacial_crew.run()
    print("\n########################")
    print("以下是分析结果")
    print("########################\n")
    print(result)

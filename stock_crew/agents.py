from crewai import Agent
from tools.search import SearchTools

class AnalysisAgents:

    def __init__(self, llm) -> None:
        self.llm = llm

    def market_research_analyst(self):
        return Agent(
            llm=self.llm,
            role="市场研究分析员",
            goal="""搜索公司的市场和财务状况，并按照找到的信息整理总结出
                 公司各方面的表现和财务状况
                 """,
            backstory="""最富经验的市场研究分析师，善于捕捉和发掘公司内在的
                 真相。请用中文思考和行动，并用中文回覆客户问题或与其他同事交流
                 """,
            tools=[SearchTools.search_info],
            allow_delegation=True,
            verbose=True
        )

    def cfa(self):
        return Agent(
            llm=self.llm,
            role="特许财务分析师",
            goal="""根据市场研究分析师搜索到的资料，重新整理并总结出公司的
                 状况，并且提供该公司的股份是否值得买入
                 """,
            backstory="""最富经验的投资者，善于透过公司细微的变化，捕捉公司
                 股份走向，现在你面对一生中最中的客户。请用中文思考和行动，并
                 用中文回答客户问题或与其他同事交流
                """,
            allow_delegation=False,
            verbose=True
        )
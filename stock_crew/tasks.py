from crewai import Task
from textwrap import dedent

class AnalysisTasks():
    
    def research_task(self, agent, company):
        return Task(
            description = dedent(
                f"""
                 搜索并总结最新的{company}公司动态和新闻。
                 特别关注重大事件。
                 """
            ),
            agent=agent,
            expected_output = "用列表的形式总结头5项最重要的公司新闻。"
        )
    
    def analysis_task(self, agent):
        return Task(
            description = dedent(
                """
                 将搜索到的信息进行分析，并且总结，最终得出是否应该买入
                 该公司股票的建议。
                """
            ),
            agent=agent,
            expected_output = """用报告的形式总结该公司的市场走向，
                最终得出是否应该买入该公司股票的建议。"""
        )
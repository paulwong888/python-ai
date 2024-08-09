import requests
import json
from langchain.tools import tool

class SearchTools:

    @tool
    def search_info(query: str):
        """在网上搜索关于指定内容的相关信息"""

        return SearchTools.search(query)
    
    def search(query: str):

        url = "https://google.serper.dev/news"

        payload = json.dumps({
            "q": query,
            "hl": "zh-tw"
        })
        headers = {
            'X-API-KEY': '31b039a12ceb6123a6f7992293217f562a64540c',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        news = response.json()["news"]

        string = []

        for the_news in news:
            try:
                string.append("\n".join([
                    f"标题： {the_news["title"]}",
                    f"时间： {the_news["date"]}",
                    f"来源： {the_news["source"]}",
                    f"内容摘要： {the_news["snippet"]}",
                ]))
            except KeyError:
                next

        content = "\n".join(string)
        return f"\n搜索结果是: {content}"
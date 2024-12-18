from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

import os

class Search():
    def __init__(self):
        os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    def tool():
        search = GoogleSearchAPIWrapper(k=1)    
        def top5_results(query):
            return search.results(query, 5)
        tool = Tool(
            name="google_search_snippets",
            description="Search Google for recent results.",
            func=top5_results,
        )
        #res = tool.invoke("Cold site:webmd.com")
        return tool

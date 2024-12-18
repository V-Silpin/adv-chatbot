from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

import os
#import search as srh
#import warnings

from dotenv import load_dotenv

class Query():
    def __init__(self):
        load_dotenv()
        os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        self.retrieval_chain = None
        self.index_name = os.getenv("INDEX_NAME")
        self.gemini_api_key=os.getenv("GEMINI_API_KEY")
        self.tracing_v2=os.getenv("LANGCHAIN_TRACING_V2")
        self.langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.prompt=ChatPromptTemplate.from_template("""
                    You are a medical diagnostic assistant. 
                    Your role is to understand the user's symptoms and provide the most accurate possible diagnosis. 
                    You will receive context to help you formulate better responses. 
                    If you cannot find the answer within the provided context, you will perform a Google search. 
                    The system will offer you a Google search tool, and you should share only the website links from the search results with the user. 
                    The way to share the website links is given below
                    <format>
                        <response>
                        ===Responses===
                        </response>
                        -
                        <sites>
                        ===Sites===
                        </sites>
                    </format>                                 
                    You will be rewarded generously for delivering high-quality answers.
                    <context>
                    {context}
                    </context>
                    Question: {input}
            """)
    def retrival_chain(self):
        search = GoogleSearchAPIWrapper(k=1)    
        def top5_results(query):
            return search.results(query, 5)
        tool = Tool(
            name="google_search_snippets",
            description="Search Google for recent results.",
            func=top5_results,
        )
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.gemini_api_key).bind_tools(tool)
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.gemini_api_key, model="models/embedding-001")
        pc = Pinecone(api_key = self.pinecone_key)
        index = pc.Index(self.index_name)
        bm25_encoder=BM25Encoder().default()
        retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
        #summary_memory = ConversationSummaryBufferMemory(llm=llm, k=2)
        document_chain = create_stuff_documents_chain(llm, self.prompt)
        #output_parser = StrOutputParser()
        self.retrieval_chain = create_retrieval_chain(
                            retriever,
                            document_chain,
                        )
    def response(self, query):
        response = self.retrieval_chain.invoke({"input": query})
        if isinstance(response, dict):
            return response #.get('answer', 'No answer found.')
        else:
            return 'Unexpected response format.'
    
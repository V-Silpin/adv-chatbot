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
#import warnings

import os
from dotenv import load_dotenv

class Query():
    def __init__(self):
        load_dotenv()
        self.retrieval_chain = None
        self.index_name = os.getenv("INDEX_NAME")
        self.gemini_api_key=os.getenv("GEMINI_API_KEY")
        self.tracing_v2=os.getenv("LANGCHAIN_TRACING_V2")
        self.langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.prompt=ChatPromptTemplate.from_messages(
            [
                (
                "system","""
                 You are a medical diagnostic assistant. 
                 Your role is to understand the user's symptoms and provide the most accurate possible diagnosis. 
                 You will receive context to help you formulate better responses. 
                 If you cannot find the answer within the provided context, you will perform a Google search. 
                 The system will offer you a Google search tool, and you should share only the website links from the search results with the user. 
                 You will be rewarded generously for delivering high-quality answers.
                 """
                ),
                ("user","Question:{question}\nContext:{context}")
            ]
        )
    def retrival_chain(self):
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.gemini_api_key)
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.gemini_api_key, model="models/embedding-001")
        pc = Pinecone(api_key = self.pinecone_key)
        index = pc.Index(self.index_name)
        bm25_encoder=BM25Encoder().default()
        retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
        summary_memory = ConversationSummaryBufferMemory(llm=llm, k=2)
        document_chain = create_stuff_documents_chain(llm, self.prompt)
        output_parser = StrOutputParser()
        self.retrieval_chain = create_retrieval_chain(
                            document_chain=document_chain,
                            retriever=retriever,
                            memory=summary_memory,
                            output_parser=output_parser,
                            verbose=True
                        )
    def response(self, query):
        response = self.retrieval_chain.invoke({"input":query})
        return response
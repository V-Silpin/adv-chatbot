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
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
index_name = os.getenv("INDEX_NAME")
gemini_api_key=os.getenv("GEMINI_API_KEY")
tracing_v2=os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

search = GoogleSearchAPIWrapper(k=1)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)

def top5_results(query):
    return search.results(query, 5)

def response(query):
    prompt=ChatPromptTemplate.from_template("""
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
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key, model="models/embedding-001")
    pc = Pinecone(api_key = pinecone_key)
    index = pc.Index(index_name)
    bm25_encoder=BM25Encoder().default()
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
                        retriever,
                        document_chain,
    )
    response = retrieval_chain.invoke({"input": query})
    return response

search_tool = Tool(
    name="google_search_snippets",
    description="Search Google for recent results.",
    func=top5_results,
)

rag_tool = Tool(
    name="pineconedb_search_tool",
    description="A tool to do hybrid search in pinecone db",
    func=response,
)

tools = [search_tool, rag_tool]
agent_executor = create_react_agent(llm, tools)
response = agent_executor.invoke({"messages": [HumanMessage(content="Wassup dude, can u search about AI agents?")]})

print(response["messages"])
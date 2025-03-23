from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.openai_tools import OpenAIAssistantWebSearchTool

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def run_market_summary_agent(state, settings=None):
    try:
        query = "latest global financial market news from the past week"
        user_msg = state.get("messages", [])[-1].content
        if "summary of" in user_msg.lower():
            query = user_msg.lower().split("summary of")[-1].strip()

        # OpenAI web search
        openai_tool = OpenAIAssistantWebSearchTool()
        openai_results = openai_tool.invoke(query)
        openai_docs = openai_results if isinstance(openai_results, str) else str(openai_results)

        # Tavily + Wikipedia
        tavily = TavilySearchResults(max_results=2)
        tavily_docs = tavily.invoke(query)
        wiki_loader = WikipediaLoader(query=query, load_max_docs=2)
        wiki_docs = wiki_loader.load()

        combined = ""
        for d in tavily_docs:
            combined += f"{d['content']}\n---\n"
        for doc in wiki_docs:
            combined += f"{doc.page_content}\n---\n"
        combined += f"\nOpenAI Web Search:\n{openai_docs}"

        # Summarize
        prompt = ChatPromptTemplate.from_template(
            "Summarize this market information in a concise, readable weekly briefing:\n\n{context}"
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"context": combined})

        return {"messages": [AIMessage(content=f"📈 Weekly Market Summary:\n\n{summary.strip()}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error in market summary agent: {str(e)}")]}

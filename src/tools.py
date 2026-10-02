from typing import List
from tavily import TavilyClient
from langchain_core.documents import Document

from config import config
from telemetry import logger

def execute_web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    Execute a web search using the Tavily API and converts the clean results
    into standard LangChain Document objects.
    
    If the API key is missing or an error occurs, it safely falls back
    to an informational Document rather than crashing the graph.
    """
    # 1. Guard check for missing API key
    if not config.tavily_api_key or config.tavily_api_key.strip() == "":
        logger.warning("Tavily API keyis missing or blank. Web search cannot execute.")
        return [
            Document(

                page_content=(
                    f"External Web Search was triggered for query: '{query}', but no Tavily_API_Key "
                    "was configured in the environment. Please add a valid key to .env ."
                ),
                metadata={"source":"system_warning", "title": "Tavily Key Missing"}
            )
        ]

    try:
        logger.info(f"Initiating Tavily web search for query: '{query}'")

        # 2. Initializing Tavily Client
        client = TavilyClient(api_key=config.tavily_api_key)

        # 3. Execute the Search
        # search_depth="basic" is faster and uses fewer credits than "advanced"
        search_response= client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )

        results = search_response.get("results", [])
        logger.info(f"Tavily returned {len(results)} clean search results.")

        # 4. Standardize results into LangChain Document format
        documents: List[Document] = []
        for index, item in enumerate(results):
            content = item.get("content", "")
            url = item.get("url", "unknown_source")
            title = item.get("title", f"Web Result {index + 1}")

            doc = Document(
                page_content = content,
                metadata = {
                    "source": url,
                    "title": title,
                    "chunk_id": f"web_search {index + 1}"
                }
            )
            documents.append(doc)

        # 5. Handle empty search return
        if not documents:
            logger.warning(f"Tavily return 0 results for query: '{query}'")
            return [
                Document(
                    page_content=f"No external search results could be retrieved for: {query}",
                    metadata={"source": "web_search_empty", "title": "No Results"}
                )
            ]

        return documents

    except Exception as exc:
        logger.error(f"Tavily search execution failed : {str(exc)}", exc_info=True)
        # Fail gracefully by providing content to the LLM instead of halting the system
        return [
            Document(
                page_content=(
                    f"An error occured while executing web search for '{query}': {str(exc)}"
                ),
                metadata={"source":"web_search_error","title":"Search Failure"}
    
            )
        ]
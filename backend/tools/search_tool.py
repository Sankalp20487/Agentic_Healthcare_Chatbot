"""
Web search tool - fallback for out-of-scope queries.
Extracted from original assistant code, delegates to WebSearchHandler service.
"""

import logging
from web_search_handler import WebSearchHandler

logger = logging.getLogger("healthcare_assistant")


def create_search_tool(search_handler: WebSearchHandler) -> callable:
    """
    Create tool for web search.
    
    Args:
        search_handler: WebSearchHandler instance
        
    Returns:
        Tool function
    """
    
    def web_search(query: str) -> str:
        """Search the web for information."""
        logger.info(f"Searching web for: {query[:50]}...")
        
        try:
            results = search_handler.search(query)
            formatted_results = search_handler.format_results(results)
            logger.info("Web search complete")
            return formatted_results
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error performing web search: {str(e)}"
    
    return web_search
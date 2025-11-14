"""
Response formatting tool - converts SQL results to natural language.
Extracted from original assistant code.
"""

import logging

logger = logging.getLogger("healthcare_assistant")


def create_response_tool(
    response_formatter,
    sql_cache_getter: callable
) -> callable:
    """
    Create tool for response formatting.
    
    Args:
        response_formatter: LLMChain for response generation
        sql_cache_getter: Function to get cached SQL results
        
    Returns:
        Tool function
    """
    
    def format_response(query: str) -> str:
        """Format query results into user-friendly response."""
        logger.info(f"Formatting response for: {query[:50]}...")
        
        try:
            # Check if we have results to format
            last_results = sql_cache_getter('results')
            last_sql = sql_cache_getter('sql')
            last_query = sql_cache_getter('query')
            
            if last_results is None:
                return "No query results available to format."
            
            # Use the LLM for formatting
            try:
                formatted_response = response_formatter.invoke({
                    "question": query or last_query,
                    "sql_query": last_sql or "Unknown SQL query",
                    "db_result": str(last_results)
                })
                
                logger.info("Response formatted successfully by LLM")
                return formatted_response.get("text", "")
            except Exception as llm_error:
                logger.warning(f"Error using LLM for formatting: {llm_error}")
                
                # Basic fallback formatting
                if not last_results:
                    return "No results were found for your query."
                    
                # Handle single value results
                if len(last_results) == 1 and isinstance(last_results[0], tuple) and len(last_results[0]) == 1:
                    value = last_results[0][0]
                    is_cost = any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"])
                    
                    if isinstance(value, (int, float)):
                        return f"The {'value is ${:,.2f}' if is_cost else 'result is {:,}'}.".format(value)
                
                # Handle multi-row results
                elif last_results:
                    # For single column results
                    if isinstance(last_results[0], tuple) and len(last_results[0]) == 1:
                        items = [str(row[0]) for row in last_results[:10]]
                        return f"Results: {', '.join(items)}"
                    
                    # For name-value pairs (common in distribution queries)
                    elif isinstance(last_results[0], tuple) and len(last_results[0]) == 2:
                        is_cost = any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"])
                        
                        response = "Results:\n\n"
                        for i, (name, value) in enumerate(last_results[:10], 1):
                            if isinstance(value, (int, float)):
                                value_str = "${:,.2f}".format(value) if is_cost else "{:,}".format(value)
                                response += f"{i}. {name}: {value_str}\n"
                            else:
                                response += f"{i}. {name}: {value}\n"
                                
                        return response
                
                # Generic fallback
                return f"Query results: {str(last_results)}"
                
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return f"Error formatting response: {str(e)}"
    
    return format_response
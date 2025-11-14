"""
Plot generation tool - creates visualizations from SQL results.
Extracted from original assistant code, delegates to GeneratingPlots service.
"""

import os
import logging
from gen_color_plot import GeneratingPlots

logger = logging.getLogger("healthcare_assistant")


def create_plot_tool(
    plot_generator: GeneratingPlots,
    sql_cache_getter: callable,
    title_generator: callable
) -> callable:
    """
    Create tool for plot generation.
    
    Args:
        plot_generator: GeneratingPlots instance
        sql_cache_getter: Function to get cached SQL results
        title_generator: Function to generate plot titles
        
    Returns:
        Tool function
    """
    
    def generate_plot(query: str) -> str:
        """Generate plot from cached SQL results."""
        logger.info(f"Generating plot for: {query[:50]}...")
        
        try:
            # Get the last SQL results from cache
            last_results = sql_cache_getter('results')
            last_sql = sql_cache_getter('sql')
            last_query = sql_cache_getter('query')
            
            if not last_results:
                logger.warning("No SQL results available to plot")
                return "No SQL results available to plot. Please run a query first."
            
            logger.debug(f"Data for plotting: {last_results[:5]} (showing first 5 rows)")
            
            # Determine plot type from query
            plot_type = "bar"  # Default
            if any(term in query.lower() for term in ["pie", "distribution", "breakdown"]):
                plot_type = "pie"
            elif any(term in query.lower() for term in ["bar", "top", "most"]):
                plot_type = "bar"
                
            # Generate a clean title
            clean_title = title_generator(query, plot_type)
            plot_query = f"{plot_type} chart of {query}"
            
            # Try primary plot type
            plot_result = plot_generator.process_query_result(
                {"result": f"Data from query: {last_sql}", "title": clean_title}, 
                plot_query, 
                last_results
            )
            
            # If first attempt fails, try alternative plot type
            if not plot_result or plot_result.startswith("Unable to extract data"):
                alt_plot_type = "bar" if plot_type == "pie" else "pie"
                alt_plot_query = f"{alt_plot_type} chart of {last_query}"
                logger.debug(f"First attempt failed. Trying alternative plot type: {alt_plot_query}")
                
                plot_result = plot_generator.process_query_result(
                    {"result": f"Data from query: {last_sql}", "title": clean_title}, 
                    alt_plot_query, 
                    last_results
                )
            
            if plot_result and not plot_result.startswith("Unable to extract data"):
                logger.info("Plot generated successfully")
                filename = os.path.basename(plot_result)
                return f"Plot generated successfully: /plots/{filename}"
            
            logger.warning("Plot generation failed")
            return "Unable to generate a meaningful plot from this data. The data might not be suitable for visualization."
            
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            return f"Error generating plot: {str(e)}"
    
    return generate_plot
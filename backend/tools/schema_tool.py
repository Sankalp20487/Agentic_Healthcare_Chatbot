"""
Schema information retrieval tool.
Provides database schema context for SQL generation.
"""

import re
from langchain_community.utilities.sql_database import SQLDatabase
import logging

logger = logging.getLogger("healthcare_assistant")


def create_schema_tool(db: SQLDatabase) -> callable:
    """
    Create tool function for retrieving database schema information.
    
    Args:
        db: SQLDatabase instance
        
    Returns:
        Tool function that retrieves schema info
    """
    def get_schema_info(query: str) -> str:
        """
        Get database schema information.
        
        Args:
            query: User query (context for logging)
            
        Returns:
            Database schema as string
        """
        logger.info(f"Getting schema info for: {query[:50]}...")
        try:
            full_schema = db.get_table_info()
            # Remove the example rows section
            schema_only = re.sub(r'/\*.*?\*/', '', full_schema, flags=re.DOTALL)
            logger.info("Retrieved schema info successfully")
            return schema_only.strip()
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return f"Error retrieving schema info: {str(e)}"
    
    return get_schema_info
"""
SQL execution tool - generates SQL from natural language and executes against database.
Extracted from original assistant code.
"""

import re
from typing import Any, List, Tuple
import logging

logger = logging.getLogger("healthcare_assistant")


def create_sql_execution_tool(
    db,
    retriever,
    query_generator,
    chat_history_getter: callable,
    sql_extractor: callable
) -> callable:
    """
    Create tool for SQL generation and execution.
    
    Args:
        db: SQLDatabase instance
        retriever: Pinecone vector store retriever
        query_generator: LLMChain for SQL generation
        chat_history_getter: Function to get formatted chat history
        sql_extractor: Function to extract SQL from LLM output
        
    Returns:
        Tool function
    """
    
    # Store results cache (will be accessed by other tools)
    cache = {
        'query': None,
        'sql': None,
        'results': None
    }
    
    def generate_and_run_sql(query: str) -> List[Tuple[Any, ...]]:
        """Generate SQL from natural language and execute against database."""
        logger.info(f"Generating SQL for: {query[:80]}...")
        
        try:
            # Get formatted chat history
            formatted_history = chat_history_getter()
            
            # Get schema info
            full_schema_info = db.get_table_info()
            schema_only = re.sub(r'/\*.*?\*/', '', full_schema_info, flags=re.DOTALL).strip()
            
            # Get metadata from retriever
            logger.debug("Retrieving metadata for SQL generation...")
            context_docs = retriever.invoke(query)
            metadata_text = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
            
            # Try broader search if needed
            if not metadata_text or len(metadata_text) < 100:
                broader_terms = []
                
                # Map keywords to search terms
                keyword_mapping = {
                    "age": ["age", "age_group"],
                    "serious": ["severity", "apr_severity_of_illness_description"],
                    "severe": ["severity", "apr_severity_of_illness_description"],
                    "diagnosis": ["diagnosis", "ccsr_diagnosis_description"],
                    "disease": ["diagnosis", "ccsr_diagnosis_description"],
                    "hospital": ["hospital", "facility_name"]
                }
                
                # Add relevant broader terms
                for keyword, terms in keyword_mapping.items():
                    if keyword in query.lower():
                        broader_terms.extend(terms)
                
                if broader_terms:
                    logger.debug(f"Using broader terms for metadata: {', '.join(broader_terms)}")
                    for term in broader_terms:
                        additional_docs = retriever.invoke(term)
                        additional_text = "\n".join([doc.page_content for doc in additional_docs if doc.page_content])
                        metadata_text += "\n" + additional_text
            
            logger.info(f"Found metadata ({len(metadata_text)} chars)")
            
            # Combine schema and metadata
            combined_info = f"""
            DATABASE SCHEMA:
            {schema_only}
            
            COLUMN METADATA:
            {metadata_text}
            """
            
            logger.debug("Using non-agentic approach for SQL generation...")
            
            try:
                contextualized_query = query_generator.invoke({
                    "chat_history": formatted_history,
                    "user_question": query,
                    "table_info": combined_info
                })
                
                # Extract the SQL query
                clean_sql = sql_extractor(contextualized_query.get("text", ""))
                logger.info(f"Generated SQL: {clean_sql[:100]}...")
                
                # Fix common issues with the SQL query
                common_fixes = {
                    r'\bFROM patients\b': "FROM hospital_data",
                    r'\bseverity_of_illness\b': "apr_severity_of_illness_description"
                }
                
                for pattern, replacement in common_fixes.items():
                    if re.search(pattern, clean_sql):
                        clean_sql = re.sub(pattern, replacement, clean_sql)
                        logger.debug(f"Fixed SQL: {pattern} -> {replacement}")
            
            except Exception as gen_error:
                logger.error(f"Error in SQL generation: {gen_error}")
                raise gen_error
            
            # Execute the SQL query
            logger.debug("Executing SQL...")
            results = db.run(clean_sql)
            logger.info(f"SQL execution complete: {len(results) if results else 0} rows")
            
            # Save the results for later use (cache for other tools)
            cache['query'] = query
            cache['sql'] = clean_sql
            cache['results'] = results
            
            return results
                
        except Exception as e:
            logger.error(f"Error generating or running SQL: {e}")
            return []
    
    # Attach cache to function so other tools can access it
    generate_and_run_sql._cache = cache
    
    return generate_and_run_sql
"""
Helper utility functions for the Healthcare Chat Assistant.
Contains pure functions with no external dependencies.
"""

import re
from typing import List, Dict


def extract_sql(text: str) -> str:
    """
    Extract SQL query from LLM-generated text.
    
    Handles various formats including markdown code blocks,
    SQL prefixes, and ensures proper semicolon termination.
    
    Args:
        text: Raw text potentially containing SQL query
        
    Returns:
        Cleaned SQL query string
    """
    # Remove markdown code blocks
    text = text.replace("```sql", "").replace("```", "").replace("`", "")
    
    # Remove "SQLQuery:" prefix if present
    if "SQLQuery:" in text:
        text = text.split("SQLQuery:", 1)[1].strip()
    
    # Extract SELECT statement
    if "SELECT" in text.upper():
        match = re.search(r'(SELECT\s+.*?)(;|\Z)', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
            # Ensure semicolon termination
            if not text.strip().endswith(";"):
                text = text.strip() + ";"
    
    return text.strip()


def clean_sql_query(query: str) -> str:
    """
    Clean SQL query by removing markdown formatting.
    
    Args:
        query: SQL query potentially with markdown
        
    Returns:
        Cleaned SQL query
    """
    return query.replace("```sql", "").replace("```", "").replace("`", "").strip()


def is_general_conversation(query: str) -> bool:
    """
    Detect if query is general conversation vs data query.
    
    Args:
        query: User's input text
        
    Returns:
        True if general conversation, False if data query
    """
    conversation_patterns = [
        r'^(hi|hello|hey|greetings|good (morning|afternoon|evening))',
        r'^how (are|is|do) you',
        r'^(thanks|thank you)',
        r'^(what can you do|help me|what are your capabilities)',
        r'^\?+$',
        r'^(ok|okay|got it|i see|understood)',
    ]
    return any(re.match(pattern, query.lower()) for pattern in conversation_patterns)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """
    Format chat history for prompt inclusion.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'message' keys
        
    Returns:
        Formatted string of conversation history
    """
    formatted_history = ""
    for entry in chat_history:
        formatted_history += f"{entry['role']}: {entry['message']}\n"
    return formatted_history


def generate_clean_title(query: str, plot_type: str) -> str:
    """
    Generate publication-quality titles for visualizations.
    
    Args:
        query: User's natural language query
        plot_type: Type of plot (bar, pie, line)
        
    Returns:
        Clean, descriptive title string
    """
    query_lower = query.lower()
    
    # Extract subject from query
    subject_keywords = {
        "hospital": "Hospitals",
        "diagnos": "Diagnoses", 
        "disease": "Diagnoses",
        "procedure": "Procedures",
        "age": "Age Groups",
        "region": "Regions",
        "race": "Demographics",
        "ethnic": "Demographics"
    }
    
    subject = "Data"
    for keyword, label in subject_keywords.items():
        if keyword in query_lower:
            subject = label
            break
    
    # Extract metric from query
    metric_keywords = {
        "cost": "Costs",
        "charge": "Costs",
        "expense": "Costs",
        "count": "Count",
        "stay": "Length of Stay"
    }
    
    metric = "Distribution"
    for keyword, label in metric_keywords.items():
        if keyword in query_lower:
            metric = label
            break
    
    # Extract limiters (e.g., "top 5")
    limiter = ""
    if "top" in query_lower:
        match = re.search(r"top\s+(\d+)", query_lower)
        if match:
            limiter = f"Top {match.group(1)}"
        else:
            limiter = "Top"
    
    # Construct final title
    title = f"{plot_type.capitalize()} Chart: "
    if limiter:
        title += f"{limiter} {subject} by {metric}"
    else:
        title += f"{subject} by {metric}"
    
    return title
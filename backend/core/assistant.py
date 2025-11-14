"""
Healthcare Data Agentic Assistant - Main Orchestrator

A production-ready LLM-powered assistant for querying healthcare data using natural language.
Implements RAG (Retrieval-Augmented Generation) with Pinecone vector search, SQL generation
with Google Gemini, and conversational memory with LangChain.

Architecture:
    - Single ReAct agent with 6 tools
    - RAG-based metadata retrieval via Pinecone
    - SQL generation with LLM prompt engineering
    - Conversational memory for multi-turn interactions
    - Rate limiting to prevent API quota exhaustion
"""

import os
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings

from pinecone import Pinecone
from langchain_pinecone import Pinecone as PineconeVectorStore

# Import connectors and services
from connectors.snowflake_connector import SnowflakeConnector
from web_search_handler import WebSearchHandler
from gen_color_plot import GeneratingPlots

# Import modular components
from utils.rate_limiter import TokenBucket
from utils.logging_config import setup_logging
from utils.helpers import (
    extract_sql, 
    clean_sql_query, 
    is_general_conversation, 
    format_chat_history,
    generate_clean_title
)

# Import tool factory functions
from tools.schema_tool import create_schema_tool
from tools.metadata_tool import create_metadata_tool
from tools.sql_execution_tool import create_sql_execution_tool
from tools.plot_tool import create_plot_tool
from tools.search_tool import create_search_tool
from tools.response_tool import create_response_tool

# Setup logging
logger = setup_logging(log_level="INFO")


class HealthcareDataAgenticAssistant:
    """
    Main assistant class implementing agentic workflow for healthcare data analysis.
    
    Architecture Flow:
        1. Query Understanding: Detect if data query or conversation
        2. Schema Retrieval: Get database structure
        3. Metadata Retrieval: Semantic search via Pinecone vector DB
        4. SQL Generation: LLM-powered query construction
        5. Execution: Run against Snowflake data warehouse
        6. Response Formatting: Natural language answer generation
        7. Visualization: Optional chart generation
        8. Fallback: Web search for out-of-scope queries
    
    Attributes:
        connector: Snowflake database connector
        plot_generator: Visualization service
        search_handler: Web search service
        db: SQLDatabase instance for LangChain
        retriever: Pinecone vector store retriever
        rate_limiter: Token bucket for API quota management
        llm: Google Gemini language model
        chat_history: Conversation history
        agent: LangChain ReAct agent
    """
    
    def __init__(self):
        """Initialize all components of the healthcare assistant."""
        logger.info("Initializing Healthcare Data Agentic Assistant")
        load_dotenv()
        
        try:
            # Initialize connectors and services
            logger.debug("Setting up Snowflake connector")
            self.connector = SnowflakeConnector()
            
            logger.debug("Setting up plot generator")
            self.plot_generator = GeneratingPlots()
            
            logger.debug("Setting up web search handler")
            self.search_handler = WebSearchHandler()
            
            logger.debug("Setting up SQL database interface")
            self.db = SQLDatabase.from_uri(self.connector.uri)
            
            # Setup Pinecone vector store for metadata retrieval
            logger.debug("Connecting to Pinecone vector database")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            
            # Initialize embeddings and retriever
            logger.debug("Loading HuggingFace embedding model")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=embedding_model,
                text_key="combined_text",
                namespace=None
            )
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            logger.info("Pinecone retriever initialized (k=5)")
            
            # Rate limiter: 15 API calls per minute (0.25 per second)
            self.rate_limiter = TokenBucket(tokens=15, fill_rate=0.25)
            logger.info("Rate limiter configured: 15 calls/minute")
            
            # Initialize LLM (Google Gemini)
            logger.debug("Initializing Google Gemini LLM")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
            logger.info("LLM initialized: gemini-2.0-flash")
            
            # Conversation history
            self.chat_history: List[Dict[str, str]] = []
            
            # Result cache (shared across tools)
            self._result_cache = {
                'query': None,
                'sql': None,
                'results': None
            }
            
            # Setup agent tools and prompts
            logger.debug("Setting up agent tools and prompts")
            self.setup_tools_and_prompts()
            
            logger.info("Assistant initialization complete")
            
        except Exception as e:
            logger.critical(f"Failed to initialize assistant: {str(e)}", exc_info=True)
            raise
    
    def _get_from_cache(self, key: str):
        """Get value from result cache."""
        return self._result_cache.get(key)
    
    def _update_cache(self, query: str, sql: str, results):
        """Update result cache."""
        self._result_cache['query'] = query
        self._result_cache['sql'] = sql
        self._result_cache['results'] = results
        # Also store on instance for backward compatibility
        self._last_query = query
        self._last_sql = sql
        self._last_results = results
    
    def setup_tools_and_prompts(self):
        """Configure LangChain agent with tools and prompt templates."""
        
        # Create LLMChains for SQL generation and response formatting
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        # SQL Generation Chain
        self.query_generator_prompt = PromptTemplate.from_template("""
        You are a smart data assistant with access to a healthcare database.
        
        Previous conversation:
        {chat_history}
        
        Current user question: {user_question}
        
        Database schema information:
        {table_info}
        
        Based on the conversation history and the current question, generate a SQL query that will answer the user's question.
        When generating SQL queries:
        - If filtering on text fields (e.g., diagnosis, procedure description), avoid using '='.
        - Instead, break the important words into individual terms and use ILIKE with OR logic:
            Example: For "heart surgery", use:
            ccsr_procedure_description ILIKE '%heart%' OR ccsr_procedure_description ILIKE '%surgery%'
        - Consider the conversation history for context when generating queries.
        - If the question seems to be a follow-up to a previous query, maintain context appropriately.
        - IMPORTANT: Focus ONLY on the current question. Do not be influenced by previous unrelated queries.
        - If the query mentions terms like "expensive", "cost", "charge", etc., make sure to use appropriate aggregation.
        - If the query asks for "top N" or "most expensive", include ORDER BY and LIMIT clauses.
        - Make sure to include COUNT(*) when needed for aggregation, especially for plots or distributions.
        - When asking for "top" diseases, diagnoses, etc., always include the COUNT in the SELECT clause.
        
        Return ONLY the SQL query, nothing else.
        """)
        
        self.query_generator = LLMChain(
            llm=self.llm,
            prompt=self.query_generator_prompt,
            verbose=False
        )
        
        # Response Formatting Chain
        self.response_formatter_prompt = PromptTemplate.from_template("""
        You are a helpful healthcare data assistant.
        
        Original user question: {question}
        SQL query that was executed: {sql_query}
        Raw database result: {db_result}
        
        Format this database result into a natural, helpful response for the user.
        - Use proper sentences and formatting
        - Include the actual values from the results
        - Format currency values with dollar signs and commas as appropriate
        - Provide context about what the numbers represent
        - Be concise but informative
        - If the result is about costs or charges, make it clear these are averages from the database
        
        Your response:
        """)
        
        self.response_formatter = LLMChain(
            llm=self.llm,
            prompt=self.response_formatter_prompt,
            verbose=False
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )
        
        # Create cache getter function for tools
        def get_cache(key: str):
            return self._get_from_cache(key)
        
        # Create chat history getter
        def get_chat_history() -> str:
            return format_chat_history(self.chat_history)
        
        # Create SQL execution tool with cache update callback
        sql_execution_tool_func = create_sql_execution_tool(
            db=self.db,
            retriever=self.retriever,
            query_generator=self.query_generator,
            chat_history_getter=get_chat_history,
            sql_extractor=extract_sql
        )
        
        # Wrap to update cache
        def sql_tool_with_cache(query: str):
            results = sql_execution_tool_func(query)
            # Update cache
            if hasattr(sql_execution_tool_func, '_cache'):
                cache = sql_execution_tool_func._cache
                self._update_cache(cache['query'], cache['sql'], cache['results'])
            return results
        
        # Create all tools using factory functions
        schema_tool_func = create_schema_tool(self.db)
        metadata_tool_func = create_metadata_tool(self.retriever)
        plot_tool_func = create_plot_tool(self.plot_generator, get_cache, generate_clean_title)
        search_tool_func = create_search_tool(self.search_handler)
        response_tool_func = create_response_tool(self.response_formatter, get_cache)
        
        # Define agent tools
        self.tools = [
            Tool("GetSchemaInfo", schema_tool_func, "Get database schema information"),
            Tool("RetrieveMetadata", metadata_tool_func, "Retrieve metadata about database columns"),
            Tool("GenerateAndRunSQL", sql_tool_with_cache, "Generate and execute SQL query"),
            Tool("GeneratePlot", plot_tool_func, "Generate a plot from query results"),
            Tool("WebSearch", search_tool_func, "Search the web for information"),
            Tool("FormatResponse", response_tool_func, "Format query results into a user-friendly response")
        ]
        
        logger.info(f"Created {len(self.tools)} agent tools")
        
        # Agent system prompt
        prefix = """
You are a healthcare data assistant with access to New York State Patient Discharge Database for 2022 and metadata about the database schema.
When answering questions about healthcare data, follow these steps in order:

1. Call GetSchemaInfo to understand what data is available in the database
2. Call RetrieveMetadata to get detailed information about relevant columns
3. Call GenerateAndRunSQL to query the database
4. Call FormatResponse to create a user-friendly response
5. If the user specifically asked for a visualization or the data is suitable for plotting, call GeneratePlot
6. Only if steps 1-4 fail to provide adequate information, call WebSearch as a last resort

When answering questions about what you can do or what all you can be asked, always mention:
1. You have access to the complete 2022 New York State Patient Discharge Database
2. You can analyze hospital stays, diagnoses, procedures, and costs specifically for New York hospitals in 2022
3. You can generate visualizations like bar charts and pie charts of the New York healthcare data
4. You can compare different hospitals, regions, or patient demographics within New York
5. You can provide insights on medical conditions, but your primary strength is analyzing the NY discharge data
6. You can also search the web for additional information if needed

Important guidelines:
- Focus on the current question only, don't get distracted by previous context
- If the question is about "top" or "most expensive", make sure to use ORDER BY and LIMIT
- If the question is about costs or charges, use appropriate aggregation functions
- Handle each query independently unless it's clearly a follow-up question
- For "serious" or "severe" conditions, look for metadata on severity classification

For general questions or conversation not requiring database access, respond directly.
"""

        suffix = """
Chat history:
{chat_history}

Now, answer the following new question:
{input}
"""

        # Initialize the agent
        logger.debug("Initializing ReAct agent")
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            handle_parsing_errors=True,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
                "input_variables": ["chat_history", "input"],
            },
        )
        logger.info("ReAct agent initialized successfully")
    
    def generate_response_for_general_query(self, query: str) -> str:
        """
        Handle general conversational queries without database access.
        
        Args:
            query: User's conversational input
            
        Returns:
            LLM-generated conversational response
        """
        logger.info(f"Generating response for general query: {query[:50]}...")
        
        formatted_history = format_chat_history(self.chat_history)
        
        prompt = f"""
        You are a knowledgeable healthcare data assistant with access to the 2022 New York State Patient Discharge Database. You can answer
        questions about healthcare data, procedures, diagnoses, and medical information.
        
        When asked what you can do, always mention that your primary function is to analyze the New York State 
    Patient Discharge Database from 2022, and that users can ask about hospitals, diagnoses, procedures, 
    patient demographics, costs, and request visualizations of this specific dataset.
    
        
        Previous conversation:
        {formatted_history}
        
        Human: {query}
        AI Assistant:
        """
        
        response = self.llm.predict(prompt)
        logger.debug(f"Generated conversational response: {len(response)} chars")
        return response
    
    def run(self):
        """Main terminal-based interaction loop."""
        logger.info("Starting Healthcare Data Assistant in terminal mode")
        logger.info(f"Rate limit: {self.rate_limiter.capacity} API calls per minute")
        
        print("\n=== Healthcare Data Assistant (Terminal Mode) ===")
        print("Commands:")
        print("  - 'exit' to quit")
        print("  - 'clear memory' to reset conversation")
        print("  - 'tokens' to check API rate limit")
        print("\nRate limit: 15 API calls per minute\n")

        while True:
            # Rate limiting
            wait_time = self.rate_limiter.consume(3)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, cooling down for {wait_time:.1f}s")
                print(f"\nAPI rate limit reached. Cooling down for {wait_time:.1f} seconds...")
                start_time = time.time()
                while time.time() - start_time < wait_time:
                    remaining = wait_time - (time.time() - start_time)
                    print(f"\rCooldown: {remaining:.1f}s remaining   ", end="", flush=True)
                    time.sleep(0.5)
                print("\rReady for your next query!                          ")
            
            user_query = input("You: ")
            
            # Handle commands
            if user_query.lower() in ["exit", "quit"]:
                logger.info("User initiated shutdown")
                print("Exiting. Have a great day!")
                break
            elif user_query.lower() == "clear memory":
                self.memory.clear()
                self.chat_history = []
                logger.info("Conversation memory cleared")
                print("Conversation memory has been cleared.")
                continue
            elif user_query.lower() == "tokens":
                tokens = self.rate_limiter.tokens
                cooldown = max(0, (3 - tokens) / self.rate_limiter.fill_rate)
                print(f"Available API tokens: {tokens:.1f}/{self.rate_limiter.capacity}")
                print(f"Time until next query possible: {cooldown:.1f} seconds")
                logger.debug(f"Token check - Available: {tokens:.1f}")
                continue
                
            # Add to history
            self.chat_history.append({"role": "Human", "message": user_query})
            logger.info(f"User query: {user_query[:80]}...")
                
            # Handle conversation
            if is_general_conversation(user_query):
                logger.debug("Processing as conversational query")
                print("\nProcessing conversational query...")
                try:
                    response = self.generate_response_for_general_query(user_query)
                    print(f"\nAssistant: {response}\n")
                    self.chat_history.append({"role": "AI", "message": response})
                    continue
                except Exception as e:
                    logger.error(f"Error handling general conversation: {e}", exc_info=True)
                    # Fall through to agent processing
            
            # Process through agent
            try:
                logger.debug("Processing query through agent")
                print("\nProcessing your query (this may take a moment)...")
                start_time = time.time()
                response = self.agent.run(input=user_query)
                process_time = time.time() - start_time
                
                logger.info(f"Query processed in {process_time:.1f}s")
                print(f"Query processed in {process_time:.1f} seconds")
                print(f"\nAssistant: {response}\n")
                self.chat_history.append({"role": "AI", "message": response})
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\nError processing query: {e}")
                print("Attempting web search fallback...")
                
                # Fallback to web search
                try:
                    logger.info("Attempting web search fallback")
                    results = self.search_handler.search(user_query)
                    fallback_response = self.search_handler.format_results(results)
                    print(f"\nAssistant: {fallback_response}\n")
                    self.chat_history.append({"role": "AI", "message": fallback_response})
                except Exception as fallback_e:
                    logger.error(f"Fallback search failed: {fallback_e}", exc_info=True)
                    error_message = "I'm having trouble processing your request right now. Please try again with a different question or try again later."
                    print(f"\nAssistant: {error_message}\n")
                    self.chat_history.append({"role": "AI", "message": error_message})


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Healthcare Data Assistant Starting")
    logger.info("="*60)
    
    try:
        assistant = HealthcareDataAgenticAssistant()
        assistant.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        print("\n\nExiting...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        import sys
        sys.exit(1)
    finally:
        logger.info("Healthcare Data Assistant Shutting Down")
        logger.info("="*60)
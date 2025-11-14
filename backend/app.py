"""
FastAPI server for Healthcare Chat Assistant.
Provides WebSocket and REST API endpoints for real-time chat interaction.
"""

import os
import json
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict
import re
from pathlib import Path

# Import assistant from new modular structure
from core.assistant import HealthcareDataAgenticAssistant

# Setup logging
logger = logging.getLogger("healthcare_assistant")

app = FastAPI(title="Healthcare Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create plots directory if it doesn't exist
PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Serve plot images statically
app.mount("/plots", StaticFiles(directory="plots"), name="plots")

# Initialize assistant
logger.info("Initializing Healthcare Data Assistant...")
assistant = HealthcareDataAgenticAssistant()
logger.info("Healthcare assistant initialized successfully")

# Store active connections
active_connections: Dict[str, WebSocket] = {}


def clean_plot_references(text: str) -> str:
    """
    Clean up plot file path references in response text.
    
    Args:
        text: Response text potentially containing plot paths
        
    Returns:
        Cleaned response text
    """
    # Handle the specific format showing in your interface
    text = re.sub(
        r"Here is the plot for the top 5 diagnoses for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", 
        "Here is the plot for the top 5 diagnoses for patients over 70:", 
        text
    )
    
    # More general patterns to catch /plots/ path formats
    text = re.sub(r"for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", "for patients over 70:", text)
    text = re.sub(
        r"for the top 5 diagnoses for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", 
        "for the top 5 diagnoses for patients over 70:", 
        text
    )
    
    # Handle other common patterns
    text = re.sub(r"The plot is saved as plots\\[a-zA-Z0-9_.-]+\.png\.?", "Here's the visualization:", text)
    text = re.sub(r"The plot is saved as plots/[a-zA-Z0-9_.-]+\.png\.?", "Here's the visualization:", text)
    text = re.sub(r"The plot is saved as .+\.png\.?", "Here's the visualization:", text)
    
    # Replace direct references to plot paths
    text = re.sub(
        r"Here is the bar plot for the top 5 diseases: plots\\[a-zA-Z0-9_.-]+\.png", 
        "Here is the bar plot for the top 5 diseases:", 
        text
    )
    text = re.sub(
        r"Here is the bar plot of the top 5 diseases: plots\\[a-zA-Z0-9_.-]+\.png", 
        "Here is the bar plot of the top 5 diseases:", 
        text
    )
    
    # Generic replacements
    text = re.sub(r": plots\\[a-zA-Z0-9_.-]+\.png", ":", text)
    text = re.sub(r": plots/[a-zA-Z0-9_.-]+\.png", ":", text)
    text = re.sub(r": /plots/[a-zA-Z0-9_.-]+\.png", ":", text)
    
    # Handle format: "A bar chart visualizing these diagnoses is available at /plots/plot_xyz.png."
    text = re.sub(
        r"A bar chart visualizing these diagnoses is available at /plots/plot_[a-zA-Z0-9_.-]+\.png\.", 
        "A bar chart visualizing these diagnoses is available:", 
        text
    )
    
    # Clean up any double spaces or awkward punctuation
    text = re.sub(r"\s\s+", " ", text)
    text = re.sub(r"\.\s+\.", ".", text)
    
    return text.strip()


def clean_assistant_response(text: str) -> str:
    """
    Remove standard capability statements from responses.
    
    Args:
        text: Response text potentially containing capability statement
        
    Returns:
        Cleaned response text
    """
    # Pattern to match the standard capability statement paragraph
    capability_statement = (
        r"I have access to the complete 2022 New York State Patient Discharge Database\. "
        r"I can analyze hospital stays, diagnoses, procedures, and costs specifically for "
        r"New York hospitals in 2022\. I can generate visualizations like bar charts and pie charts "
        r"of the New York healthcare data\. I can compare different hospitals, regions, or patient "
        r"demographics within New York\. I can provide insights on medical conditions, but my primary "
        r"strength is analyzing the NY discharge data\. I can also search the web for additional "
        r"information if needed\."
    )
    
    # Remove the capability statement
    cleaned_text = re.sub(capability_statement, "", text)
    
    # Clean up any double newlines that might result
    cleaned_text = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


class ConnectionManager:
    """Manages WebSocket connections for real-time chat."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: str, client_id: str):
        """Send text message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def send_plot(self, plot_path: str, client_id: str):
        """Send plot URL to specific client."""
        if client_id in self.active_connections:
            # Extract filename from path, normalize slashes
            filename = os.path.basename(plot_path.replace('\\', '/'))
            # Convert local path to URL
            plot_url = f"/plots/{filename}"
            plot_data = {
                "type": "plot",
                "url": plot_url
            }
            await self.active_connections[client_id].send_text(json.dumps(plot_data))
            logger.debug(f"Sent plot to client {client_id}: {plot_url}")


manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Healthcare Chatbot API is running"}


@app.get("/check-plot/{filename}")
async def check_plot(filename: str):
    """Check if a specific plot file exists."""
    plot_path = PLOTS_DIR / filename
    exists = plot_path.exists()
    logger.debug(f"Plot check for {filename}: {exists}")
    return {"exists": exists}


@app.get("/plot/{filename}")
async def get_plot(filename: str):
    """Serve a specific plot file."""
    plot_path = PLOTS_DIR / filename
    if not plot_path.exists():
        logger.warning(f"Plot not found: {filename}")
        raise HTTPException(status_code=404, detail="Plot not found")
    
    logger.debug(f"Serving plot: {filename}")
    return FileResponse(plot_path)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            query = message.get("message", "")
            reset = message.get("reset", False)
            
            if reset:
                # Reset the assistant's memory
                assistant.chat_history = []
                logger.info(f"Conversation reset by client {client_id}")
                await manager.send_message(json.dumps({
                    "type": "response",
                    "message": "Conversation memory has been cleared."
                }), client_id)
                continue
            
            if not query:
                await manager.send_message(json.dumps({
                    "type": "error",
                    "message": "No message provided"
                }), client_id)
                continue
            
            logger.info(f"[WebSocket] Query from {client_id}: {query[:80]}...")
            
            # Send "thinking" message
            await manager.send_message(json.dumps({
                "type": "thinking",
                "message": "Processing your query..."
            }), client_id)
            
            # Process the query
            try:
                # Add query to chat history
                assistant.chat_history.append({"role": "Human", "message": query})
                
                # Check if it's general conversation
                if assistant.is_general_conversation(query):
                    logger.debug("Processing as general conversation")
                    response = assistant.generate_response_for_general_query(query)
                    
                    # Clean the response
                    clean_response = clean_assistant_response(response)
                    
                    assistant.chat_history.append({"role": "AI", "message": clean_response})
                    
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": clean_response
                    }), client_id)
                else:
                    # Process through agent
                    logger.debug("Processing through agent")
                    response = await asyncio.to_thread(assistant.agent.run, input=query)
                    
                    # Check for visualization requests
                    visualization_requested = any(term in query.lower() for term in 
                        ["plot", "chart", "graph", "visualize", "visualization", "show me", "display", "compare"])
                    
                    # Look for plot path in the response
                    plot_path = None
                    plot_match = re.search(r"(/plots/[a-zA-Z0-9_.-]+\.png|plots[/\\][a-zA-Z0-9_.-]+\.png)", response)
                    
                    # Also check for plot references in the text
                    plot_mentioned = "plot" in response.lower() or "chart" in response.lower() or "is available at" in response.lower()
                    
                    # Clean up the response text
                    clean_response = clean_plot_references(response)
                    clean_response = clean_assistant_response(clean_response)
                    
                    # Add clean response to chat history
                    assistant.chat_history.append({"role": "AI", "message": clean_response})
                    
                    # Send the text response
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": clean_response
                    }), client_id)
                    
                    # Process plot - if we found a direct path reference, use that
                    if plot_match:
                        # Plot path directly found in the response
                        plot_path_str = plot_match.group(1)
                        
                        # Normalize the path 
                        normalized_path = plot_path_str.replace('\\', '/')
                        
                        # Get just the filename
                        filename = os.path.basename(normalized_path)
                        
                        # The file is already in the plots directory
                        plot_path = str(Path("plots") / filename)
                        
                        logger.info(f"Plot detected: {plot_path}")
                        
                        # Add a slight delay to ensure the text response is processed first
                        await asyncio.sleep(0.5)
                        await manager.send_plot(plot_path, client_id)
                    
                    # If no direct path but plot was mentioned or requested, try to generate one directly
                    elif (visualization_requested or plot_mentioned) and hasattr(assistant, '_last_results') and assistant._last_results:
                        logger.info("Generating plot via direct streaming")
                        
                        # Generate a clean title 
                        title = "Data Visualization"
                        if hasattr(assistant, 'generate_clean_title'):
                            title = assistant.generate_clean_title(query, 
                                "pie" if "pie" in query.lower() else "bar")
                        
                        # Generate and stream the plot
                        try:
                            await assistant.plot_generator.generate_and_stream_plot(
                                websocket, 
                                assistant._last_results,
                                query,
                                title
                            )
                            logger.info("Plot streaming successful")
                        except Exception as plot_err:
                            logger.error(f"Error streaming plot: {plot_err}", exc_info=True)
                    else:
                        logger.debug("No plot needed for this response")
            
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                
                # Use web search as fallback
                try:
                    logger.info("Attempting web search fallback")
                    results = assistant.search_handler.search(query)
                    fallback_response = assistant.search_handler.format_results(results)
                    
                    # Clean the fallback response
                    fallback_response = clean_assistant_response(fallback_response)
                    
                    assistant.chat_history.append({"role": "AI", "message": fallback_response})
                    
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": fallback_response
                    }), client_id)
                except Exception as fallback_e:
                    logger.error(f"Fallback search failed: {fallback_e}", exc_info=True)
                    error_message = "I'm having trouble processing your request right now. Please try again with a different question."
                    assistant.chat_history.append({"role": "AI", "message": error_message})
                    
                    await manager.send_message(json.dumps({
                        "type": "error",
                        "message": error_message
                    }), client_id)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")


@app.post("/api/chat")
async def chat(request: Request):
    """REST API endpoint for chat (alternative to WebSocket)."""
    data = await request.json()
    query = data.get("message", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="No message provided")
    
    logger.info(f"[REST API] Query received: {query[:80]}...")
    
    # Add query to chat history
    assistant.chat_history.append({"role": "Human", "message": query})
    
    try:
        # Check if it's general conversation
        if assistant.is_general_conversation(query):
            logger.debug("Processing as general conversation")
            response = assistant.generate_response_for_general_query(query)
            # Clean the response
            clean_response = clean_assistant_response(response)
        else:
            # Process through agent
            logger.debug("Processing through agent")
            response = assistant.agent.run(input=query)
            
            # Clean up the response
            clean_response = clean_plot_references(response)
            clean_response = clean_assistant_response(clean_response)
            
        # Check if there's a plot in the response
        plot_data = None
        plot_match = re.search(r"(/plots/[a-zA-Z0-9_.-]+\.png|plots[/\\][a-zA-Z0-9_.-]+\.png)", response)
        
        if plot_match:
            # Plot path directly found in the response
            plot_path_str = plot_match.group(1)
            
            # Normalize the path
            normalized_path = plot_path_str.replace('\\', '/')
            
            # Get just the filename
            filename = os.path.basename(normalized_path)
            
            # Generate plot URL
            plot_url = f"/plots/{filename}"
            plot_data = {"url": plot_url}
            
            logger.info(f"Plot detected: {plot_url}")
        
        # Add clean response to chat history
        assistant.chat_history.append({"role": "AI", "message": clean_response})
        
        return {
            "response": clean_response,
            "plot": plot_data
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        
        # Use web search as fallback
        try:
            logger.info("Attempting web search fallback")
            results = assistant.search_handler.search(query)
            fallback_response = assistant.search_handler.format_results(results)
            
            # Clean the fallback response
            fallback_response = clean_assistant_response(fallback_response)
            
            assistant.chat_history.append({"role": "AI", "message": fallback_response})
            
            return {
                "response": fallback_response
            }
        except Exception as fallback_e:
            logger.error(f"Fallback search failed: {fallback_e}", exc_info=True)
            error_message = "I'm having trouble processing your request right now. Please try again with a different question."
            assistant.chat_history.append({"role": "AI", "message": error_message})
            
            return JSONResponse(
                status_code=500,
                content={"error": error_message}
            )


@app.get("/api/history")
async def get_history():
    """Get the conversation history."""
    logger.debug("Retrieving conversation history")
    return {"history": assistant.chat_history}


@app.post("/api/reset")
async def reset_conversation():
    """Reset the conversation history."""
    assistant.chat_history = []
    logger.info("Conversation history reset")
    return {"message": "Conversation memory has been cleared."}


if __name__ == "__main__":
    # Use the PORT environment variable if available (for cloud deployment)
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
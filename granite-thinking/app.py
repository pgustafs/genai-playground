from typing import Annotated, Generator, Dict, List, Any
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os
import gradio as gr
import uuid

# Configure model settings from environment variables or use defaults
model_endpoint_url = os.getenv("MODEL_ENDPOINT_URL", "http://127.0.0.1:8000")
model_service = f"{model_endpoint_url}/v1"
model_name = os.getenv("MODEL_NAME", "granite-3.2-8b-instruct")

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gradio-chatbot")

# Define the state structure for the conversation graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define system prompts - regular and thinking mode
system_prompt = {"role": "system", "content": "You are Granite, developed by IBM. You are a helpful AI assistant. Respond to every user query in a comprehensive and detailed way."}

system_prompt_thinking = {"role": "system", "content": "You are Granite, developed by IBM. You are a helpful AI assistant. Respond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts after 'Here is my thought process:' and write your response after 'Here is my response:' for each user query."}

# Initialize graph components
graph_builder = StateGraph(State)
memory = MemorySaver()

# Initialize the language model
llm = ChatOpenAI(
    model_name=model_name,
    base_url=model_service,
    api_key="EMPTY",
    streaming=True,
    temperature=0
)

# Define the chatbot node for the graph
def chatbot(state: State):
    """Process messages through the language model and return updated state"""
    return {"messages": [llm.invoke(state["messages"])]}

# Configure the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph with memory checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Function to stream responses from the LLM to the Gradio interface
def stream_response(message: str, history: List[Dict[str, str]], session_id: str, thinking_mode: bool) -> Generator[List[Dict[str, str]], Any, None]:
    """
    Stream the model's response for a given message
    
    Args:
        message: The user's input message (empty in the modified flow)
        history: Conversation history (already contains the user's message)
        session_id: Unique identifier for the user session
        thinking_mode: Whether to use thinking mode prompt
    
    Yields:
        Updated history with streamed response chunks
    """
    current_response = []
    try:
        # Select the appropriate system prompt based on thinking mode
        current_prompt = system_prompt_thinking if thinking_mode else system_prompt
        
        # Get the user's message from the history (it's already been added)
        user_message = history[-1]["content"]
        
        # Prepare messages for the model (excluding the last user message from history since we add it manually)
        messages = [current_prompt] + history[:-1] + [{"role": "user", "content": user_message}]
        # Configure unique thread ID for this session
        config = {"configurable": {"thread_id": session_id}}
        # Stream the response from the model
        for event in graph.stream({"messages": messages}, config=config):
            for value in event.values():
                content = value["messages"][-1].content
                current_response.append(content)
                # Since user message is already in history, just append assistant response
                yield history + [
                    {"role": "assistant", "content": "".join(current_response)}
                ]
    except Exception as e:
        logger.error(f"Error during response streaming: {e}", exc_info=True)
        # Since user message is already in history, just append error response
        yield history + [
            {"role": "assistant", "content": "An error occurred."}
        ]

# Create the Gradio interface
with gr.Blocks() as demo:
    # Create a unique session ID for each browser session
    session_id = gr.State(lambda: str(uuid.uuid4()))
    
    # Create UI components
    with gr.Row():
        with gr.Column(scale=10):
            gr.HTML("<h2>Granite AI Assistant</h2>")
        with gr.Column(scale=1):
            thinking_mode = gr.Checkbox(label="Thinking Mode", value=False)
    
    chatbot = gr.Chatbot(value=[], height=600, type="messages")
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here...", scale=9)
        clear = gr.ClearButton([msg, chatbot], scale=1)

    # Helper function to update chat with user message and clear input
    def user_message_submitted(message, history):
        """Immediately add user message to chat history and clear input"""
        return "", history + [{"role": "user", "content": message}]
    
    # Set up event handlers
    msg.submit(
        user_message_submitted,  # First add user message and clear input
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        stream_response,  # Then start streaming the assistant response
        [gr.Textbox(value="", visible=False), chatbot, session_id, thinking_mode],
        [chatbot]
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Log configuration information for debugging
    logger.info("Starting server with the following configuration:")
    logger.info(f"Model Endpoint URL: {model_endpoint_url}")
    logger.info(f"Model Service: {model_service}")
    logger.info(f"Model Name: {model_name}")
    
    # Launch with server listening on all interfaces for container environments
    demo.queue()
    # Get the root path from environment variable, default to "/"
    root_path = os.getenv("ROOT_PATH", "/")
    demo.launch(server_name="0.0.0.0", share=False, root_path=root_path)
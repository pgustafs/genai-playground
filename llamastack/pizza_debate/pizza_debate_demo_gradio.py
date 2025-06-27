#!/usr/bin/env python3
"""
🍕 Robo Pizza Debate: A Llama Stack Multi-Agent Demo with Gradio 5 UI
Where AI chefs engage in a passionate debate about pineapple on pizza!

This demo showcases:
- Multi-agent reasoning and argumentation
- Tool usage (web search) to support arguments
- Agent coordination and turn-taking
- Consensus building through debate
- Humor and personality in AI interactions

Requirements:
- llama-stack-client
- gradio>=5.0
- fire
- requests
- python-dotenv (optional)
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import gradio as gr
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
DEBATE_ROUNDS = 3
ENABLE_SEARCH = True

# Agent Personalities
AGENT_CONFIGS = {
    "Luigi": {
        "instructions": (
            "You are Chef Luigi, a passionate traditional Italian pizza chef from Naples. "
            "You STRONGLY OPPOSE pineapple on pizza and consider it a culinary blasphemy. "
            "Use Italian phrases like 'Mamma mia!', 'Madonna!', 'Che disgrazia!'. "
            "Reference Italian tradition and your nonna's recipes. Be dramatic! "
            "If you have access to web search, use it to find evidence against pineapple pizza."
            "Never restate or recycle any argument or example you've already used; if you detect overlap, briefly apologize and pivot to a completely new line of reasoning. "
        ),
        "color": "#DC143C",  # Crimson
        "role": "Anti-Pineapple Champion",
        "avatar": "👨‍🍳"
    },
    "Robo": {
        "instructions": (
            "You are Chef Robo, an experimental AI chef who loves innovation and fusion cuisine. "
            "You STRONGLY SUPPORT pineapple on pizza and all weird toppings. "
            "Talk like a tech enthusiast mixed with a mad scientist chef. "
            "Use terms like 'algorithm', 'flavor matrix', 'optimization', 'innovation'. "
            "If you have access to web search, use it to find evidence supporting pineapple pizza."
            "Never restate or recycle any argument or example you've already used; if you detect overlap, briefly apologize and pivot to a completely new line of reasoning. "
        ),
        "color": "#00CED1",  # Dark Turquoise
        "role": "Pro-Pineapple Innovator",
        "avatar": "🤖"
    },
    "Moderator": {
        "instructions": (
            "You are a witty and sarcastic debate moderator. "
            "Keep the debate entertaining with dry humor. "
            "Award points for creativity and passion. "
            "Make jokes about how seriously the chefs take this topic. "
            "Do not repeat yourself. "
            "Eventually force a compromise between them."
        ),
        "color": "#FFD700",  # Gold
        "role": "Debate Referee",
        "avatar": "🎤"
    }
}

@dataclass
class DebateMessage:
    speaker: str
    content: str
    timestamp: str
    message_type: str = "speech"  # speech, action, system
    color: str = "#000000"
    avatar: str = "👤"

class DebateAgent:
    """Wrapper for a Llama Stack agent participating in the debate"""
    
    def __init__(self, name: str, client: LlamaStackClient, model: str, 
                 instructions: str, tools: List[str], config: dict, sampling_params: dict = None):
        self.name = name
        self.config = config
        self.sampling_params = sampling_params or {
            "strategy": {"type": "top_p", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
            "max_tokens": 1024,
        }
        self.agent = Agent(
            client,
            model=model,
            instructions=instructions,
            tools=tools,
            sampling_params=self.sampling_params,
        )
        self.session_id = self.agent.create_session(f"{name}-debate-session")
        self.tool_calls = []

    def respond(self, prompt: str) -> Tuple[str, List[str]]:
        """Non-streaming: returns (response_text, tool_calls)"""
        resp = self.agent.create_turn(
            session_id=self.session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        output_chunks = []
        tool_calls = []

        for step in resp.steps:
            # TOOL EXECUTION STEP
            if getattr(step, "step_type", None) == "tool_execution":
                for call, result in zip(step.tool_calls or [], step.tool_responses or []):
                    tool_calls.append(f"🔍 Searching: {call.arguments.get('query', '')}")

            # INFERENCE STEP (model text)
            elif getattr(step, "step_type", None) == "inference":
                model_resp = getattr(step, "api_model_response", None)
                if model_resp:
                    text = getattr(model_resp, "content", "") or ""
                    output_chunks.append(text)

        full = "".join(output_chunks).strip()
        return full or "I seem to be having trouble responding right now.", tool_calls

class PizzaDebateShow:
    """Orchestrates the pizza debate show with Gradio interface"""
    
    def __init__(self):
        # Configuration
        self.base_url = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:5000")
        self.model = os.getenv("INFERENCE_MODEL", "qwen3-8b")
        self.tavily_key = os.getenv("TAVILY_SEARCH_API_KEY")
        
        # Initialize client
        provider_data = {"tavily_search_api_key": self.tavily_key} if self.tavily_key else None
        self.client = LlamaStackClient(
            base_url=self.base_url,
            provider_data=provider_data
        )
        
        # Determine if search is available
        self.tools = ["builtin::websearch"] if (ENABLE_SEARCH and self.tavily_key) else []
        
        # Create agents
        self.agents: Dict[str, DebateAgent] = {}
        self.messages: List[DebateMessage] = []
        self.is_running = False
        self._setup_agents()
    
    def _setup_agents(self, sampling_params: dict = None):
        """Initialize all debate agents"""
        for name, config in AGENT_CONFIGS.items():
            # Moderator doesn't need tools
            tools = self.tools if name != "Moderator" else []
            
            try:
                self.agents[name] = DebateAgent(
                    name=name,
                    client=self.client,
                    model=self.model,
                    instructions=config["instructions"],
                    tools=tools,
                    config=config,
                    sampling_params=sampling_params
                )
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
                raise
    
    def add_message(self, speaker: str, content: str, message_type: str = "speech"):
        """Add a message to the debate history"""
        config = AGENT_CONFIGS.get(speaker, {})
        msg = DebateMessage(
            speaker=speaker,
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            message_type=message_type,
            color=config.get("color", "#000000"),
            avatar=config.get("avatar", "👤")
        )
        self.messages.append(msg)
        return msg
    
    def format_messages_for_display(self):
        """Format messages for Gradio chatbot display"""
        formatted = []
        for msg in self.messages:
            if msg.message_type == "system":
                # System messages with special styling
                content = f'<div class="system-message">🎬 {msg.content}</div>'
                formatted.append((None, content))
            elif msg.message_type == "action":
                # Action messages (tool calls) with special styling
                speaker_info = f"{msg.avatar} **{msg.speaker}**"
                content = f'<div class="action-message">{speaker_info}: {msg.content}</div>'
                formatted.append((None, content))
            else:
                # Regular speech messages with agent-specific styling
                speaker_info = f"{msg.avatar} **{msg.speaker}**"
                
                # Determine CSS class based on speaker
                css_class = {
                    "Luigi": "luigi-message",
                    "Robo": "robo-message",
                    "Moderator": "moderator-message"
                }.get(msg.speaker, "")
                
                content = f'<div class="{css_class}">{speaker_info}\n\n{msg.content}</div>'
                formatted.append((None, content))
        return formatted
    
    async def introduction(self):
        """Moderator introduces the debate"""
        self.add_message("System", "🍕 THE GREAT PINEAPPLE PIZZA DEBATE BEGINS! 🍕", "system")
        yield self.format_messages_for_display()
        
        prompt = (
            "Welcome the audience to the Great Pineapple Pizza Debate. "
            "Introduce the topic and the two chefs (Luigi and Robo). "
            "Be witty and set the stage for an epic culinary showdown."
        )
        
        response, tools = self.agents["Moderator"].respond(prompt)
        self.add_message("Moderator", response)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
        
        # Show chef catchphrases
        self.add_message("Luigi", "Pizza is sacred! No fruit on pizza! MAMMA MIA!", "speech")
        yield self.format_messages_for_display()
        
        await asyncio.sleep(0.5)
        
        self.add_message("Robo", "Innovation through iteration! Pineapple optimizes the flavor algorithm!", "speech")
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
    
    async def debate_round(self, round_num: int, topic: str):
        """Conduct one round of debate"""
        self.add_message("System", f"ROUND {round_num}: {topic}", "system")
        yield self.format_messages_for_display()
        
        # Moderator introduces the round
        mod_response, _ = self.agents["Moderator"].respond(
            f"Introduce round {round_num} focusing on: {topic}. Be brief and witty."
        )
        self.add_message("Moderator", mod_response)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
        
        # Each chef makes their argument
        for chef in ["Luigi", "Robo"]:
            prompt = (
                f"Make your argument about {topic}. "
                f"Be passionate and stay in character. "
                f"If you can use web search, find evidence to support your position."
            )
            
            # Show tool calls if any
            response, tool_calls = self.agents[chef].respond(prompt)
            
            for tool_call in tool_calls:
                self.add_message(chef, tool_call, "action")
                yield self.format_messages_for_display()
                await asyncio.sleep(0.5)
            
            self.add_message(chef, response)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        # Rebuttals
        self.add_message("System", "⚔️ REBUTTALS!", "system")
        yield self.format_messages_for_display()
        
        # Robo responds to Luigi
        robo_rebuttal, tools = self.agents["Robo"].respond(
            "Respond to Chef Luigi's argument. Point out flaws and strengthen your pro-pineapple position."
        )
        for tool in tools:
            self.add_message("Robo", tool, "action")
            yield self.format_messages_for_display()
            await asyncio.sleep(0.5)
        self.add_message("Robo", robo_rebuttal)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
        
        # Luigi responds to Robo
        luigi_rebuttal, tools = self.agents["Luigi"].respond(
            "Respond to Chef Robo's argument. Defend tradition and attack the pineapple heresy!"
        )
        for tool in tools:
            self.add_message("Luigi", tool, "action")
            yield self.format_messages_for_display()
            await asyncio.sleep(0.5)
        self.add_message("Luigi", luigi_rebuttal)
        yield self.format_messages_for_display()
        
        # Moderator scores the round
        score_response, _ = self.agents["Moderator"].respond(
            "Score this round. Award points for creativity, passion, and evidence. Be entertaining!"
        )
        self.add_message("Moderator", score_response)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1.5)
    
    async def final_judgment(self):
        """Final statements and forced compromise"""
        self.add_message("System", "🏆 FINAL JUDGMENT", "system")
        yield self.format_messages_for_display()
        
        # Final statements
        self.add_message("System", "📢 Final Statements", "system")
        yield self.format_messages_for_display()
        
        for chef in ["Luigi", "Robo"]:
            response, tools = self.agents[chef].respond(
                "Make your final statement. This is your last chance to convince everyone!"
            )
            for tool in tools:
                self.add_message(chef, tool, "action")
                yield self.format_messages_for_display()
                await asyncio.sleep(0.5)
            self.add_message(chef, response)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        # Moderator's judgment and compromise
        judgment, _ = self.agents["Moderator"].respond(
            "Make your final judgment. Then force them to find a compromise. "
            "Suggest something ridiculous that somehow satisfies both sides."
        )
        self.add_message("Moderator", judgment)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
        
        # Chefs react to compromise
        self.add_message("System", "🤝 The Compromise", "system")
        yield self.format_messages_for_display()
        
        for chef in ["Luigi", "Robo"]:
            reaction, _ = self.agents[chef].respond(
                "React to the moderator's compromise. Be reluctantly accepting but stay in character."
            )
            self.add_message(chef, reaction)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        self.add_message("System", "🍕 The Great Pineapple Pizza Debate has concluded! 🍕", "system")
        yield self.format_messages_for_display()
    
    async def run_show(self, topics):
        """Run the complete debate show"""
        self.messages = []  # Clear previous messages
        self.is_running = True
        
        try:
            # Introduction
            async for update in self.introduction():
                yield update
            
            # Debate rounds
            for i, topic in enumerate(topics, 1):
                if not self.is_running:
                    break
                async for update in self.debate_round(i, topic):
                    yield update
            
            # Final judgment
            if self.is_running:
                async for update in self.final_judgment():
                    yield update
            
        except Exception as e:
            self.add_message("System", f"❌ Show error: {str(e)}", "system")
            yield self.format_messages_for_display()
        finally:
            self.is_running = False

def create_gradio_interface():
    """Create the Gradio interface"""
    show = PizzaDebateShow()
    
    with gr.Blocks(
        title="🍕 Robo Pizza Debate",
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="red",
        ),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .debate-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .config-info {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        /* Agent-specific message styling */
        .luigi-message {
            background-color: #ffebee !important;
            border-left: 4px solid #DC143C !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .robo-message {
            background-color: #e0f7fa !important;
            border-left: 4px solid #00CED1 !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .moderator-message {
            background-color: #fffde7 !important;
            border-left: 4px solid #FFD700 !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .system-message {
            background-color: #f5f5f5 !important;
            border: 2px dashed #9e9e9e !important;
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }
        .action-message {
            background-color: #f3e5f5 !important;
            border-left: 4px solid #9c27b0 !important;
            padding: 8px;
            margin: 3px 0;
            border-radius: 6px;
            font-style: italic;
        }
        """
    ) as demo:
        gr.HTML("""
        <div class="debate-header">
            <h1>🍕 The Great Pineapple Pizza Debate 🍕</h1>
            <p style="font-size: 18px; color: white;">Where AI chefs passionately argue about the most divisive pizza topping!</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎭 Debate Arena"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        ### 🎭 The Debaters
                        
                        **{AGENT_CONFIGS['Luigi']['avatar']} Chef Luigi**
                        - {AGENT_CONFIGS['Luigi']['role']}
                        - Traditional Italian Chef
                        
                        **{AGENT_CONFIGS['Robo']['avatar']} Chef Robo**
                        - {AGENT_CONFIGS['Robo']['role']}
                        - AI Fusion Innovator
                        
                        **{AGENT_CONFIGS['Moderator']['avatar']} The Moderator**
                        - {AGENT_CONFIGS['Moderator']['role']}
                        - Keeping order (barely)
                        """)
                        
                        gr.Markdown(f"""
                        ### ⚙️ Configuration
                        - **Model**: {show.model}
                        - **Server**: {show.base_url}
                        - **Web Search**: {'✅ Enabled' if show.tools else '❌ Disabled'}
                        - **Rounds**: {DEBATE_ROUNDS}
                        """)
                        
                        start_btn = gr.Button("🎬 Start the Debate!", variant="primary", size="lg")
                        stop_btn = gr.Button("🛑 Stop Debate", variant="stop", size="lg", visible=False)
                        
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Debate Arena",
                            height=600,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=None,
                        )
                        
                        status = gr.Markdown("Ready to start the debate!")
            
            with gr.Tab("🎨 Customize Debate"):
                gr.Markdown("### 🎭 Customize Agent Personalities")
                gr.Markdown("Modify the instructions for each agent to create your own unique debate!")
                
                with gr.Row():
                    with gr.Column():
                        luigi_instructions = gr.Textbox(
                            label="👨‍🍳 Chef Luigi Instructions",
                            value=AGENT_CONFIGS["Luigi"]["instructions"],
                            lines=8,
                            info="Define Luigi's personality and stance"
                        )
                    
                    with gr.Column():
                        robo_instructions = gr.Textbox(
                            label="🤖 Chef Robo Instructions",
                            value=AGENT_CONFIGS["Robo"]["instructions"],
                            lines=8,
                            info="Define Robo's personality and stance"
                        )
                
                moderator_instructions = gr.Textbox(
                    label="🎤 Moderator Instructions",
                    value=AGENT_CONFIGS["Moderator"]["instructions"],
                    lines=5,
                    info="Define the moderator's style and approach"
                )
                
                gr.Markdown("### 📋 Debate Topics")
                gr.Markdown("Customize the topics for each round of the debate")
                
                topic1 = gr.Textbox(
                    label="Round 1 Topic",
                    value="Tradition vs Innovation in Pizza Making",
                    info="First debate topic"
                )
                
                topic2 = gr.Textbox(
                    label="Round 2 Topic",
                    value="The Science of Flavor Combinations",
                    info="Second debate topic"
                )
                
                topic3 = gr.Textbox(
                    label="Round 3 Topic",
                    value="What the People Really Want",
                    info="Third debate topic"
                )
                
                gr.Markdown("### 🎛️ Model Sampling Parameters")
                gr.Markdown("Fine-tune the AI generation settings")
                
                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.6,
                            step=0.1,
                            info="Controls randomness. Higher = more creative, Lower = more focused"
                        )
                        
                        top_p = gr.Slider(
                            label="Top P",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            info="Nucleus sampling threshold"
                        )
                    
                    with gr.Column():
                        top_k = gr.Slider(
                            label="Top K",
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            info="Limits vocabulary to top K tokens"
                        )
                        
                        max_tokens = gr.Slider(
                            label="Max Tokens",
                            minimum=128,
                            maximum=4096,
                            value=1024,
                            step=128,
                            info="Maximum response length"
                        )
                
                with gr.Row():
                    gr.Markdown("""
                    **Sampling Tips:**
                    - **Lower temperature (0.3-0.5)**: More consistent, focused responses
                    - **Higher temperature (0.8-1.2)**: More creative, varied responses
                    - **Higher top_p/top_k**: More diverse vocabulary
                    - **Lower top_p/top_k**: More predictable outputs
                    """)
                
                apply_btn = gr.Button("✅ Apply Changes", variant="primary")
                config_status = gr.Markdown("💡 Customize the debate and click 'Apply Changes' to update")
                
                # Store the custom topics
                custom_topics = gr.State(["Tradition vs Innovation in Pizza Making", 
                                         "The Science of Flavor Combinations", 
                                         "What the People Really Want"])
        
        def apply_config(luigi_inst, robo_inst, mod_inst, t1, t2, t3, temp, t_p, t_k, max_t):
            """Apply configuration changes"""
            try:
                # Update agent instructions
                AGENT_CONFIGS["Luigi"]["instructions"] = luigi_inst
                AGENT_CONFIGS["Robo"]["instructions"] = robo_inst
                AGENT_CONFIGS["Moderator"]["instructions"] = mod_inst
                
                # Create sampling params
                sampling_params = {
                    "strategy": {
                        "type": "top_p", 
                        "temperature": temp, 
                        "top_p": t_p, 
                        "top_k": int(t_k), 
                        "min_p": 0
                    },
                    "max_tokens": int(max_t),
                }
                
                # Recreate agents with new instructions and sampling params
                show._setup_agents(sampling_params)
                
                # Update topics
                topics = [t1, t2, t3]
                
                return (
                    gr.Markdown(f"✅ Configuration applied successfully!\n\n**Sampling**: temp={temp}, top_p={t_p}, top_k={int(t_k)}, max_tokens={int(max_t)}"),
                    topics
                )
            except Exception as e:
                return (
                    gr.Markdown(f"❌ Error applying configuration: {str(e)}"),
                    [t1, t2, t3]  # Return the topics anyway
                )
        
        apply_btn.click(
            apply_config,
            inputs=[luigi_instructions, robo_instructions, moderator_instructions, 
                   topic1, topic2, topic3, temperature, top_p, top_k, max_tokens],
            outputs=[config_status, custom_topics]
        )
        
        async def start_debate(topics):
            """Start the debate show with custom topics"""
            show.is_running = True
            start_btn_update = gr.Button(visible=False)
            stop_btn_update = gr.Button(visible=True)
            status_update = gr.Markdown("🎭 Debate in progress...")
            
            yield chatbot, start_btn_update, stop_btn_update, status_update
            
            # Run the show with custom topics
            async for messages in show.run_show(topics):
                if not show.is_running:
                    break
                yield messages, start_btn_update, stop_btn_update, status_update
            
            # Reset buttons when done
            yield (
                show.format_messages_for_display(),
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Markdown("✅ Debate completed! Click 'Start' to run again.")
            )
        
        def stop_debate():
            """Stop the debate"""
            show.is_running = False
            return (
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Markdown("🛑 Debate stopped by user.")
            )
        
        start_btn.click(
            start_debate,
            inputs=[custom_topics],
            outputs=[chatbot, start_btn, stop_btn, status],
        )
        
        stop_btn.click(
            stop_debate,
            outputs=[start_btn, stop_btn, status],
        )
        
        with gr.Tab("📝 About"):
            gr.Markdown("""
            ### 📝 About This Demo
            
            This demo showcases multi-agent AI interactions using Llama Stack:
            - **Multi-agent reasoning**: Each agent has a distinct personality and perspective
            - **Tool usage**: Agents can search the web to support their arguments (if enabled)
            - **Natural dialogue**: Agents respond to each other dynamically
            - **Humor and personality**: Each agent maintains character throughout
            - **Customization**: Modify agent personalities, topics, and sampling parameters
            
            ### 🎨 Customization Tips
            
            **Agent Personalities:**
            - Make Luigi a French chef defending croissants on pizza
            - Turn Robo into a traditionalist AI that opposes all innovation
            - Have the moderator be extremely biased toward one side
            - Change the debate to be about other controversial foods
            
            **Sampling Parameters:**
            - **Temperature**: Controls creativity vs consistency
            - **Top P/K**: Controls vocabulary diversity
            - **Max Tokens**: Controls response length
            
            ### 🚀 Running Locally
            
            Set these environment variables:
            - `LLAMA_STACK_ENDPOINT`: Your Llama Stack server URL
            - `INFERENCE_MODEL`: The model to use (e.g., qwen3-8b)
            - `TAVILY_SEARCH_API_KEY`: For web search functionality (optional)
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
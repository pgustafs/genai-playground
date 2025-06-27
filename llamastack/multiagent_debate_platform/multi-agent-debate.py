#!/usr/bin/env python3
"""
Multi-Agent Debate Platform: A Llama Stack Demo with Gradio UI
A serious, customizable debate platform for any topic.

Features:
- Configurable number of debate rounds
- Customizable agent personalities and positions
- Support for any debate topic
- Professional, neutral interface
- Optional web search for evidence
- Real-time visualization

Requirements:
- llama-stack-client
- gradio>=5.0
- python-dotenv (optional)
"""

import os
import logging
import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import gradio as gr
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default Agent Configurations
DEFAULT_AGENTS = {
    "Proponent": {
        "instructions": "You are a thoughtful advocate who supports the proposition. Present logical arguments with evidence. Be respectful but firm in your position.",
        "color": "#2E7D32",  # Green
        "role": "Supporting Position",
        "avatar": "👤"
    },
    "Opponent": {
        "instructions": "You are a thoughtful critic who opposes the proposition. Present logical counterarguments with evidence. Be respectful but firm in your position.",
        "color": "#C62828",  # Red
        "role": "Opposing Position",
        "avatar": "👥"
    },
    "Moderator": {
        "instructions": "You are a neutral, professional debate moderator. Keep the discussion focused and balanced. Ensure both sides get equal opportunity to present their arguments.",
        "color": "#1565C0",  # Blue
        "role": "Neutral Moderator",
        "avatar": "⚖️"
    }
}

@dataclass
class DebateMessage:
    speaker: str
    content: str
    timestamp: str
    message_type: str = "speech"
    color: str = "#000000"
    avatar: str = "👤"

class DebateAgent:
    """Wrapper for a Llama Stack agent in the debate"""
    
    def __init__(self, name: str, client: LlamaStackClient, model: str, 
                 instructions: str, tools: List[str], config: dict, sampling_params: dict = None):
        self.name = name
        self.config = config
        self.sampling_params = sampling_params or {
            "strategy": {"type": "top_p", "temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0},
            "max_tokens": 1024,
        }
        self.agent = Agent(
            client,
            model=model,
            instructions=instructions,
            tools=tools,
            sampling_params=self.sampling_params,
        )
        self.session_id = self.agent.create_session(f"{name}-session")

    def respond(self, prompt: str) -> Tuple[str, List[str]]:
        """Get response from agent"""
        resp = self.agent.create_turn(
            session_id=self.session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        output_chunks = []
        tool_calls = []

        for step in resp.steps:
            if getattr(step, "step_type", None) == "tool_execution":
                for call, result in zip(step.tool_calls or [], step.tool_responses or []):
                    tool_calls.append(f"📊 Researching: {call.arguments.get('query', '')}")

            elif getattr(step, "step_type", None) == "inference":
                model_resp = getattr(step, "api_model_response", None)
                if model_resp:
                    text = getattr(model_resp, "content", "") or ""
                    output_chunks.append(text)

        full = "".join(output_chunks).strip()
        return full or "Unable to generate response.", tool_calls

class DebatePlatform:
    """Orchestrates the debate"""
    
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
        
        # Tools
        self.tools = ["builtin::websearch"] if self.tavily_key else []
        
        # State
        self.agents: Dict[str, DebateAgent] = {}
        self.messages: List[DebateMessage] = []
        self.is_running = False
        self.agent_configs = DEFAULT_AGENTS.copy()
    
    def setup_agents(self, agent_configs: dict = None, sampling_params: dict = None):
        """Initialize agents with given configurations"""
        if agent_configs:
            self.agent_configs = agent_configs
            
        self.agents = {}
        for name, config in self.agent_configs.items():
            tools = self.tools if name != "Moderator" else []
            
            self.agents[name] = DebateAgent(
                name=name,
                client=self.client,
                model=self.model,
                instructions=config["instructions"],
                tools=tools,
                config=config,
                sampling_params=sampling_params
            )
    
    def add_message(self, speaker: str, content: str, message_type: str = "speech"):
        """Add message to debate history"""
        config = self.agent_configs.get(speaker, {})
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
        """Format messages for Gradio display"""
        formatted = []
        for msg in self.messages:
            if msg.message_type == "system":
                content = f'<div class="system-message">{msg.content}</div>'
                formatted.append((None, content))
            elif msg.message_type == "research":
                speaker_info = f"{msg.avatar} **{msg.speaker}**"
                content = f'<div class="research-message">{speaker_info}: {msg.content}</div>'
                formatted.append((None, content))
            else:
                speaker_info = f"{msg.avatar} **{msg.speaker}**"
                css_class = {
                    "Proponent": "proponent-message",
                    "Opponent": "opponent-message", 
                    "Moderator": "moderator-message"
                }.get(msg.speaker, "")
                
                content = f'<div class="{css_class}">{speaker_info}\n\n{msg.content}</div>'
                formatted.append((None, content))
        return formatted
    
    async def introduction(self, topic: str):
        """Moderator introduces the debate"""
        self.add_message("System", f"DEBATE TOPIC: {topic}", "system")
        yield self.format_messages_for_display()
        
        prompt = f"""Welcome everyone to today's debate on: "{topic}"
        
        Please introduce:
        1. The topic and its importance
        2. The format of the debate
        3. The two positions that will be argued
        
        Be professional and neutral."""
        
        response, _ = self.agents["Moderator"].respond(prompt)
        self.add_message("Moderator", response)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
    
    async def debate_round(self, round_num: int, round_topic: str):
        """Conduct one round of debate"""
        self.add_message("System", f"ROUND {round_num}: {round_topic}", "system")
        yield self.format_messages_for_display()
        
        # Moderator introduces the round
        mod_response, _ = self.agents["Moderator"].respond(
            f"Introduce round {round_num} focusing on: {round_topic}. Be brief and professional."
        )
        self.add_message("Moderator", mod_response)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1)
        
        # Arguments from both sides
        for agent in ["Proponent", "Opponent"]:
            prompt = f"""Present your argument regarding: {round_topic}
            
            - Make clear, logical points
            - Use evidence if available
            - Be respectful but persuasive
            - Stay focused on this specific aspect of the debate"""
            
            response, tool_calls = self.agents[agent].respond(prompt)
            
            # Show research if performed
            for tool_call in tool_calls:
                self.add_message(agent, tool_call, "research")
                yield self.format_messages_for_display()
                await asyncio.sleep(0.5)
            
            self.add_message(agent, response)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        # Rebuttals
        self.add_message("System", "REBUTTALS", "system")
        yield self.format_messages_for_display()
        
        # Each side responds to the other
        for responder, target in [("Opponent", "Proponent"), ("Proponent", "Opponent")]:
            rebuttal_prompt = f"Respond to the {target}'s argument. Address their key points and strengthen your position."
            
            rebuttal, tools = self.agents[responder].respond(rebuttal_prompt)
            for tool in tools:
                self.add_message(responder, tool, "research")
                yield self.format_messages_for_display()
                await asyncio.sleep(0.5)
                
            self.add_message(responder, rebuttal)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        # Moderator summary
        summary, _ = self.agents["Moderator"].respond(
            "Briefly summarize the key arguments presented in this round. Remain neutral."
        )
        self.add_message("Moderator", summary)
        yield self.format_messages_for_display()
        
        await asyncio.sleep(1.5)
    
    async def closing_statements(self):
        """Final statements from both sides"""
        self.add_message("System", "CLOSING STATEMENTS", "system")
        yield self.format_messages_for_display()
        
        for agent in ["Proponent", "Opponent"]:
            response, tools = self.agents[agent].respond(
                "Present your closing statement. Summarize your strongest arguments and why your position should prevail."
            )
            
            for tool in tools:
                self.add_message(agent, tool, "research")
                yield self.format_messages_for_display()
                await asyncio.sleep(0.5)
                
            self.add_message(agent, response)
            yield self.format_messages_for_display()
            await asyncio.sleep(1)
        
        # Moderator conclusion
        conclusion, _ = self.agents["Moderator"].respond(
            "Conclude the debate by summarizing both positions neutrally. Thank the participants."
        )
        self.add_message("Moderator", conclusion)
        yield self.format_messages_for_display()
        
        self.add_message("System", "DEBATE CONCLUDED", "system")
        yield self.format_messages_for_display()
    
    async def run_debate(self, topic: str, round_topics: List[str]):
        """Run the complete debate"""
        self.messages = []
        self.is_running = True
        
        try:
            # Introduction
            async for update in self.introduction(topic):
                yield update
            
            # Debate rounds
            for i, round_topic in enumerate(round_topics, 1):
                if not self.is_running:
                    break
                async for update in self.debate_round(i, round_topic):
                    yield update
            
            # Closing statements
            if self.is_running:
                async for update in self.closing_statements():
                    yield update
            
        except Exception as e:
            self.add_message("System", f"Error: {str(e)}", "system")
            yield self.format_messages_for_display()
        finally:
            self.is_running = False

def create_interface():
    """Create the Gradio interface"""
    platform = DebatePlatform()
    
    with gr.Blocks(
        title="Multi-Agent Debate Platform",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .debate-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%);
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
        }
        .proponent-message {
            background-color: #E8F5E9 !important;
            border-left: 4px solid #2E7D32 !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .opponent-message {
            background-color: #FFEBEE !important;
            border-left: 4px solid #C62828 !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .moderator-message {
            background-color: #E3F2FD !important;
            border-left: 4px solid #1565C0 !important;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        .system-message {
            background-color: #F5F5F5 !important;
            border: 2px solid #9E9E9E !important;
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }
        .research-message {
            background-color: #FFF3E0 !important;
            border-left: 4px solid #F57C00 !important;
            padding: 8px;
            margin: 3px 0;
            border-radius: 6px;
            font-style: italic;
        }
        """
    ) as demo:
        gr.HTML("""
        <div class="debate-header">
            <h1>Multi-Agent Debate Platform</h1>
            <p style="font-size: 18px;">Professional AI-powered debates on any topic</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎭 Debate Arena"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        ### Participants
                        
                        **{DEFAULT_AGENTS['Proponent']['avatar']} Proponent**
                        - Supports the proposition
                        
                        **{DEFAULT_AGENTS['Opponent']['avatar']} Opponent**
                        - Opposes the proposition
                        
                        **{DEFAULT_AGENTS['Moderator']['avatar']} Moderator**
                        - Neutral facilitator
                        """)
                        
                        gr.Markdown(f"""
                        ### System Info
                        - **Model**: {platform.model}
                        - **Server**: {platform.base_url}
                        - **Research**: {'Enabled' if platform.tools else 'Disabled'}
                        """)
                        
                        start_btn = gr.Button("Start Debate", variant="primary", size="lg")
                        stop_btn = gr.Button("Stop Debate", variant="stop", size="lg", visible=False)
                        
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Debate Transcript",
                            height=600,
                            show_copy_button=True,
                            bubble_full_width=False,
                        )
                        
                        status = gr.Markdown("Configure the debate in the Setup tab, then click Start.")
            
            with gr.Tab("⚙️ Setup"):
                gr.Markdown("### 📋 Debate Configuration")
                
                main_topic = gr.Textbox(
                    label="Main Debate Topic",
                    value="Should artificial intelligence be regulated by international law?",
                    info="The central question or proposition to be debated"
                )
                
                num_rounds = gr.Slider(
                    label="Number of Rounds",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    info="How many rounds of argumentation"
                )
                
                # Dynamic round topics
                round_topics = []
                for i in range(5):
                    round_topic = gr.Textbox(
                        label=f"Round {i+1} Focus",
                        value=["Ethical considerations", "Economic impact", "Technical feasibility", "Social implications", "Future scenarios"][i] if i < 3 else "",
                        visible=i < 3,
                        info="Specific aspect to focus on in this round"
                    )
                    round_topics.append(round_topic)
                
                def update_round_visibility(n):
                    return [gr.update(visible=i < n) for i in range(5)]
                
                num_rounds.change(
                    update_round_visibility,
                    inputs=[num_rounds],
                    outputs=round_topics
                )
                
                gr.Markdown("### 👥 Agent Instructions")
                
                proponent_inst = gr.Textbox(
                    label="Proponent Instructions",
                    value=DEFAULT_AGENTS["Proponent"]["instructions"],
                    lines=4,
                    info="How the supporting side should argue"
                )
                
                opponent_inst = gr.Textbox(
                    label="Opponent Instructions",
                    value=DEFAULT_AGENTS["Opponent"]["instructions"],
                    lines=4,
                    info="How the opposing side should argue"
                )
                
                moderator_inst = gr.Textbox(
                    label="Moderator Instructions",
                    value=DEFAULT_AGENTS["Moderator"]["instructions"],
                    lines=4,
                    info="How the moderator should facilitate"
                )
                
                gr.Markdown("### 🎛️ Model Parameters")
                
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        info="Response creativity"
                    )
                    
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=128,
                        info="Maximum response length"
                    )
                
                apply_btn = gr.Button("Apply Configuration", variant="primary")
                config_status = gr.Markdown("")
                
                # Store configuration
                config_state = gr.State({})
                
                def apply_config(main_topic, n_rounds, *args):
                    """Apply all configuration settings"""
                    try:
                        # Extract round topics (first n_rounds from args[0:5])
                        topics = [args[i] for i in range(int(n_rounds))]
                        
                        # Extract other settings
                        prop_inst, opp_inst, mod_inst, temp, max_t = args[5:10]
                        
                        # Update agent configs
                        agent_configs = {
                            "Proponent": {
                                "instructions": prop_inst,
                                "color": "#2E7D32",
                                "role": "Supporting Position",
                                "avatar": "👤"
                            },
                            "Opponent": {
                                "instructions": opp_inst,
                                "color": "#C62828",
                                "role": "Opposing Position",
                                "avatar": "👥"
                            },
                            "Moderator": {
                                "instructions": mod_inst,
                                "color": "#1565C0",
                                "role": "Neutral Moderator",
                                "avatar": "⚖️"
                            }
                        }
                        
                        # Create sampling params
                        sampling_params = {
                            "strategy": {
                                "type": "top_p",
                                "temperature": temp,
                                "top_p": 0.9,
                                "top_k": 40,
                                "min_p": 0
                            },
                            "max_tokens": int(max_t),
                        }
                        
                        # Setup agents
                        platform.setup_agents(agent_configs, sampling_params)
                        
                        # Store config
                        config = {
                            "topic": main_topic,
                            "round_topics": topics
                        }
                        
                        return (
                            gr.Markdown(f"✅ Configuration applied! Ready to debate with {int(n_rounds)} rounds."),
                            config
                        )
                    except Exception as e:
                        return (
                            gr.Markdown(f"❌ Error: {str(e)}"),
                            {}
                        )
                
                apply_btn.click(
                    apply_config,
                    inputs=[main_topic, num_rounds] + round_topics + [proponent_inst, opponent_inst, moderator_inst, temperature, max_tokens],
                    outputs=[config_status, config_state]
                )
                
            with gr.Tab("📖 Guide"):
                gr.Markdown("""
                ### How to Use
                
                1. **Configure the Debate** in the Setup tab:
                   - Set your main topic/question
                   - Choose number of rounds
                   - Define focus areas for each round
                   - Customize agent instructions
                   - Adjust model parameters
                
                2. **Apply Configuration** to save your settings
                
                3. **Start the Debate** in the Debate Arena tab
                
                ### Tips for Good Debates
                
                - **Clear Topics**: Frame topics as clear yes/no questions or propositions
                - **Focused Rounds**: Each round should explore a different aspect
                - **Balanced Instructions**: Ensure both sides have equal constraints
                - **Temperature**: 0.5-0.7 for formal debates, 0.8-1.0 for creative discussions
                
                ### Example Topics
                
                - "Should universal basic income be implemented?"
                - "Is space exploration worth the investment?"
                - "Should social media platforms be held liable for user content?"
                - "Is nuclear energy the solution to climate change?"
                - "Should gene editing in humans be permitted?"
                """)
        
        async def start_debate(config):
            """Start the debate with stored configuration"""
            if not config or "topic" not in config:
                yield (
                    gr.Chatbot(value=[(None, "Please configure the debate in the Setup tab first.")]),
                    gr.Button(visible=True),
                    gr.Button(visible=False),
                    gr.Markdown("❌ No configuration found. Please use the Setup tab.")
                )
                return
            
            platform.is_running = True
            start_update = gr.Button(visible=False)
            stop_update = gr.Button(visible=True)
            status_update = gr.Markdown("🎭 Debate in progress...")
            
            yield chatbot, start_update, stop_update, status_update
            
            # Run debate
            async for messages in platform.run_debate(config["topic"], config["round_topics"]):
                if not platform.is_running:
                    break
                yield messages, start_update, stop_update, status_update
            
            # Reset
            yield (
                platform.format_messages_for_display(),
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Markdown("✅ Debate completed!")
            )
        
        def stop_debate():
            """Stop the ongoing debate"""
            platform.is_running = False
            return (
                gr.Button(visible=True),
                gr.Button(visible=False),
                gr.Markdown("🛑 Debate stopped.")
            )
        
        start_btn.click(
            start_debate,
            inputs=[config_state],
            outputs=[chatbot, start_btn, stop_btn, status],
        )
        
        stop_btn.click(
            stop_debate,
            outputs=[start_btn, stop_btn, status],
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
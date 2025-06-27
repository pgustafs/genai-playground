#!/usr/bin/env python3
"""
🍕 Robo Pizza Debate: A Llama Stack Multi-Agent Demo
Where AI chefs engage in a passionate debate about pineapple on pizza!

This demo showcases:
- Multi-agent reasoning and argumentation
- Tool usage (web search) to support arguments
- Agent coordination and turn-taking
- Consensus building through debate
- Humor and personality in AI interactions

Requirements:
- llama-stack-client
- termcolor
- python-dotenv (optional)
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from termcolor import cprint
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client import AgentEventLogger
from rich.pretty import pprint



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
            "Do not repeat yourself."
            "If you have access to web search, use it to find evidence against pineapple pizza."
        ),
        "color": "red",
        "role": "Anti-Pineapple Champion"
    },
    "Robo": {
        "instructions": (
            "You are Chef Robo, an experimental AI chef who loves innovation and fusion cuisine. "
            "You STRONGLY SUPPORT pineapple on pizza and all weird toppings. "
            "Talk like a tech enthusiast mixed with a mad scientist chef. "
            "Use terms like 'algorithm', 'flavor matrix', 'optimization', 'innovation'. "
            "Do not repeat yourself. "
            "If you have access to web search, use it to find evidence supporting pineapple pizza."
        ),
        "color": "cyan",
        "role": "Pro-Pineapple Innovator"
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
        "color": "yellow",
        "role": "Debate Referee"
    }
}

class DebateAgent:
    """Wrapper for a Llama Stack agent participating in the debate"""
    
    def __init__(self, name: str, client: LlamaStackClient, model: str, 
                 instructions: str, tools: List[str], color: str):
        self.name = name
        self.color = color
        self.agent = Agent(
            client,
            model=model,
            instructions=instructions,
            tools=tools,
            # According to https://huggingface.co/Qwen/Qwen3-8B#best-practices
            sampling_params={
                "strategy": {"type": "top_p", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
                "max_tokens": 2048,
            },
            #enable_session_persistence=True
        )
        self.session_id = self.agent.create_session(f"{name}-debate-session")

    def respond(self, prompt: str) -> str:
        """Non-streaming: inspect resp.steps for tool calls + model text."""
        resp = self.agent.create_turn(
            session_id=self.session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        # Debug: see the raw structure if you ever need it
        # pprint(resp.steps)

        output_chunks: list[str] = []

        for step in resp.steps:
            # TOOL EXECUTION STEP
            if getattr(step, "step_type", None) == "tool_execution":
                for call, result in zip(step.tool_calls or [], step.tool_responses or []):
                    cprint(f"\n🔍 {self.name} CALLING '{call.tool_name}'", "green")
                    cprint(f"   Args: {call.arguments}", "green")
                    #cprint(f"✅ {self.name} GOT    '{call.tool_name}': {result.content}", "green")

            # INFERENCE STEP (model text)
            elif getattr(step, "step_type", None) == "inference":
                model_resp = getattr(step, "api_model_response", None)
                if model_resp:
                    text = getattr(model_resp, "content", "") or ""
                    output_chunks.append(text)

        full = "".join(output_chunks).strip()
        return full or "I seem to be having trouble responding right now."
    
    def print_response(self, response: str):
        """Print the agent's response with color"""
        cprint(f"\n🍕 {self.name}:", self.color, attrs=["bold"])
        cprint(response, self.color)

class PizzaDebateShow:
    """Orchestrates the pizza debate show"""
    
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
        self._setup_agents()
        
        # Show configuration
        self._print_header()
    
    def _setup_agents(self):
        """Initialize all debate agents"""
        cprint("Setting up debate participants...\n", "cyan")
        
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
                    color=config["color"]
                )
                cprint(f"✅ {name} ({config['role']}) ready!", config["color"])
            except Exception as e:
                cprint(f"❌ Failed to create {name}: {e}", "red")
                raise
    
    def _print_header(self):
        """Print show header"""
        cprint("\n" + "="*60, "magenta")
        cprint("🍕 ROBO PIZZA DEBATE SHOW 🍕", "magenta", attrs=["bold"])
        cprint("The Ultimate Showdown: Does Pineapple Belong on Pizza?", "white")
        cprint("="*60, "magenta")
        cprint(f"Model: {self.model}", "cyan")
        cprint(f"Server: {self.base_url}", "cyan")
        cprint(f"Web Search: {'Enabled' if self.tools else 'Disabled'}", "cyan")
        print()
    
    def introduction(self):
        """Moderator introduces the debate"""
        prompt = (
            "Welcome the audience to the Great Pineapple Pizza Debate. "
            "Introduce the topic and the two chefs (Luigi and Robo). "
            "Be witty and set the stage for an epic culinary showdown."
        )
        
        response = self.agents["Moderator"].respond(prompt)
        self.agents["Moderator"].print_response(response)
        
        # Show chef catchphrases
        time.sleep(1)
        cprint("\n🍕 Chef Luigi:", "red", attrs=["bold"])
        cprint("Pizza is sacred! No fruit on pizza! MAMMA MIA!", "red", attrs=["bold"])
        
        cprint("\n🍕 Chef Robo:", "cyan", attrs=["bold"])
        cprint("Innovation through iteration! Pineapple optimizes the flavor algorithm!", "cyan", attrs=["bold"])
        
        time.sleep(2)
    
    def debate_round(self, round_num: int, topic: str):
        """Conduct one round of debate"""
        cprint(f"\n{'='*60}", "white")
        cprint(f"ROUND {round_num}: {topic}", "white", attrs=["bold"])
        cprint('='*60, "white")
        
        # Moderator introduces the round
        mod_response = self.agents["Moderator"].respond(
            f"Introduce round {round_num} focusing on: {topic}. Be brief and witty."
        )
        self.agents["Moderator"].print_response(mod_response)
        
        # Each chef makes their argument
        for chef in ["Luigi", "Robo"]:
            prompt = (
                f"Make your argument about {topic}. "
                f"Be passionate and stay in character. "
                f"If you can use web search, find evidence to support your position."
            )
            response = self.agents[chef].respond(prompt)
            self.agents[chef].print_response(response)
            time.sleep(1)
        
        # Rebuttals
        cprint("\n⚔️  REBUTTALS!", "yellow", attrs=["bold"])
        
        # Robo responds to Luigi
        robo_rebuttal = self.agents["Robo"].respond(
            "Respond to Chef Luigi's argument. Point out flaws and strengthen your pro-pineapple position."
        )
        self.agents["Robo"].print_response(robo_rebuttal)
        
        # Luigi responds to Robo
        luigi_rebuttal = self.agents["Luigi"].respond(
            "Respond to Chef Robo's argument. Defend tradition and attack the pineapple heresy!"
        )
        self.agents["Luigi"].print_response(luigi_rebuttal)
        
        # Moderator scores the round
        score_response = self.agents["Moderator"].respond(
            "Score this round. Award points for creativity, passion, and evidence. Be entertaining!"
        )
        self.agents["Moderator"].print_response(score_response)
        
        time.sleep(2)
    
    def final_judgment(self):
        """Final statements and forced compromise"""
        cprint(f"\n{'='*60}", "magenta")
        cprint("🏆 FINAL JUDGMENT", "magenta", attrs=["bold"])
        cprint('='*60, "magenta")
        
        # Final statements
        cprint("\n📢 Final Statements:", "white", attrs=["bold"])
        
        for chef in ["Luigi", "Robo"]:
            response = self.agents[chef].respond(
                "Make your final statement. This is your last chance to convince everyone!"
            )
            self.agents[chef].print_response(response)
            time.sleep(1)
        
        # Moderator's judgment and compromise
        judgment = self.agents["Moderator"].respond(
            "Make your final judgment. Then force them to find a compromise. "
            "Suggest something ridiculous that somehow satisfies both sides."
        )
        self.agents["Moderator"].print_response(judgment)
        
        # Chefs react to compromise
        cprint("\n🤝 The Compromise:", "green", attrs=["bold"])
        
        for chef in ["Luigi", "Robo"]:
            reaction = self.agents[chef].respond(
                "React to the moderator's compromise. Be reluctantly accepting but stay in character."
            )
            self.agents[chef].print_response(reaction)
            time.sleep(1)
    
    def run_show(self):
        """Run the complete debate show"""
        try:
            # Introduction
            self.introduction()
            
            # Debate rounds
            topics = [
                "Tradition vs Innovation in Pizza Making",
                "The Science of Flavor Combinations",
                "What the People Really Want"
            ]
            
            for i, topic in enumerate(topics, 1):
                self.debate_round(i, topic)
            
            # Final judgment
            self.final_judgment()
            
            # Closing
            cprint("\n" + "="*60, "magenta")
            cprint("🍕 Thank you for watching the Great Pineapple Pizza Debate! 🍕", "green", attrs=["bold"])
            cprint("Remember: Pizza brings us together, even when toppings tear us apart!", "white")
            cprint("="*60, "magenta")
            
        except KeyboardInterrupt:
            cprint("\n\n🛑 Debate interrupted! The pineapple question remains unsolved...", "red")
        except Exception as e:
            cprint(f"\n❌ Show error: {e}", "red")
            raise

def main():
    """Entry point"""
    show = PizzaDebateShow()
    show.run_show()

if __name__ == "__main__":
    main()
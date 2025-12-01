# agent.py
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.genai import types

from .prompts import PROMPT_SLOW_QUERY_ANALYZER
from .utils import fetch_slow_queries, fetch_single_query

__dir__ = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(__dir__, "../../.env"))

# Get the model from environment variable
MODEL = os.getenv('MODEL')

# Single agent that orchestrates the entire workflow
slow_query_analyzer = LlmAgent(
    name="slow_query_analyzer",
    model=MODEL,
    description="Analyzes slow queries by fetching them individually to avoid token limits.",
    instruction=PROMPT_SLOW_QUERY_ANALYZER,
    tools=[fetch_slow_queries, fetch_single_query],
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=8192
    )
)

# Alias for ADK CLI compatibility
root_agent = slow_query_analyzer

"""Data Collection Agent — produces structured data collection protocols."""

from __future__ import annotations

import json

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.research_design import DataCollectionPlan

DATA_COLLECTION_PROMPT = """\
You are a Data Collection Agent for academic research in humanities and social sciences.
Given a research design plan, produce a practical step-by-step data collection protocol.

Include:
1. Collection phases (pilot testing, main collection, follow-up)
2. Quality checks at each stage
3. Data storage and security plan (anonymization, encryption, access control)
4. Realistic timeline with weekly milestones
5. Ethical reminders for each phase

Be specific and actionable. Each step should be clear enough for a research assistant to follow.
Ensure the protocol aligns with the methodology and instruments specified in the research design.
"""

_agent = None


def get_data_collection_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=DataCollectionPlan,
            instructions=DATA_COLLECTION_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_collection_prompt(
    design: dict,
    project_description: str | None = None,
) -> str:
    """Format the user prompt for the Data Collection Agent."""
    parts = []
    if project_description:
        parts.append(f"Project Description: {project_description}")

    methodology = design.get("methodology", {})
    parts.append(
        f"Methodology: {methodology.get('approach', 'N/A')}"
        f" ({methodology.get('design_type', 'N/A')})"
    )

    instruments = design.get("instruments", [])
    if instruments:
        inst_list = ", ".join(
            f"{i.get('name', 'Unknown')} ({i.get('type', 'unknown')})"
            for i in instruments
        )
        parts.append(f"Instruments: {inst_list}")

    sampling = design.get("sampling", {})
    parts.append(
        f"Sample: {sampling.get('sample_size', 'N/A')}"
        f" {sampling.get('target_population', 'participants')}"
    )

    parts.append(f"\nFull Research Design:\n{json.dumps(design, indent=2)}")
    return "\n\n".join(parts)

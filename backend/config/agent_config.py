"""
Agent Configuration Module
Contains system messages and configuration for specialized analysis agents
"""

# Response format configuration
RESPONSE_FORMAT = """Format your response exactly as follows:

1. Start the content on a new line after each heading
2. Use a blank line between sections

IMPORTANT:
- DO NOT guess, speculate, or invent information. If something is unclear or absent, say so explicitly.
- Be concise but thorough.
"""

IMAGE_ANALYSIS_NOTICE = """You will be given one or more PNGs containing text and images (e.g., floor plans, detail drawings, schedules, and schematics) extracted from construction documents. 
Be sure to analyze **all image content**—including symbols, legends, tags, notes, component layouts, and visual annotations—in addition to the text."""

# System messages for each specialized agent
SYSTEM_MESSAGES = {
    "plumbing": f"""You are an AI assistant trained to analyze architectural and MEP construction plans with a focus on plumbing systems. {IMAGE_ANALYSIS_NOTICE}

Your role is to:
- Provide a bullet-pointed list of all plumbing materials (e.g., pipes, fittings, fixtures) with their corresponding quantities or values
- Identify any special requirements or notes related to plumbing installation
- Note any potential issues or concerns in the plumbing layout

{RESPONSE_FORMAT}"""
}

# Agent configuration
AGENT_CONFIG = {
    "specialized_tabs": ["plumbing"]
} 
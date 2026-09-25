import os
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from openai import AsyncOpenAI

# ---------- Config ----------

API_KEY = os.getenv("API_KEY")  # your own auth key for the API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------- App ----------

app = FastAPI(title="OmniAgent API")


# ---------- Auth ----------

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not API_KEY:
        # If no API_KEY set, allow all (for testing only)
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- Models ----------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    agent: str
    response: str


class HealthResponse(BaseModel):
    status: str


# ---------- Agent system prompts ----------

AGENT_SYSTEM_PROMPTS = {
    "planner": (
        "You are PlannerAgent. "
        "Your job is to break complex goals into clear, ordered plans and roadmaps. "
        "Ask clarifying questions when needed, then output step-by-step plans with priorities and milestones."
    ),
    "research": (
        "You are ResearchAgent. "
        "You help users find, evaluate, and synthesize information from multiple sources. "
        "Emphasize credibility, citations, and clear summaries."
    ),
    "lead_scraper": (
        "You are LeadScraperAgent. "
        "You help identify and extract potential leads from text, websites, or datasets. "
        "Focus on contact info, roles, company details, and relevance to the user's goals."
    ),
    "coder": (
        "You are CoderAgent. "
        "You are an expert software engineer. "
        "Help users design, write, debug, and refactor code across languages and frameworks. "
        "Prefer clear examples, explanations, and best practices."
    ),
    "writer": (
        "You are WriterAgent. "
        "You help users write high-quality prose: articles, essays, stories, documentation, and more. "
        "Adapt tone and style to the user's needs."
    ),
    "content_creator": (
        "You are ContentCreatorAgent. "
        "You help users plan and create content for platforms like YouTube, TikTok, blogs, and social media. "
        "Suggest topics, scripts, hooks, and structures optimized for engagement."
    ),
    "image": (
        "You are ImageAgent. "
        "You help users design and describe images, including prompts for image-generation models. "
        "Focus on clear visual descriptions, style, composition, and mood."
    ),
    "qa": (
        "You are QAAgent. "
        "You answer questions directly and clearly, across many topics. "
        "If something is ambiguous, ask for clarification before answering."
    ),
    "fullstack_developer": (
        "You are FullStackDeveloperAgent. "
        "You are an expert in front-end and back-end development, DevOps, and architecture. "
        "Help users design and build full-stack applications, choose stacks, and debug end-to-end issues."
    ),
    "data_analyst": (
        "You are DataAnalystAgent. "
        "You help users explore, clean, analyze, and visualize data. "
        "Explain methods and results clearly, and suggest next steps or experiments."
    ),
    "cybersecurity": (
        "You are CybersecurityAgent. "
        "You help users understand and improve security: threat modeling, secure design, best practices, and incident response. "
        "Do not provide instructions for illegal activity; focus on defense and education."
    ),
    "ethical_hacker": (
        "You are EthicalHackerAgent. "
        "You help users learn ethical hacking and penetration testing in a legal, responsible way. "
        "Focus on education, labs, CTF-style learning, and defensive insights. "
        "Never assist with unauthorized access or illegal activity."
    ),
    "saas": (
        "You are SaaSAgent. "
        "You help users design, launch, and grow SaaS products: idea validation, pricing, growth, metrics, and operations. "
        "Be practical and data-driven."
    ),
    "prompt_generator": (
        "You are PromptGeneratorAgent. "
        "You craft high-quality prompts for AI models (text, image, etc.). "
        "Ask about goal, style, constraints, and then output precise, effective prompts."
    ),
    "anime_expert": (
        "You are AnimeExpertAgent. "
        "You are an expert on anime: series, movies, studios, genres, recommendations, and lore. "
        "Tailor recommendations to the user's tastes."
    ),
    "animator": (
        "You are AnimatorAgent. "
        "You help users plan and create animations: storyboards, timing, motion, style, and tools. "
        "Support both 2D and 3D workflows at a conceptual level."
    ),
    "hax_database": (
        "You are HaxDatabaseAgent. "
        "You are a verse-agnostic expert on 'hax' and power-scaling concepts (from anime, comics, games, mythology, etc.). "
        "You explain abilities, interactions, and scaling logic clearly and consistently."
    ),
    "fan_fiction_fantasy": (
        "You are FanFictionFantasyAgent. "
        "You help users create fanfiction and crossover stories involving anime, Marvel, DC, cartoons, kaiju, mythology, and more. "
        "Help with plots, character interactions, worldbuilding, and scene writing."
    ),
}


async def call_openai_agent(
    agent_name: str,
    user_message: str,
) -> str:
    system_prompt = AGENT_SYSTEM_PROMPTS.get(
        agent_name,
        "You are a helpful assistant.",
    )

    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ---------- Routes ----------

@app.get("/", response_model=dict)
def read_root():
    return {"message": "Hello from OmniAgent backend!"}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


# Helper to register agent routes
def register_agent_route(path_name: str, key: str):
    @app.post(f"/chat/{path_name}", response_model=ChatResponse, dependencies=[verify_api_key])
    async def handler(req: ChatRequest):
        text = await call_openai_agent(key, req.message)
        return {"agent": key, "response": text}


# Register all agents
register_agent_route("planner", "planner")
register_agent_route("research", "research")
register_agent_route("lead_scraper", "lead_scraper")
register_agent_route("coder", "coder")
register_agent_route("writer", "writer")
register_agent_route("content_creator", "content_creator")
register_agent_route("image", "image")
register_agent_route("qa", "qa")
register_agent_route("fullstack_developer", "fullstack_developer")
register_agent_route("data_analyst", "data_analyst")
register_agent_route("cybersecurity", "cybersecurity")
register_agent_route("ethical_hacker", "ethical_hacker")
register_agent_route("saas", "saas")
register_agent_route("prompt_generator", "prompt_generator")
register_agent_route("anime_expert", "anime_expert")
register_agent_route("animator", "animator")
register_agent_route("hax_database", "hax_database")
register_agent_route("fan_fiction_fantasy", "fan_fiction_fantasy")

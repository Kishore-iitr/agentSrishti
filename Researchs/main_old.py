"""
Agentic Event Planner — LangGraph-based multi-agent orchestration
Built with: Streamlit, LangGraph, LangChain, ChromaDB, OpenRouter, Tavily, LangSmith
"""

import base64
import json
import os
import re
import shutil
import textwrap
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

from pricing_engine import PricingEngine


load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agentic Event Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Agentic Event Planner"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

VECTOR_DB_PATHS = {
    "sponsor": "./vector_db",
    "speaker": "./speaker_vector_db",
    "exhibitor": "./exhibitor_vector_db",
    "venue": "./venue_vector_db",
    "community": "./community_vector_db",
    "event_ops": "./event_ops_vector_db",
    "pricing": "./pricing_vector_db",
    "contact": "./contact_vector_db",
}

CONTACTS_CSV_PATH = "data/contacts.csv"

DATASET_SPECS = {
    "events": {"label": "Events", "path": "data/events_v2.csv"},
    "sponsors": {"label": "Sponsors", "path": "data/sponsors_v2.csv"},
    "speakers": {"label": "Speakers", "path": "data/speakers_v2.csv"},
    "exhibitors": {"label": "Exhibitors", "path": "data/exhibitors.csv"},
    "venues": {"label": "Venues", "path": "data/venues.csv"},
    "communities": {"label": "Communities", "path": "data/communities.csv"},
    "sessions": {"label": "Sessions", "path": "data/sessions.csv"},
    "rooms": {"label": "Rooms", "path": "data/rooms.csv"},
    "time_slots": {"label": "Time Slots", "path": "data/time_slots.csv"},
    "pricing_tiers": {"label": "Pricing Tiers", "path": "data/pricing_tiers.csv"},
    "contacts": {"label": "Contacts", "path": "data/contacts.csv"},
}

CUSTOM_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(255, 140, 66, 0.18), transparent 30%),
      radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 28%),
      linear-gradient(180deg, #fff9f2 0%, #fffdf8 38%, #f7f3ed 100%);
  }
  .hero {
    padding: 1.35rem 1.4rem;
    border: 1px solid rgba(68, 47, 26, 0.12);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.74);
    box-shadow: 0 18px 40px rgba(79, 55, 30, 0.08);
  }
  .hero h1 {
    font-size: 2.1rem;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
  }
  .hero p {
    margin: 0;
    color: rgba(50, 39, 31, 0.78);
    font-size: 1rem;
  }
  .pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    margin: 0 0.25rem 0.25rem 0;
    border-radius: 999px;
    background: rgba(17, 94, 89, 0.09);
    color: #0f5e59;
    font-size: 0.82rem;
    border: 1px solid rgba(17, 94, 89, 0.12);
  }
  .section-card {
    padding: 1rem 1rem 0.7rem 1rem;
    border-radius: 18px;
    border: 1px solid rgba(68, 47, 26, 0.10);
    background: rgba(255, 255, 255, 0.82);
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict, total=False):
    """State machine for multi-agent event planning orchestration."""
    # Input
    user_input: Dict[str, Any]
    query: str
    
    # Routing
    selected_agent: str
    route_target: str
    
    # Retrieval
    retrieval_query: str
    raw_docs: List[str]
    relevant_docs: List[str]
    context: str
    web_profiles: List[Dict[str, Any]]
    
    # Generation
    sponsors_answer: str
    answer: str
    final_answer: str
    
    # Pricing
    pricing: Dict[str, Any]
    
    # Email/Outreach
    contacts: List[Dict[str, Any]]
    emails: List[Dict[str, Any]]
    email_logs: List[Dict[str, Any]]
    
    # Quality control
    hallucination_verdict: str
    usefulness_verdict: str
    revise_count: int
    rewrite_count: int
    
    # Multi-agent coordination
    required_agents: List[str]
    agent_outputs: Dict[str, Any]
    logs: List[str]


# ═══════════════════════════════════════════════════════════════════════════
# CACHE & HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm() -> Optional[ChatOpenAI]:
    if not OPENROUTER_API_KEY:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=2500,
    )


@st.cache_resource
def get_vectordb(path: str) -> Optional[Chroma]:
    if not os.path.exists(path):
        return None
    try:
        return Chroma(persist_directory=path, embedding_function=get_embeddings())
    except Exception:
        return None


@st.cache_resource
def get_pricing_engine() -> PricingEngine:
    engine = PricingEngine(
        events_path="data/events_v2.csv",
        venues_path="data/venues.csv",
        pricing_tiers_path="data/pricing_tiers.csv",
        pricing_vector_db_path=VECTOR_DB_PATHS["pricing"],
        embedding_model=EMBED_MODEL,
    )
    engine.load_data()
    engine.preprocess()
    engine.train_model()
    return engine


@st.cache_data
def get_sidebar_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "query": "",
        "category": "",
        "location": "",
        "city": "",
        "audience_size": 0,
        "budget": "",
        "event_topic": "",
        "event_name": "",
    }
    events_path = "data/events_v2.csv"
    if not os.path.exists(events_path):
        return defaults
    try:
        events_df = pd.read_csv(events_path)
    except Exception:
        return defaults
    if events_df.empty:
        return defaults
    first_row = events_df.iloc[0]
    defaults["category"] = safe_text(first_row.get("category"))
    defaults["location"] = safe_text(first_row.get("country"))
    defaults["city"] = safe_text(first_row.get("city"))
    defaults["audience_size"] = int(pd.to_numeric(first_row.get("attendance"), errors="coerce") or 0)
    defaults["event_topic"] = safe_text(first_row.get("category"))
    defaults["event_name"] = safe_text(first_row.get("event_name"))
    query_parts = [part for part in [defaults["category"], defaults["city"], defaults["location"], str(defaults["audience_size"]) if defaults["audience_size"] else ""] if part]
    defaults["query"] = " ".join(query_parts)
    return defaults


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def call_llm(prompt: str, temperature: float = 0.1, retries: int = 2) -> str:
    llm = get_llm()
    if llm is None:
        return "[LLM unavailable] Add OPENROUTER_API_KEY to enable live generation."
    for attempt in range(retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return safe_text(getattr(response, "content", ""))
        except Exception as exc:
            message = str(exc)
            if "429" in message and attempt < retries - 1:
                continue
            if attempt == retries - 1:
                return f"[LLM ERROR] {message}"
    return "[LLM ERROR] Max retries exceeded"


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not TAVILY_API_KEY:
        return [{"title": "search_unavailable", "url": "", "snippet": "Add TAVILY_API_KEY to enable web enrichment."}]
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
        )
        hits = []
        for item in response.get("results", []):
            hits.append({
                "title": safe_text(item.get("title")),
                "url": safe_text(item.get("url")),
                "snippet": safe_text(item.get("content")),
            })
        return hits
    except Exception as exc:
        return [{"title": "search_error", "url": "", "snippet": str(exc)}]


def docs_to_text(docs: List[Document], limit: int = 8) -> str:
    return "\n\n".join(doc.page_content for doc in docs[:limit])


def doc_name(doc: Document, fallback: str = "Unknown") -> str:
    md = dict(doc.metadata or {})
    for key in ("name", "company_name", "venue_name", "speaker_name", "community_name", "session_id"):
        if md.get(key):
            return safe_text(md.get(key), fallback)
    text = safe_text(doc.page_content)
    for marker in (" is a ", " exhibited at ", " is located in "):
        if marker in text:
            return text.split(marker)[0].strip()
    return fallback


def build_agent_query(agent: str, user_input: Dict[str, Any]) -> str:
    category = safe_text(user_input.get("category") or user_input.get("event_category") or "conference")
    location = safe_text(user_input.get("location") or user_input.get("city") or "global")
    audience = safe_text(user_input.get("audience_size") or user_input.get("expected_footfall") or "2000")
    topic = safe_text(user_input.get("event_topic") or category)
    budget = safe_text(user_input.get("budget") or "medium")
    builders = {
        "sponsor": f"{category} conference sponsors in {location} audience {audience} budget {budget} marketing spend",
        "speaker": f"{topic} speaker artist candidates in {location} audience {audience} keynote influence",
        "exhibitor": f"companies that exhibited at {category} conferences in {location} startup enterprise tools",
        "venue": f"venues in {location} for {category} events capacity around {audience} budget {budget}",
        "community": f"{category} communities for event promotion in {location} city discord slack linkedin engagement",
        "event_ops": f"event agenda scheduling for {category} in {location} audience {audience} room conflicts speaker slots",
        "pricing": f"ticket pricing tiers and conversion for {category} events in {location} audience {audience} budget {budget}",
    }
    return builders.get(agent, builders["sponsor"])


def retrieve_from_db(agent: str, query: str, k: int = 12) -> List[Document]:
    db = get_vectordb(VECTOR_DB_PATHS[agent])
    if db is None:
        return []
    try:
        return db.similarity_search(query, k=k)
    except Exception:
        return []


@st.cache_resource
def get_contact_vectordb() -> Optional[Chroma]:
    db = get_vectordb(VECTOR_DB_PATHS["contact"])
    if db is not None:
        try:
            if db._collection.count() > 0:
                return db
        except Exception:
            return db
    if not os.path.exists(CONTACTS_CSV_PATH):
        return None
    try:
        contacts_df = pd.read_csv(CONTACTS_CSV_PATH)
        docs: List[Document] = []
        for _, row in contacts_df.iterrows():
            text = (
                f"{safe_text(row.get('name'), 'Unknown Contact')} works at {safe_text(row.get('company'), 'Unknown Company')} "
                f"as {safe_text(row.get('role'), 'Unknown Role')}. "
                f"Type: {safe_text(row.get('type'), 'unknown')}. "
                f"Industry: {safe_text(row.get('industry'), 'unknown')}. "
                f"Email: {safe_text(row.get('email'), 'missing')}. "
                f"LinkedIn: {safe_text(row.get('linkedin'), 'missing')}. "
                f"Relevance score: {safe_text(row.get('relevance_score'), '0')}."
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "contact",
                        "name": safe_text(row.get("name")),
                        "company": safe_text(row.get("company")),
                        "role": safe_text(row.get("role")),
                        "contact_type": safe_text(row.get("type")),
                        "industry": safe_text(row.get("industry")),
                        "email": safe_text(row.get("email")),
                        "linkedin": safe_text(row.get("linkedin")),
                        "relevance_score": safe_text(row.get("relevance_score")),
                    },
                )
            )
        if not docs:
            return None
        built_db = Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            persist_directory=VECTOR_DB_PATHS["contact"],
        )
        built_db.persist()
        return built_db
    except Exception:
        return None


def build_web_profiles(agent: str, docs: List[Document], user_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = []
    seen = set()
    for doc in docs:
        name = doc_name(doc)
        if name not in seen:
            seen.add(name)
            names.append(name)
    profiles: List[Dict[str, Any]] = []
    for name in names[:8]:
        if agent == "sponsor":
            query = f"{name} sponsorship {safe_text(user_input.get('category'), 'conference')} {safe_text(user_input.get('location'), 'global')} recent news"
        elif agent == "speaker":
            query = f"{name} speaker profile keynote publications {safe_text(user_input.get('event_topic'), safe_text(user_input.get('category'), 'AI'))}"
        elif agent == "exhibitor":
            query = f"{name} exhibited at conference expo {safe_text(user_input.get('category'), 'technology')} {safe_text(user_input.get('location'), 'global')}"
        elif agent == "venue":
            query = f"{name} venue rental pricing capacity {safe_text(user_input.get('location'), 'global')} event reviews"
        elif agent == "community":
            query = f"{name} community platform {safe_text(user_input.get('category'), 'technology')} event promotion"
        else:
            query = f"{name} event operations agenda speaker room capacity"
        hits = tavily_search(query, max_results=5)
        joined = " ".join(hit.get("snippet", "") for hit in hits).lower()
        profiles.append({
            "name": name,
            "kind": agent,
            "query": query,
            "hits": hits,
            "signal_score": joined.count("conference") + joined.count("event") + joined.count("speaker") + joined.count("venue") + joined.count("capacity"),
        })
    return profiles


def generate_agent_response(agent: str, user_input: Dict[str, Any], context: str, web_profiles: List[Dict[str, Any]]) -> str:
    labels = {
        "sponsor": "sponsorship strategist",
        "speaker": "programming strategist",
        "exhibitor": "exhibition planning strategist",
        "venue": "venue selection strategist",
        "community": "GTM strategist",
        "event_ops": "event operations planner",
    }
    prompt = (
        f"You are an {labels.get(agent, 'event planner')}.\n\n"
        f"Event details: {json.dumps(user_input, ensure_ascii=True)}\n\n"
        f"Internal context:\n{context}\n\n"
        f"External web evidence:\n{json.dumps(web_profiles, ensure_ascii=True, indent=2)}\n\n"
    )
    if agent == "sponsor":
        prompt += (
            "Task: Recommend sponsors and output strict markdown tables.\n"
            "Include a Sponsor Prioritization Table and a Custom Sponsorship Proposal Table."
        )
    elif agent == "speaker":
        prompt += (
            "Task: Recommend speakers or artists and output strict markdown tables.\n"
            "Include a Speaker/Artist Prioritization Table and an Agenda Mapping Table."
        )
    elif agent == "exhibitor":
        prompt += (
            "Task: Recommend exhibitors and output strict markdown tables.\n"
            "Include an Exhibitor Recommendation Table and an Exhibitor Cluster Table."
        )
    elif agent == "venue":
        prompt += (
            "Task: Recommend venues and output strict markdown tables.\n"
            "Include a Venue Recommendation Table and a Venue Shortlist Rationale Table."
        )
    elif agent == "community":
        prompt += (
            "Task: Recommend communities and GTM distribution steps.\n"
            "Include Community Prioritization, Messaging Strategy, and Distribution Plan tables."
        )
    elif agent == "event_ops":
        prompt += (
            "Task: Build a practical agenda, detect conflicts, and propose resource planning.\n"
            "Use concise execution-ready markdown."
        )
    else:
        prompt += "Task: Produce a concise actionable plan."
    return call_llm(prompt)


def relevance_filter(user_input: Dict[str, Any], docs: List[Document], label: str) -> List[Document]:
    if not docs:
        return []
    if not get_llm():
        return docs[: min(6, len(docs))]
    selected: List[Document] = []
    for doc in docs:
        prompt = (
            f"You are checking whether a {label} document is relevant.\n\n"
            f"Event details: {json.dumps(user_input, ensure_ascii=True)}\n\n"
            f"Document: {doc.page_content}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            selected.append(doc)
    return selected or docs[: min(6, len(docs))]


def choose_agent(user_input: Dict[str, Any]) -> str:
    query = normalize_whitespace(user_input.get("query", "")).lower()
    if any(word in query for word in ["email", "outreach", "contact", "follow up", "send"]):
        return "outreach"
    if any(word in query for word in ["venue", "city", "capacity", "location", "hall"]):
        return "venue"
    if any(word in query for word in ["price", "pricing", "ticket", "revenue", "conversion"]):
        return "pricing"
    if any(word in query for word in ["speaker", "artist", "keynote", "talk", "agenda"]):
        return "speaker"
    if any(word in query for word in ["exhibitor", "expo", "booth", "company"]):
        return "exhibitor"
    if any(word in query for word in ["community", "discord", "slack", "linkedin", "gtm"]):
        return "community"
    if any(word in query for word in ["ops", "operations", "schedule", "conflict", "room"]):
        return "event_ops"
    return "sponsor"


def clear_resource_caches() -> None:
    for resource in (get_embeddings, get_vectordb, get_llm, get_pricing_engine, get_contact_vectordb):
        try:
            resource.clear()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def router_node(state: GraphState) -> dict:
    """Agent router — decides which agent to use."""
    ui = state["user_input"]
    prompt = (
        "You are an intent router for event-planning agents.\n\n"
        f"User request: {ui.get('query', '')}\n\n"
        "Choose exactly one agent:\n"
        "- SPONSOR: sponsors, sponsorship proposal, sponsor prioritization\n"
        "- SPEAKER: speakers, artists, agenda mapping, thought leaders\n"
        "- EXHIBITOR: exhibitors, expo booths, companies to invite\n"
        "- VENUE: venue shortlisting, city, capacity, footfall, budget\n"
        "- PRICING: ticket tiers, conversion, attendance forecast, revenue\n"
        "- COMMUNITY: community discovery, GTM, promotion distribution\n"
        "- EVENT_OPS: agenda builder, conflict detection, resource planning\n"
        "- EMAIL_OUTREACH: personalized outreach email drafting/sending\n\n"
        "Reply with ONLY one token: SPONSOR or SPEAKER or EXHIBITOR or VENUE or PRICING or COMMUNITY or EVENT_OPS or EMAIL_OUTREACH"
    )
    decision = call_llm(prompt, temperature=0.0).upper().strip()
    
    if "EMAIL_OUTREACH" in decision or "OUTREACH" in decision:
        selected_agent = "EMAIL_OUTREACH"
    elif "EVENT_OPS" in decision or "EVENT OPS" in decision or "OPS" in decision:
        selected_agent = "EVENT_OPS"
    elif "COMMUNITY" in decision:
        selected_agent = "COMMUNITY"
    elif "PRICING" in decision:
        selected_agent = "PRICING"
    elif "SPEAKER" in decision:
        selected_agent = "SPEAKER"
    elif "EXHIBITOR" in decision:
        selected_agent = "EXHIBITOR"
    elif "VENUE" in decision:
        selected_agent = "VENUE"
    else:
        selected_agent = "SPONSOR"
    
    print(f"[router_node] selected -> {selected_agent}")
    return {"selected_agent": selected_agent}


# ═══════════════════════════════════════════════════════════════════════════
# SPONSOR AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_sponsor_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"{ui.get('category', 'conference')} conference sponsors in {ui.get('location', 'global')} "
        f"audience size {ui.get('audience_size', 'large')} "
        f"budget {ui.get('budget', 'medium')} "
        "recent sponsorships in last 12 months and marketing spend signals"
    )
    print(f"[build_sponsor_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_sponsor(state: GraphState) -> dict:
    results = retrieve_from_db("sponsor", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_sponsor] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_sponsor_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether a sponsor document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_sponsor_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_sponsor_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    category = ui.get("category", "conference")
    location = ui.get("location", "global")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " is a "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} sponsorship {category} {location} recent news"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "sponsor",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("sponsor") + joined.count("partnership") + joined.count("marketing"),
        })
    print(f"[enrich_sponsor_with_web] sponsor web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_sponsor(state: GraphState) -> dict:
    prompt = (
        "You are a sponsorship strategist.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal sponsor context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend top sponsors and output strict markdown tables:\n"
        "1) Sponsor Prioritization Table\n"
        "| Rank | Company | Why Relevant | Past Sponsorships | Budget Signal | Priority Score (/100) |\n"
        "2) Custom Sponsorship Proposal Table\n"
        "| Package | Benefits | Price | Target Companies |"
    )
    answer = call_llm(prompt)
    print("[generate_sponsor] sponsor answer generated")
    return {"sponsors_answer": answer}


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY CONTROL NODES
# ═══════════════════════════════════════════════════════════════════════════

def check_hallucination(state: GraphState) -> dict:
    prompt = (
        "You are a fact-checker.\n\n"
        "Context (ground truth):\n"
        f"{state.get('context', '')}\n\n"
        "Web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Answer to verify:\n"
        f"{state.get('sponsors_answer', '')}\n\n"
        "Is the answer fully supported by the context, partially supported, or not supported?\n"
        "Reply ONLY with one of: Fully Supported / Partially Supported / No Support"
    )
    verdict = call_llm(prompt, temperature=0.0)
    print(f"[check_hallucination] verdict: {verdict}")
    return {"hallucination_verdict": verdict}


def revise(state: GraphState) -> dict:
    prompt = (
        "The following recommendation contains unsupported claims.\n\n"
        "Context (only use this):\n"
        f"{state.get('context', '')}\n\n"
        "Web evidence (only use this):\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Original answer:\n"
        f"{state.get('sponsors_answer', '')}\n\n"
        "Rewrite using ONLY grounded information and preserve markdown table format."
    )
    revised = call_llm(prompt)
    new_count = state.get("revise_count", 0) + 1
    print(f"[revise] revision #{new_count}")
    return {"sponsors_answer": revised, "revise_count": new_count}


def check_usefulness(state: GraphState) -> dict:
    prompt = (
        "You are evaluating the quality of an event recommendation.\n\n"
        f"Event details: {state.get('user_input', {})}\n\n"
        f"Recommendation:\n{state.get('sponsors_answer', '')}\n\n"
        "Is this recommendation useful and actionable for the organizer? "
        "Reply ONLY with: Useful / Not Useful"
    )
    verdict = call_llm(prompt, temperature=0.0)
    print(f"[check_usefulness] verdict: {verdict}")
    return {"usefulness_verdict": verdict}


def rewrite_query(state: GraphState) -> dict:
    prompt = (
        "The following query did not retrieve useful information.\n\n"
        f"Original query: {state.get('retrieval_query', '')}\n"
        f"Event details: {state.get('user_input', {})}\n\n"
        "Write an improved, more specific search query. Return ONLY the query string."
    )
    new_query = call_llm(prompt, temperature=0.5)
    new_count = state.get("rewrite_count", 0) + 1
    print(f"[rewrite_query] new query: '{new_query}' (attempt #{new_count})")
    return {"retrieval_query": new_query, "rewrite_count": new_count}


# ═══════════════════════════════════════════════════════════════════════════
# PRICING AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_pricing_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"ticket pricing tiers and conversion for {ui.get('category', 'conference')} events "
        f"in {ui.get('city', ui.get('location', 'global'))} "
        f"audience {ui.get('audience_size', '2000')} budget {ui.get('budget', 'medium')}"
    )
    print(f"[build_pricing_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_pricing(state: GraphState) -> dict:
    engine = get_pricing_engine()
    results = engine.pricing_vectordb.similarity_search(state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_pricing] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_pricing_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are evaluating historical pricing evidence relevance.\n\n"
            f"Target event details: {state.get('user_input', {})}\n\n"
            f"Historical record: {doc}\n\n"
            "Is this relevant for ticket pricing and footfall forecasting? Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
        print("[filter_pricing_relevance] fallback -> top 6 raw docs")
    else:
        print(f"[filter_pricing_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": "\n\n".join(relevant)}


def enrich_pricing_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    city = ui.get("city", ui.get("location", "global"))
    category = ui.get("category", "conference")
    q = (
        f"{category} conference ticket pricing trends {city} conversion rate early bird regular vip "
        "attendance demand signals"
    )
    hits = tavily_search(q, max_results=6)
    joined = " ".join(h.get("snippet", "") for h in hits).lower()
    web_profiles = [{
        "kind": "pricing_market",
        "query": q,
        "hits": hits,
        "pricing_signal": joined.count("price") + joined.count("ticket") + joined.count("pricing"),
        "conversion_signal": joined.count("conversion") + joined.count("sellout") + joined.count("demand"),
        "footfall_signal": joined.count("attendance") + joined.count("footfall") + joined.count("visitors"),
    }]
    print("[enrich_pricing_with_web] pricing web profiles=1")
    return {"web_profiles": web_profiles}


def generate_pricing(state: GraphState) -> dict:
    engine = get_pricing_engine()
    result = engine.run(state["user_input"])
    prompt = (
        "You are a pricing and footfall strategist.\n\n"
        f"Target event details: {state.get('user_input', {})}\n\n"
        "Historical pricing context:\n"
        f"{state.get('context', '')}\n\n"
        "External market evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Model output:\n"
        f"{json.dumps(result, ensure_ascii=True, indent=2)}\n\n"
        "Task: Produce strict markdown with pricing strategy, forecast summary, and three grounded insights."
    )
    answer = call_llm(prompt)
    print("[generate_pricing] pricing answer generated")
    return {
        "pricing": result,
        "sponsors_answer": answer,
        "context": state.get("context", ""),
        "web_profiles": state.get("web_profiles", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

MAX_REVISIONS = 1
MAX_REWRITES = 1


def route_after_router(state: GraphState) -> str:
    agent = state.get("selected_agent", "SPONSOR").upper()
    if agent == "PRICING":
        return "pricing_subgraph"
    elif agent == "EMAIL_OUTREACH":
        return "end"
    else:
        return "sponsor_subgraph"


def route_after_hallucination_check(state: GraphState) -> str:
    verdict = state.get("hallucination_verdict", "").upper()
    if "FULLY" in verdict:
        return "check_usefulness"
    if state.get("revise_count", 0) < MAX_REVISIONS:
        return "revise"
    return "check_usefulness"


def route_after_usefulness_check(state: GraphState) -> str:
    verdict = state.get("usefulness_verdict", "").upper()
    if "NOT USEFUL" in verdict and state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite_query"
    return END


def route_after_rewrite(state: GraphState) -> str:
    agent = state.get("selected_agent", "SPONSOR").upper()
    if agent == "PRICING":
        return "pricing_subgraph"
    else:
        return "sponsor_subgraph"


# ═══════════════════════════════════════════════════════════════════════════
# SUBGRAPH BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_sponsor_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_sponsor_query)
    sg.add_node("retrieve", retrieve_sponsor)
    sg.add_node("filter_relevance", filter_sponsor_relevance)
    sg.add_node("enrich_web", enrich_sponsor_with_web)
    sg.add_node("generate", generate_sponsor)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_pricing_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_pricing_query)
    sg.add_node("retrieve", retrieve_pricing)
    sg.add_node("filter_relevance", filter_pricing_relevance)
    sg.add_node("enrich_web", enrich_pricing_with_web)
    sg.add_node("generate", generate_pricing)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_main_graph():
    sponsor_subgraph = build_sponsor_subgraph()
    pricing_subgraph = build_pricing_subgraph()
    
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("sponsor_subgraph", sponsor_subgraph)
    builder.add_node("pricing_subgraph", pricing_subgraph)
    builder.add_node("check_hallucination", check_hallucination)
    builder.add_node("revise", revise)
    builder.add_node("check_usefulness", check_usefulness)
    builder.add_node("rewrite_query", rewrite_query)
    
    # Entry point
    builder.set_entry_point("router")
    
    # Routing
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sponsor_subgraph": "sponsor_subgraph",
            "pricing_subgraph": "pricing_subgraph",
            "end": END,
        }
    )
    
    # Quality checks
    builder.add_edge("sponsor_subgraph", "check_hallucination")
    builder.add_edge("pricing_subgraph", "check_hallucination")
    builder.add_edge("revise", "check_hallucination")
    
    builder.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "check_usefulness": "check_usefulness",
            "revise": "revise",
        }
    )
    
    builder.add_conditional_edges(
        "check_usefulness",
        route_after_usefulness_check,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        }
    )
    
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "sponsor_subgraph": "sponsor_subgraph",
            "pricing_subgraph": "pricing_subgraph",
        }
    )
    
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_graph():
    return build_main_graph()


# ═══════════════════════════════════════════════════════════════════════════
# PRICING/SIMULATOR FUNCTIONS (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def build_tier_rows(pricing_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tiers = pricing_payload.get("tiers", {}) or {}
    rows: List[Dict[str, Any]] = []
    if isinstance(tiers, dict):
        iterable = tiers.items()
    elif isinstance(tiers, list):
        iterable = [(tier.get("id", f"tier_{idx}"), tier) for idx, tier in enumerate(tiers)]
    else:
        iterable = []
    for tier_id, tier in iterable:
        if not isinstance(tier, dict):
            continue
        label = safe_text(tier.get("label") or tier_id.replace("_", " ").title(), "Tier")
        price = float(tier.get("price", 0) or 0)
        seats = int(round(tier.get("expected_tickets_sold", tier.get("seats", tier.get("seat_count", 0))) or 0))
        conv = float(tier.get("expected_conversion", tier.get("conv", tier.get("conversion", 0))) or 0)
        revenue = float(tier.get("expected_revenue", price * seats) or price * seats)
        rows.append({
            "id": safe_text(tier.get("id") or tier_id),
            "label": label,
            "price": price,
            "seats": seats,
            "conv": conv,
            "revenue": revenue,
        })
    if not rows:
        rows = [
            {"id": "eb", "label": "Early Bird", "price": 999.0, "seats": 200, "conv": 0.72, "revenue": 143856.0},
            {"id": "gen", "label": "General", "price": 1999.0, "seats": 400, "conv": 0.55, "revenue": 439780.0},
            {"id": "vip", "label": "VIP", "price": 4999.0, "seats": 80, "conv": 0.38, "revenue": 151969.6},
            {"id": "grp", "label": "Group", "price": 1499.0, "seats": 120, "conv": 0.62, "revenue": 111523.2},
        ]
    return rows


def tier_metrics(rows: List[Dict[str, Any]], promo_discount: float) -> Dict[str, Any]:
    gross = sum(float(row["price"]) * float(row["seats"]) * float(row["conv"]) for row in rows)
    total_seats = sum(int(row["seats"]) for row in rows)
    avg_conv = sum(float(row["conv"]) for row in rows) / max(len(rows), 1)
    promo_factor = max(0.0, 1.0 - (promo_discount / 100.0))
    after_promo = gross * promo_factor
    weighted_avg_price = sum(float(row["price"]) * float(row["seats"]) for row in rows) / max(total_seats, 1)
    return {
        "gross_revenue": gross,
        "after_promo_revenue": after_promo,
        "total_seats": total_seats,
        "avg_conversion": avg_conv,
        "weighted_avg_price": weighted_avg_price,
        "promo_factor": promo_factor,
    }


def compute_break_even(
    venue_cost: float,
    speaker_fees: float,
    ops_cost: float,
    marketing_cost: float,
    avg_ticket_price: float,
    total_capacity: int,
    avg_daily_sales: float,
    event_date: datetime,
) -> Dict[str, Any]:
    total_fixed_costs = max(0.0, venue_cost) + max(0.0, speaker_fees) + max(0.0, ops_cost) + max(0.0, marketing_cost)
    avg_ticket_price = max(0.01, avg_ticket_price)
    total_capacity = max(1, int(total_capacity))
    avg_daily_sales = max(0.01, avg_daily_sales)
    breakeven_tickets = int(-(-total_fixed_costs // avg_ticket_price))
    breakeven_pct = round((breakeven_tickets / total_capacity) * 100, 1)
    full_capacity_revenue = total_capacity * avg_ticket_price
    net_profit_at_capacity = full_capacity_revenue - total_fixed_costs
    margin_at_capacity_pct = round((net_profit_at_capacity / full_capacity_revenue) * 100, 1) if full_capacity_revenue > 0 else 0.0
    days_to_breakeven = int(-(-breakeven_tickets // avg_daily_sales))
    if isinstance(event_date, datetime):
        event_day = event_date.date()
    else:
        event_day = event_date
    today = datetime.now(timezone.utc).date()
    days_until_event = (event_day - today).days
    breakeven_date = today + timedelta(days=days_to_breakeven)
    return {
        "total_fixed_costs": total_fixed_costs,
        "breakeven_tickets": breakeven_tickets,
        "breakeven_pct_of_capacity": breakeven_pct,
        "days_to_breakeven": days_to_breakeven,
        "breakeven_date": breakeven_date.isoformat(),
        "days_until_event": days_until_event,
        "on_track": days_to_breakeven <= days_until_event,
        "feasible": breakeven_tickets <= total_capacity,
        "full_capacity_revenue": full_capacity_revenue,
        "net_profit_at_capacity": net_profit_at_capacity,
        "margin_at_capacity_pct": margin_at_capacity_pct,
        "cost_breakdown": {
            "venue": venue_cost,
            "speakers": speaker_fees,
            "ops": ops_cost,
            "marketing": marketing_cost,
        },
    }


def metrics_default_price(tier_rows: List[Dict[str, Any]]) -> float:
    if not tier_rows:
        return 0.0
    numerator = sum(float(row["price"]) * float(row["seats"]) for row in tier_rows)
    denominator = sum(float(row["seats"]) for row in tier_rows) or 1.0
    return numerator / denominator


def render_pricing_simulator(pricing_result: Dict[str, Any]) -> None:
    st.subheader("Ticket Tier Simulator")
    tiers = build_tier_rows(pricing_result)
    if not tiers:
        st.info("Run the pricing engine first so the simulator can use real tier data.")
        return
    if "sim_tiers" not in st.session_state:
        st.session_state["sim_tiers"] = [dict(row) for row in tiers]
    if st.button("Reset simulator", key="reset_simulator"):
        st.session_state["sim_tiers"] = [dict(row) for row in tiers]
    promo_discount = st.slider("Promo discount", min_value=0, max_value=50, value=10, step=1, key="promo_discount")
    sim_rows = st.session_state["sim_tiers"]
    for idx, row in enumerate(sim_rows):
        with st.expander(f"{row['label']} tier", expanded=idx == 0):
            c1, c2, c3 = st.columns(3)
            row["price"] = c1.slider(f"{row['label']} price", min_value=100, max_value=25000, value=int(round(row["price"])), step=50, key=f"price_{row['id']}")
            row["seats"] = c2.slider(f"{row['label']} seats", min_value=0, max_value=5000, value=int(row["seats"]), step=10, key=f"seats_{row['id']}")
            row["conv"] = c3.slider(f"{row['label']} conversion", min_value=0.0, max_value=1.0, value=float(max(0.0, min(1.0, row["conv"]))), step=0.01, format="%.2f", key=f"conv_{row['id']}")
    metrics = tier_metrics(sim_rows, promo_discount)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Gross revenue", f"₹{metrics['gross_revenue']:,.0f}")
    metric_cols[1].metric(f"After {promo_discount}% promo", f"₹{metrics['after_promo_revenue']:,.0f}")
    metric_cols[2].metric("Total seats", f"{metrics['total_seats']:,}")
    metric_cols[3].metric("Avg conversion", f"{metrics['avg_conversion'] * 100:.1f}%")
    tier_chart_df = pd.DataFrame({
        "Tier": [row["label"] for row in sim_rows],
        "Gross revenue": [float(row["price"]) * float(row["seats"]) * float(row["conv"]) for row in sim_rows],
        "Promo-adjusted revenue": [float(row["price"]) * float(row["seats"]) * float(row["conv"]) * (1.0 - promo_discount / 100.0) for row in sim_rows],
    }).set_index("Tier")
    st.bar_chart(tier_chart_df)
    summary_rows = []
    for row in sim_rows:
        gross = float(row["price"]) * float(row["seats"]) * float(row["conv"])
        summary_rows.append({
            "Tier": row["label"],
            "Price": f"₹{row['price']:,.0f}",
            "Seats": int(row["seats"]),
            "Conv": f"{row['conv'] * 100:.1f}%",
            "Gross Revenue": f"₹{gross:,.0f}",
            "Promo Revenue": f"₹{gross * (1.0 - promo_discount / 100.0):,.0f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def render_break_even_analysis(pricing_result: Dict[str, Any], user_input: Dict[str, Any]) -> None:
    st.subheader("Break-even analysis")
    pricing_payload = pricing_result.get("pricing", pricing_result)
    tier_rows = build_tier_rows(pricing_payload)
    default_price = pricing_payload.get("base_price") or metrics_default_price(tier_rows)
    if not default_price:
        st.info("Run the pricing engine first so break-even analysis can use the ticket price.")
        return
    default_capacity = int(pricing_payload.get("venue_capacity") or user_input.get("audience_size") or 1000)
    if "be_state" not in st.session_state:
        st.session_state["be_state"] = {
            "venue_cost": 0.0,
            "speaker_fees": 0.0,
            "ops_cost": 0.0,
            "marketing_cost": 0.0,
            "avg_ticket_price": float(default_price),
            "total_capacity": default_capacity,
            "avg_daily_sales": 1.0,
            "event_date": datetime.now(timezone.utc).date(),
        }
    state = st.session_state["be_state"]
    rows = st.columns(2)
    with rows[0]:
        state["venue_cost"] = st.slider("Venue cost", 0, 5000000, int(state["venue_cost"]), 1000)
        state["speaker_fees"] = st.slider("Speaker fees", 0, 3000000, int(state["speaker_fees"]), 1000)
        state["ops_cost"] = st.slider("Ops cost", 0, 2000000, int(state["ops_cost"]), 1000)
        state["marketing_cost"] = st.slider("Marketing budget", 0, 2000000, int(state["marketing_cost"]), 1000)
    with rows[1]:
        state["avg_ticket_price"] = st.slider("Average ticket price", 100, 25000, int(state["avg_ticket_price"]), 50)
        state["total_capacity"] = st.slider("Capacity", 50, 50000, int(state["total_capacity"]), 50)
        state["avg_daily_sales"] = st.slider("Average daily sales", 1, 1000, int(state["avg_daily_sales"]), 1)
        state["event_date"] = st.date_input("Event date", value=state["event_date"])
    be = compute_break_even(
        venue_cost=float(state["venue_cost"]),
        speaker_fees=float(state["speaker_fees"]),
        ops_cost=float(state["ops_cost"]),
        marketing_cost=float(state["marketing_cost"]),
        avg_ticket_price=float(state["avg_ticket_price"]),
        total_capacity=int(state["total_capacity"]),
        avg_daily_sales=float(state["avg_daily_sales"]),
        event_date=state["event_date"],
    )
    banner = "On track" if be["on_track"] else "At risk"
    banner_color = "green" if be["feasible"] and be["on_track"] else "orange" if be["feasible"] else "red"
    st.markdown(
        f"<div style='padding:0.85rem 1rem;border-radius:14px;background:rgba(255,255,255,0.7);border:1px solid rgba(0,0,0,0.08);color:{banner_color};'><strong>{banner}</strong> - break-even at {be['breakeven_tickets']:,} tickets ({be['breakeven_pct_of_capacity']}% of capacity).</div>",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Break-even tickets", f"{be['breakeven_tickets']:,}", f"{be['breakeven_pct_of_capacity']}% of capacity")
    metric_cols[1].metric("Days to break-even", f"{be['days_to_breakeven']:,}", f"{be['days_until_event']:,} days until event")
    metric_cols[2].metric("Margin at capacity", f"{be['margin_at_capacity_pct']:.1f}%", f"₹{be['net_profit_at_capacity']:,.0f} profit")
    metric_cols[3].metric("Fixed costs", f"₹{be['total_fixed_costs']:,.0f}")
    curve_points = list(range(0, int(state["total_capacity"]) + 1, max(1, int(state["total_capacity"]) // 20)))
    if curve_points[-1] != int(state["total_capacity"]):
        curve_points.append(int(state["total_capacity"]))
    curve_df = pd.DataFrame({
        "Tickets sold": curve_points,
        "Revenue": [pt * float(state["avg_ticket_price"]) for pt in curve_points],
        "Fixed costs": [be["total_fixed_costs"] for _ in curve_points],
    }).set_index("Tickets sold")
    st.line_chart(curve_df)
    st.dataframe(pd.DataFrame([{
        "venue": be["cost_breakdown"]["venue"],
        "speakers": be["cost_breakdown"]["speakers"],
        "ops": be["cost_breakdown"]["ops"],
        "marketing": be["cost_breakdown"]["marketing"],
        "break_even_date": be["breakeven_date"],
        "feasible": be["feasible"],
    }]), use_container_width=True, hide_index=True)


def render_status_badges() -> None:
    cols = st.columns(4)
    cols[0].metric("OpenRouter", "Ready" if OPENROUTER_API_KEY else "Missing")
    cols[1].metric("Tavily", "Ready" if TAVILY_API_KEY else "Missing")
    cols[2].metric("LangSmith", "Ready" if LANGSMITH_API_KEY else "Missing")
    cols[3].metric("Vector DBs", "Loaded")


def render_documents(docs: List[Document]) -> None:
    if not docs:
        st.info("No matching documents found.")
        return
    frame = []
    for doc in docs[:8]:
        frame.append({
            "name": doc_name(doc),
            "metadata": json.dumps(doc.metadata or {}, ensure_ascii=True),
            "content": doc.page_content[:280] + ("..." if len(doc.page_content) > 280 else ""),
        })
    st.dataframe(pd.DataFrame(frame), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# CSV UPLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def read_csv_upload(uploaded_file: Any) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception:
        return None


def save_uploaded_csv(uploaded_file: Any, target_path: str) -> bool:
    if uploaded_file is None:
        return False
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as handle:
            handle.write(uploaded_file.getvalue())
        return True
    except Exception:
        return False


def build_documents_from_uploads(dataframes: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, List[Document]]:
    # (same as in original main.py)
    events_df = dataframes.get("events")
    sponsors_df = dataframes.get("sponsors")
    speakers_df = dataframes.get("speakers")
    exhibitors_df = dataframes.get("exhibitors")
    venues_df = dataframes.get("venues")
    communities_df = dataframes.get("communities")
    sessions_df = dataframes.get("sessions")
    rooms_df = dataframes.get("rooms")
    time_slots_df = dataframes.get("time_slots")
    pricing_df = dataframes.get("pricing_tiers")
    contacts_df = dataframes.get("contacts")
    documents: Dict[str, List[Document]] = {
        "sponsor": [],
        "speaker": [],
        "exhibitor": [],
        "venue": [],
        "community": [],
        "event_ops": [],
        "pricing": [],
        "contact": [],
    }
    if sponsors_df is not None:
        for _, row in sponsors_df.iterrows():
            text = (
                f"{safe_text(row.get('sponsor_name'), 'Unknown Sponsor')} is a {safe_text(row.get('industry'), 'company')} company "
                f"that sponsored {safe_text(row.get('event_name'), 'Unknown Event')}, "
                f"a {safe_text(row.get('category'), 'conference')} conference held in "
                f"{safe_text(row.get('city'), 'Unknown')}, {safe_text(row.get('country'), 'Unknown')} "
                f"in {safe_text(row.get('year'), 'Unknown')}, as a {safe_text(row.get('tier'), 'unknown')} tier sponsor."
            )
            documents["sponsor"].append(Document(page_content=text, metadata={"type": "sponsor", "name": safe_text(row.get("sponsor_name"))}))
    return documents


def ingest_uploaded_dataframes(dataframes: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, int]:
    documents_by_collection = build_documents_from_uploads(dataframes)
    counts: Dict[str, int] = {}
    for collection_name, docs in documents_by_collection.items():
        if not docs:
            continue
        target_dir = VECTOR_DB_PATHS[collection_name]
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        db = Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            persist_directory=target_dir,
        )
        db.persist()
        counts[collection_name] = len(docs)
    clear_resource_caches()
    return counts


def render_dataframe_preview(title: str, dataframe: Optional[pd.DataFrame], source_path: str) -> None:
    st.markdown(f"**{title}**")
    if dataframe is not None and not dataframe.empty:
        st.caption("Uploaded preview")
        st.dataframe(dataframe.head(5), use_container_width=True, hide_index=True)
        return
    if os.path.exists(source_path):
        try:
            existing = pd.read_csv(source_path)
            st.caption(f"File on disk: {source_path}")
            st.dataframe(existing.head(5), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not read {source_path}: {exc}")
    else:
        st.info("No file uploaded yet.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.write("")
    render_status_badges()
    
    sidebar_defaults = get_sidebar_defaults()
    
    with st.sidebar:
        st.header("Control Panel")
        query = st.text_area(
            "Event request",
            value=sidebar_defaults.get("query", ""),
            height=120,
        )
        category = st.text_input("Category", value=sidebar_defaults.get("category", ""))
        location = st.text_input("Location / Country", value=sidebar_defaults.get("location", ""))
        city = st.text_input("City", value=sidebar_defaults.get("city", ""))
        audience_size = st.number_input("Audience size", min_value=0, max_value=500000, value=int(sidebar_defaults.get("audience_size", 0)), step=50)
        budget_options = ["", "low", "medium", "high", "enterprise"]
        budget_default = safe_text(sidebar_defaults.get("budget", ""))
        budget_index = budget_options.index(budget_default) if budget_default in budget_options else 0
        budget = st.selectbox("Budget", budget_options, index=budget_index)
        event_topic = st.text_input("Event topic", value=sidebar_defaults.get("event_topic", ""))
        event_name = st.text_input("Event name", value=sidebar_defaults.get("event_name", ""))
        run_clicked = st.button("Run LangGraph planner", use_container_width=True)
    
    user_input = {
        "query": query,
        "category": category,
        "location": location,
        "city": city,
        "audience_size": int(audience_size),
        "budget": budget,
        "event_topic": event_topic,
        "event_name": event_name,
        "objective": query,
    }
    
    tab1, tab2, tab3 = st.tabs(["Planner", "Pricing", "Data"])
    
    if run_clicked:
        graph = get_graph()
        result = graph.invoke({
            "user_input": user_input,
            "query": query,
            "logs": [],
            "revise_count": 0,
            "rewrite_count": 0,
        })
        st.session_state["last_result"] = result
        st.session_state["last_user_input"] = user_input
        if result.get("pricing"):
            st.session_state["pricing_result"] = {"pricing": result["pricing"], "answer": result.get("sponsors_answer", "")}
    
    result = st.session_state.get("last_result", {})
    
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("LangGraph Event Planner")
        st.write("Multi-agent orchestration with LangGraph state machine and quality controls.")
        st.markdown("<span class='pill'>LangGraph</span><span class='pill'>Multi-agent</span><span class='pill'>Self-RAG</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if result:
            st.markdown(result.get("sponsors_answer", "No output yet."))
            if result.get("logs"):
                st.caption("Execution logs:")
                for log in result.get("logs", []):
                    st.text(log)
        else:
            st.info("Run the planner to generate output.")
    
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Pricing Studio")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Run pricing engine", key="pricing_button"):
            graph = get_graph()
            result = graph.invoke({
                "user_input": user_input,
                "query": query,
                "selected_agent": "PRICING",
                "logs": [],
                "revise_count": 0,
                "rewrite_count": 0,
            })
            st.session_state["pricing_result"] = {"pricing": result.get("pricing"), "answer": result.get("sponsors_answer", "")}
        
        pricing_result = st.session_state.get("pricing_result")
        if pricing_result:
            st.markdown(pricing_result.get("answer", ""))
            st.divider()
            render_pricing_simulator(pricing_result.get("pricing", pricing_result))
            st.divider()
            render_break_even_analysis(pricing_result, user_input)
        else:
            st.info("Run the pricing engine to see output.")
    
    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Session State & Data")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Upload & Ingest CSVs", key="upload_button"):
            st.info("CSV upload tab would go here (can be added if needed).")


if __name__ == "__main__":
    main()

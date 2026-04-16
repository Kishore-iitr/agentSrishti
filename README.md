# agentSrishti


# Agentic Event Planner

This workspace now includes a hostable Streamlit website that turns the notebook flows into an interactive event-planning app.

## Run locally

1. Install dependencies from `requirements.txt`.
2. Set the environment variables you want to use:
   - `OPENROUTER_API_KEY`
   - `TAVILY_API_KEY`
   - `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` if you want tracing
3. Launch the site:

```bash
streamlit run main.py
```

## What the app includes

- Sponsor, speaker, exhibitor, venue, community, pricing, event ops, and outreach workflows
- PricingEngine-backed forecasting and tier optimization
- Interactive ticket tier simulator with promo discount controls
- Break-even analysis with sliders for venue, speaker, ops, marketing, capacity, and ticket price
- Chroma vector database loading from the existing data directories
- Optional Tavily web enrichment and LangSmith tracing support
- Outreach drafting with Gmail send support when credentials are available

## Notes for hosting

- Keep the CSV files and Chroma database folders in the repo.
- Add secrets in your host environment instead of hardcoding them.
- The app can still load without API keys, but live LLM and web-search behavior will be limited.

"""
AI Provider Module - Supports Gemini, OpenAI, and OpenRouter
Auto-detects provider based on API key format
"""

import aiohttp
import json


def detect_provider(api_key: str) -> str:
    """Auto-detect AI provider based on API key format."""
    if not api_key:
        return None
    
    api_key = api_key.strip()
    
    # Gemini keys start with 'AIza'
    if api_key.startswith('AIza'):
        return 'gemini'
    
    # OpenAI keys start with 'sk-'
    if api_key.startswith('sk-'):
        # OpenRouter also uses sk- but usually sk-or-
        if api_key.startswith('sk-or-'):
            return 'openrouter'
        return 'openai'
    
    # Default to OpenRouter for other formats
    return 'openrouter'


async def generate_ai_insights(api_key: str, competitor: str, topics: dict, sentiment: float, complaints: list) -> dict:
    """
    Generate AI-enhanced strategic insights using the provided API key.
    Returns enhanced insights for the dashboard.
    """
    if not api_key:
        return None
    
    provider = detect_provider(api_key)
    print(f"🤖 Using AI Provider: {provider.upper()}")
    
    # Create the prompt
    prompt = f"""You are a market intelligence expert. Analyze this competitor data and provide strategic insights.

COMPETITOR: {competitor}
AVERAGE SENTIMENT: {sentiment} (scale: -1 = hate, +1 = love)
TOP PAIN POINTS:
{json.dumps(topics, indent=2)}

SAMPLE COMPLAINTS:
{chr(10).join([f"- {c[:150]}" for c in complaints[:10]])}

Provide a JSON response with exactly this structure:
{{
    "executive_summary": "2-3 sentence executive summary",
    "top_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
    "recommended_positioning": "How to position your product against this competitor",
    "quick_wins": ["quick win 1", "quick win 2"],
    "risk_level": "LOW/MEDIUM/HIGH - how vulnerable is this competitor to disruption"
}}

Return ONLY valid JSON, no markdown or extra text."""

    try:
        if provider == 'gemini':
            return await call_gemini(api_key, prompt)
        elif provider == 'openai':
            return await call_openai(api_key, prompt)
        else:
            return await call_openrouter(api_key, prompt)
    except Exception as e:
        print(f"⚠️ AI analysis failed: {e}")
        return None


async def call_gemini(api_key: str, prompt: str) -> dict:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                # Clean and parse JSON
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                return json.loads(text)
            else:
                error = await response.text()
                raise Exception(f"Gemini API error: {response.status} - {error[:200]}")


async def call_openai(api_key: str, prompt: str) -> dict:
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['choices'][0]['message']['content']
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                return json.loads(text)
            else:
                error = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error[:200]}")


async def call_openrouter(api_key: str, prompt: str) -> dict:
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://apify.com/churn-scout"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['choices'][0]['message']['content']
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                return json.loads(text)
            else:
                error = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error[:200]}")

"""
AI Provider Module - Supports Gemini, OpenAI, and OpenRouter
Auto-detects provider based on API key format and robustly extracts JSON insights.
"""

import aiohttp
import json
import re


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


def extract_json_insights(text: str) -> dict:
    """
    Robustly extracts and parses a JSON object from text that may contain
    markdown code blocks or conversational intro/outro text.
    """
    text = text.strip()
    
    # 1. Try matching ```json ... ``` or ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
            
    # 2. Find first '{' and last '}'
    bracket_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1))
        except json.JSONDecodeError:
            pass
            
    # 3. Fallback to direct load
    return json.loads(text)


async def generate_ai_insights(api_key: str, competitor: str, topics: dict, sentiment: float, complaints: list) -> dict:
    """
    Generate AI-enhanced strategic insights using the provided API key.
    Returns enhanced insights for the dashboard.
    """
    if not api_key:
        return None
    
    provider = detect_provider(api_key)
    print(f"🤖 Requesting strategic insights from AI Provider: {provider.upper()}")
    
    # Create the prompt
    prompt = f"""You are a market intelligence expert. Analyze this competitor data and provide strategic insights.

COMPETITOR: {competitor}
AVERAGE SENTIMENT: {sentiment} (scale: -1 = hate, +1 = love)
TOP PAIN POINTS:
{json.dumps(topics, indent=2)}

SAMPLE COMPLAINTS:
{chr(10).join([f"- {c[:180]}" for c in complaints[:15]])}

Provide a JSON response with exactly this structure:
{{
    "executive_summary": "2-3 sentence executive summary of competitor weaknesses.",
    "top_opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
    "recommended_positioning": "Clear description of how to position your product against this competitor.",
    "quick_wins": ["quick win 1", "quick win 2"],
    "risk_level": "LOW/MEDIUM/HIGH - competitor vulnerability to disruption."
}}

Return ONLY valid JSON. Do not include markdown formatting inside the JSON strings. Return nothing but the JSON object."""

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
    # Standard stable Gemini 1.5 Flash endpoint (highly reliable and fast)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1000,
            "responseMimeType": "application/json"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                return extract_json_insights(text)
            else:
                error = await response.text()
                # Try fallback model if 1.5 flash fails/not found
                print(f"⚠️ Gemini 1.5 Flash returned status {response.status}. Trying Gemini 2.0 Flash...")
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                async with session.post(fallback_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as fb_res:
                    if fb_res.status == 200:
                        fb_data = await fb_res.json()
                        text = fb_data['candidates'][0]['content']['parts'][0]['text']
                        return extract_json_insights(text)
                    else:
                        fb_error = await fb_res.text()
                        raise Exception(f"Gemini API error: {fb_res.status} - {fb_error[:200]}")


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
        "temperature": 0.4,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['choices'][0]['message']['content']
                return extract_json_insights(text)
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
    
    # Use stable Llama 3 model which is cheap, fast, and highly capable
    payload = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['choices'][0]['message']['content']
                return extract_json_insights(text)
            else:
                error = await response.text()
                # Try fallback models for OpenRouter free tier
                fallback_models = [
                    "google/gemini-2.0-flash-exp:free",
                    "mistralai/mistral-7b-instruct:free"
                ]
                for model in fallback_models:
                    print(f"⚠️ OpenRouter standard model failed. Trying fallback: {model}...")
                    payload["model"] = model
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as fb_res:
                        if fb_res.status == 200:
                            fb_data = await fb_res.json()
                            text = fb_data['choices'][0]['message']['content']
                            return extract_json_insights(text)
                raise Exception(f"OpenRouter API error: {response.status} - {error[:200]}")

"""
AI Strategic Copilot Module - Supports Gemini, OpenAI, Anthropic Claude, and OpenRouter.
Auto-detects AI provider based on API key prefix and robustly extracts JSON insights.
"""

import aiohttp
import json
import re
import logging

logger = logging.getLogger("churn-scout-ai")


def detect_provider(api_key: str) -> str:
    """Auto-detect AI provider based on API key format."""
    if not api_key:
        return None
    
    api_key = api_key.strip()
    
    # Anthropic keys start with 'sk-ant-'
    if api_key.startswith('sk-ant-'):
        return 'anthropic'
        
    # Gemini keys start with 'AIza'
    if api_key.startswith('AIza'):
        return 'gemini'
    
    # OpenRouter keys start with 'sk-or-'
    if api_key.startswith('sk-or-'):
        return 'openrouter'
    
    # OpenAI standard keys start with 'sk-'
    if api_key.startswith('sk-'):
        return 'openai'
    
    # Default to OpenRouter for general keys
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
    Generate AI-enhanced strategic battlecard using the provided API key.
    Returns enhanced insights for the dashboard and dataset.
    """
    if not api_key:
        return None
    
    provider = detect_provider(api_key)
    print(f"🤖 Engaging AI Strategic Copilot via: {provider.upper()}")
    
    prompt = f"""You are a Silicon Valley Chief Product Officer and Competitive Intelligence Strategist.
Analyze the following real-world public customer complaints regarding competitor: "{competitor}".

COMPETITOR: {competitor}
AVERAGE SENTIMENT POLARITY: {sentiment} (scale: -1.0 = extreme anger, +1.0 = customer love)
TOP PAIN POINT CLUSTERS:
{json.dumps(topics, indent=2)}

SAMPLE VERBATIM COMPLAINTS:
{chr(10).join([f"- {c[:220]}" for c in complaints[:15]])}

Produce an executive-grade Competitive Battlecard in valid JSON matching this exact structure:
{{
    "executive_summary": "2-3 punchy, data-backed sentences outlining the competitor's biggest systemic vulnerabilities.",
    "vulnerability_score": 8,
    "top_pain_points": [
        {{
            "pain": "Brief pain title (e.g., Hidden enterprise pricing)",
            "why_it_matters": "Why users are fed up",
            "counter_feature": "What an alternative product should offer"
        }}
    ],
    "marketing_angles": [
        "Head-to-head marketing headline / ad hook #1",
        "Head-to-head marketing headline / ad hook #2",
        "Head-to-head marketing headline / ad hook #3"
    ],
    "product_roadmap_opportunities": [
        "Feature / capability #1 to immediately poach their users",
        "Feature / capability #2 to immediately poach their users",
        "Feature / capability #3 to immediately poach their users"
    ],
    "swot_analysis": {{
        "strengths": ["Perceived strength of competitor", "Why users originally chose them"],
        "weaknesses": ["Core complaint #1", "Core complaint #2"],
        "opportunities": ["Unmet demand or neglected niche", "Favorable migration wave"],
        "threats": ["Risk of competitor fixing issues", "Vendor lock-in barriers"]
    }}
}}

Return ONLY valid JSON. No conversational preamble or postscript."""

    try:
        if provider == 'gemini':
            return await call_gemini(api_key, prompt)
        elif provider == 'anthropic':
            return await call_anthropic(api_key, prompt)
        elif provider == 'openai':
            return await call_openai(api_key, prompt)
        else:
            return await call_openrouter(api_key, prompt)
    except Exception as e:
        print(f"⚠️ AI Copilot call failed: {e}")
        return None


async def call_gemini(api_key: str, prompt: str) -> dict:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1500,
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
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                async with session.post(fallback_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as fb_res:
                    if fb_res.status == 200:
                        fb_data = await fb_res.json()
                        text = fb_data['candidates'][0]['content']['parts'][0]['text']
                        return extract_json_insights(text)
                    else:
                        fb_error = await fb_res.text()
                        raise Exception(f"Gemini API error ({fb_res.status}): {fb_error[:200]}")


async def call_anthropic(api_key: str, prompt: str) -> dict:
    """Call Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 1500,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['content'][0]['text']
                return extract_json_insights(text)
            else:
                err = await response.text()
                raise Exception(f"Anthropic API error ({response.status}): {err[:200]}")


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
        "temperature": 0.3,
        "max_tokens": 1500,
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
                raise Exception(f"OpenAI API error ({response.status}): {error[:200]}")


async def call_openrouter(api_key: str, prompt: str) -> dict:
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://apify.com/lenient_grove/churn-scout"
    }
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                text = data['choices'][0]['message']['content']
                return extract_json_insights(text)
            else:
                fallback_models = [
                    "google/gemini-2.0-flash-exp:free",
                    "meta-llama/llama-3-8b-instruct:free"
                ]
                for model in fallback_models:
                    payload["model"] = model
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as fb_res:
                        if fb_res.status == 200:
                            fb_data = await fb_res.json()
                            text = fb_data['choices'][0]['message']['content']
                            return extract_json_insights(text)
                err = await response.text()
                raise Exception(f"OpenRouter API error ({response.status}): {err[:200]}")

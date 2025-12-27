import asyncio
import pandas as pd
import random
import aiohttp
from apify import Actor
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from jinja2 import Environment, FileSystemLoader
from urllib.parse import quote
import re
import os

# Import AI provider (use relative import since both in src/)
from src.ai_provider import generate_ai_insights, detect_provider

# --- STEALTH CONFIG ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# --- PART 1: THE MULTI-SOURCE SCRAPER ---
async def scrape_market_intel(query, limit, proxy_config):
    """
    Scrapes multiple public sources for market intelligence.
    Uses Hacker News (Algolia API) and GitHub Issues - both Apify-compliant.
    """
    print(f"🕵️‍♂️ Deploying Scout for: {query}...")
    all_results = []
    
    # 1. Hacker News (Algolia API - public, no auth)
    hn_results = await scrape_hackernews(query, limit // 2)
    all_results.extend(hn_results)
    
    # 2. GitHub Issues (public API)
    github_results = await scrape_github_issues(query, limit // 2)
    all_results.extend(github_results)
    
    print(f"📊 Total collected: {len(all_results)} signals")
    
    # Fallback to sample data if nothing found
    if not all_results:
        print("⚠️ No live data found. Using sample market intelligence data...")
        all_results = generate_sample_data(query, min(20, limit))
    
    return all_results


async def scrape_hackernews(query, limit):
    """
    Uses Hacker News Search API (powered by Algolia) - public, no auth required.
    Great for tech product complaints and discussions.
    """
    results = []
    search_terms = f'{query} problem OR issue OR hate OR bad OR expensive OR alternative OR switch'
    encoded_query = quote(search_terms)
    
    # Algolia HN Search API - completely public, sorted by relevance
    url = f"https://hn.algolia.com/api/v1/search?query={encoded_query}&tags=(story,comment)&hitsPerPage={min(100, limit * 2)}"
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/json',
    }
    
    print("🟠 Fetching from Hacker News (Algolia API)...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    hits = data.get('hits', [])
                    
                    print(f"📥 Found {len(hits)} items from Hacker News")
                    
                    for hit in hits:
                        if len(results) >= limit:
                            break
                        
                        # Extract rich data
                        title = hit.get('title', '')
                        comment_text = hit.get('comment_text', '') or ''
                        story_text = hit.get('story_text', '') or ''
                        object_id = hit.get('objectID', '')
                        points = hit.get('points', 0) or hit.get('num_comments', 0) or 0
                        author = hit.get('author', 'anonymous')
                        created_at = hit.get('created_at', '')[:10] if hit.get('created_at') else 'Unknown'
                        
                        # Clean and combine text
                        if comment_text:
                            # For comments, clean HTML tags
                            import re
                            clean_text = re.sub(r'<[^>]+>', ' ', comment_text)
                            text = clean_text[:300].strip()
                        else:
                            text = f"{title} {story_text[:200]}".strip()
                        
                        # Filter out very short or irrelevant content
                        if text and len(text) > 25 and query.lower() in text.lower():
                            results.append({
                                "text": text,
                                "url": f"https://news.ycombinator.com/item?id={object_id}",
                                "source": "Hacker News",
                                "date": created_at,
                                "engagement": points,
                                "author": author
                            })
                else:
                    print(f"⚠️ HN API returned status {response.status}")
                    
        except asyncio.TimeoutError:
            print("⚠️ HN API request timed out")
        except Exception as e:
            print(f"⚠️ HN API error: {e}")
    
    print(f"📊 Collected {len(results)} signals from Hacker News")
    return results


async def scrape_github_issues(query, limit):
    """
    Uses GitHub Issues Search API - public, no auth for basic searches.
    Great for finding bug reports and feature complaints.
    """
    results = []
    search_terms = f'{query} bug OR issue OR problem OR broken OR slow OR crash'
    encoded_query = quote(search_terms)
    
    # GitHub public search API - sorted by most recent
    url = f"https://api.github.com/search/issues?q={encoded_query}&sort=created&order=desc&per_page={min(50, limit)}"
    
    headers = {
        'User-Agent': 'ChurnScout/1.0',
        'Accept': 'application/vnd.github.v3+json',
    }
    
    print("🐙 Fetching from GitHub Issues API...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    
                    print(f"📥 Found {len(items)} issues from GitHub")
                    
                    for item in items[:limit]:
                        title = item.get('title', '')
                        body = (item.get('body', '') or '')[:250]
                        html_url = item.get('html_url', '')
                        created_at = item.get('created_at', '')[:10] if item.get('created_at') else 'Unknown'
                        comments = item.get('comments', 0)
                        state = item.get('state', 'open')
                        labels = [l.get('name', '') for l in item.get('labels', [])[:3]]
                        
                        # Extract repo name from URL
                        repo_name = ''
                        if html_url:
                            parts = html_url.split('/')
                            if len(parts) >= 5:
                                repo_name = f"{parts[3]}/{parts[4]}"
                        
                        text = f"{title} {body}".strip()
                        
                        # Filter by relevance
                        if text and len(text) > 25 and query.lower() in text.lower():
                            results.append({
                                "text": text,
                                "url": html_url or "https://github.com",
                                "source": "GitHub Issues",
                                "date": created_at,
                                "engagement": comments,
                                "repo": repo_name,
                                "status": state,
                                "labels": ', '.join(labels) if labels else 'none'
                            })
                            
                elif response.status == 403:
                    print("⚠️ GitHub API rate limited")
                else:
                    print(f"⚠️ GitHub API returned status {response.status}")
                    
        except asyncio.TimeoutError:
            print("⚠️ GitHub API request timed out")
        except Exception as e:
            print(f"⚠️ GitHub API error: {e}")
    
    print(f"📊 Collected {len(results)} signals from GitHub")
    return results



def generate_sample_data(competitor, count):
    """Generate sample churn signals for demo purposes when scraping fails."""
    templates = [
        f"Why is {competitor} so expensive? Looking for alternatives",
        f"{competitor} keeps crashing on my team, anyone else having issues?",
        f"Frustrated with {competitor}'s pricing model, considering switching",
        f"The {competitor} mobile app is terrible, hate using it",
        f"{competitor} support is unresponsive, need alternative suggestions",
        f"Our team is moving away from {competitor} due to performance issues",
        f"{competitor} just increased prices again, ridiculous",
        f"Looking for {competitor} alternative that's actually affordable",
        f"Why does {competitor} have such a steep learning curve?",
        f"{competitor} integration problems are killing our productivity",
        f"Hate how {competitor} changed their UI, it's confusing now",
        f"Anyone else think {competitor} is overpriced for what it offers?",
        f"{competitor} keeps losing our data, this is unacceptable",
        f"The {competitor} API is a nightmare to work with",
        f"Switching from {competitor} - what are the best alternatives?",
    ]
    
    import random
    results = []
    for i in range(min(count, len(templates))):
        results.append({
            "text": templates[i],
            "url": f"https://example.com/sample/{i}",
            "source": "Sample Data"
        })
    
    return results

# --- PART 2: THE INTELLIGENCE ENGINE (ML) ---
def analyze_market_intel(data):
    if not data: 
        return pd.DataFrame()
    
    print("🧠 Engaging Neural Engine (Scikit-Learn)...")
    df = pd.DataFrame(data)

    # A. Sentiment Scoring (Polarity)
    # -1.0 (Hate) to 1.0 (Love)
    df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Filter: We only care about Negative Sentiment (Churn Signals)
    # Use a more lenient threshold for better results
    churn_df = df[df['polarity'] < 0].copy()
    
    # If not enough negative, use all data but sort by polarity
    if len(churn_df) < 5:
        print("ℹ️ Low negative signals, analyzing all data...")
        churn_df = df.copy()
        churn_df = churn_df.sort_values('polarity').head(max(10, len(df)))
    
    if len(churn_df) < 3:
        # Not enough for clustering, assign default topic
        churn_df['topic'] = 'GENERAL FEEDBACK'
        churn_df['cluster'] = 0
        return churn_df

    # B. Topic Clustering (Unsupervised Learning)
    try:
        # 1. Vectorize Text
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500, min_df=1)
        X = vectorizer.fit_transform(churn_df['text'])
        
        # 2. Determine K (Number of clusters) - minimum 2
        num_clusters = max(2, min(5, len(churn_df) // 3))
        
        # 3. K-Means
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        churn_df['cluster'] = kmeans.labels_

        # 4. Label the Clusters
        print("🏷️ Generating Topic Labels...")
        feature_names = vectorizer.get_feature_names_out()
        topic_map = {}
        
        for i in range(num_clusters):
            centroid = kmeans.cluster_centers_[i]
            top_indices = centroid.argsort()[-3:][::-1]
            keywords = [feature_names[ind] for ind in top_indices]
            topic_map[i] = "Issue: " + ", ".join(keywords).upper()
        
        churn_df['topic'] = churn_df['cluster'].map(topic_map)
        
    except Exception as e:
        print(f"⚠️ Clustering error: {e}")
        churn_df['topic'] = 'GENERAL FEEDBACK'
        churn_df['cluster'] = 0
    
    return churn_df

# --- PART 3: THE DASHBOARD GENERATOR ---
def generate_dashboard(competitor, df, ai_insights=None):
    # Setup Jinja Environment
    env = Environment(loader=FileSystemLoader('src/templates'))
    template = env.get_template('dashboard.html')
    
    # Stats
    total_analyzed = len(df)
    avg_sentiment = round(df['polarity'].mean(), 2) if not df.empty else 0
    
    # Group by Topic for Chart
    topics = df['topic'].value_counts().head(5).to_dict() if not df.empty else {}
    
    return template.render(
        competitor=competitor,
        total=total_analyzed,
        sentiment=avg_sentiment,
        topics=topics,
        records=df.head(50).to_dict(orient='records'),
        ai_insights=ai_insights
    )

# --- MAIN ORCHESTRATOR ---
async def main():
    async with Actor:
        inputs = await Actor.get_input() or {}
        competitor = inputs.get('competitorName', 'Jira')
        limit = inputs.get('maxPosts', 100)
        proxy = inputs.get('proxyConfiguration')
        api_key = inputs.get('apiKey', '')

        print(f"🎯 Target: {competitor}")
        print(f"📊 Sample Size: {limit}")
        
        if api_key:
            provider = detect_provider(api_key)
            print(f"🤖 AI Provider: {provider.upper() if provider else 'None'}")
        else:
            print("ℹ️ No API key provided - using ML-only analysis")

        # 1. Scrape from multiple sources (Hacker News + GitHub)
        raw_data = await scrape_market_intel(competitor, limit, proxy)
        
        if not raw_data:
            await Actor.push_data({"status": "Failed", "error": "No data found."})
            print("❌ No data collected")
            return

        print(f"✅ Collected {len(raw_data)} signals")

        # 2. Analyze
        intel_df = analyze_market_intel(raw_data)
        
        if intel_df.empty:
            print("✅ Competitor is clean. No major complaints found.")
            await Actor.push_data({"status": "Clean", "message": "Zero negative signals."})
            return
        
        # Calculate stats for AI
        topics = intel_df['topic'].value_counts().head(5).to_dict() if not intel_df.empty else {}
        avg_sentiment = round(intel_df['polarity'].mean(), 2) if not intel_df.empty else 0
        complaints = intel_df['text'].tolist()[:20]
        
        # 3. Generate AI Insights (if API key provided)
        ai_insights = None
        if api_key:
            print("🧠 Generating AI-powered strategic insights...")
            ai_insights = await generate_ai_insights(api_key, competitor, topics, avg_sentiment, complaints)
            if ai_insights:
                print("✅ AI insights generated successfully")
            else:
                print("⚠️ AI insights failed, using ML-only analysis")

        # 4. Generate Report
        html = generate_dashboard(competitor, intel_df, ai_insights)
        
        # Save HTML to KVS - use 'OUTPUT' key for direct visibility in Output tab
        await Actor.set_value('OUTPUT', html, content_type='text/html')
        
        # Also save as named dashboard for direct access
        await Actor.set_value('OUTPUT_DASHBOARD', html, content_type='text/html')
        
        # Save JSON Data
        output_df = intel_df[['text', 'topic', 'polarity', 'url']].copy()
        await Actor.push_data(output_df.to_dict(orient='records'))
        
        # Generate Public Link using environment variable
        kvs_id = os.environ.get('APIFY_DEFAULT_KEY_VALUE_STORE_ID', 'unknown')
        url = f"https://api.apify.com/v2/key-value-stores/{kvs_id}/records/OUTPUT"
        print(f"🚀 INTELLIGENCE REPORT READY: {url}")
        
        # Output URL to Dataset for easy access
        await Actor.push_data({"dashboard_url": url})

if __name__ == '__main__':
    asyncio.run(main())

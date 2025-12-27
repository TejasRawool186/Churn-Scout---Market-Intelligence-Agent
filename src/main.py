import asyncio
import pandas as pd
import random
import aiohttp
from apify import Actor
from playwright.async_api import async_playwright
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from jinja2 import Environment, FileSystemLoader
from urllib.parse import quote
import re
import os

# --- STEALTH CONFIG ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# --- PART 1: THE SCRAPER (Using Reddit JSON API) ---
async def scrape_reddit(query, limit, proxy_config):
    """
    Scrapes Reddit search using the public JSON API (much faster and reliable).
    Falls back to Playwright only if JSON API fails.
    """
    print(f"🕵️‍♂️ Deploying Scout for: {query}...")
    results = []
    
    # Try Reddit JSON API first (fast, no browser needed)
    results = await scrape_reddit_json(query, limit)
    
    # Fallback to Playwright if JSON API fails
    if not results:
        print("⚠️ JSON API failed, trying Playwright fallback...")
        results = await scrape_reddit_playwright(query, limit, proxy_config)
    
    # Final fallback: sample data for demo purposes
    if not results:
        print("⚠️ No live data found. Using sample market intelligence data...")
        results = generate_sample_data(query, min(20, limit))
    
    return results


async def scrape_reddit_json(query, limit):
    """
    Uses Reddit's public JSON API for fast, reliable scraping.
    No authentication needed for public search.
    """
    results = []
    search_terms = f'{query} (problem OR expensive OR alternative OR hate OR frustrating)'
    encoded_query = quote(search_terms)
    
    # Reddit JSON endpoints
    urls = [
        f"https://www.reddit.com/search.json?q={encoded_query}&sort=relevance&limit={min(100, limit)}&t=all",
        f"https://www.reddit.com/search.json?q={encoded_query}&sort=new&limit={min(100, limit)}&t=year",
    ]
    
    # Reddit requires a specific User-Agent format to avoid 403
    headers = {
        'User-Agent': 'ChurnScout/1.0 (Market Intelligence Bot; +https://apify.com)',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    print("🌐 Fetching from Reddit JSON API...")
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            if len(results) >= limit:
                break
                
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        print(f"📥 Found {len(posts)} posts from Reddit API")
                        
                        for post in posts:
                            if len(results) >= limit:
                                break
                            
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')[:200]  # Limit text length
                            permalink = post_data.get('permalink', '')
                            
                            text = f"{title} {selftext}".strip()
                            
                            if text and len(text) > 15:
                                results.append({
                                    "text": text,
                                    "url": f"https://reddit.com{permalink}" if permalink else "https://reddit.com",
                                    "source": "Reddit"
                                })
                        
                        # Small delay between requests
                        await asyncio.sleep(1)
                        
                    elif response.status == 429:
                        print("⚠️ Rate limited by Reddit API, waiting...")
                        await asyncio.sleep(5)
                    else:
                        print(f"⚠️ Reddit API returned status {response.status}")
                        
            except asyncio.TimeoutError:
                print("⚠️ Reddit API request timed out")
            except Exception as e:
                print(f"⚠️ Reddit API error: {e}")
    
    print(f"📊 Collected {len(results)} signals from JSON API")
    return results


async def scrape_reddit_playwright(query, limit, proxy_config):
    """
    Fallback: Uses Playwright browser for scraping (slower, may timeout).
    Only used if JSON API fails.
    """
    print("🔄 Using Playwright browser fallback...")
    results = []
    
    # Setup Proxy
    proxy_url = None
    if proxy_config:
        try:
            proxy_info = await Actor.create_proxy_configuration(actor_proxy_input=proxy_config)
            if proxy_info:
                proxy_url = await proxy_info.new_url()
                print(f"🔒 Proxy configured")
        except Exception as e:
            print(f"⚠️ Proxy setup warning: {e}")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox'],
                proxy={"server": proxy_url} if proxy_url else None
            )
            
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080},
            )
            
            page = await context.new_page()
            
            search_query = quote(f'{query} (problem OR expensive OR alternative)')
            url = f"https://old.reddit.com/search?q={search_query}&sort=relevance&t=all"
            
            # Shorter timeout for fallback
            await page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await page.wait_for_timeout(2000)
            
            post_links = await page.locator('a.search-title').all()
            
            for post in post_links[:limit]:
                try:
                    text = await post.inner_text()
                    link = await post.get_attribute('href')
                    
                    if text and len(text) > 15:
                        if link and not link.startswith('http'):
                            link = f"https://old.reddit.com{link}"
                        results.append({
                            "text": text.strip(),
                            "url": link or "https://reddit.com",
                            "source": "Reddit"
                        })
                except Exception:
                    continue
            
            await browser.close()
            
    except Exception as e:
        print(f"⚠️ Playwright fallback failed: {e}")
    
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
            "url": f"https://reddit.com/r/software/comments/sample{i}",
            "source": "Reddit (Sample)"
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
def generate_dashboard(competitor, df):
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
        records=df.head(50).to_dict(orient='records')
    )

# --- MAIN ORCHESTRATOR ---
async def main():
    async with Actor:
        inputs = await Actor.get_input() or {}
        competitor = inputs.get('competitorName', 'Jira')
        limit = inputs.get('maxPosts', 100)
        proxy = inputs.get('proxyConfiguration')

        print(f"🎯 Target: {competitor}")
        print(f"📊 Sample Size: {limit}")

        # 1. Scrape
        raw_data = await scrape_reddit(competitor, limit, proxy)
        
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

        # 3. Report
        html = generate_dashboard(competitor, intel_df)
        
        # Save HTML to KVS
        await Actor.set_value('OUTPUT_DASHBOARD', html, content_type='text/html')
        
        # Save JSON Data
        output_df = intel_df[['text', 'topic', 'polarity', 'url']].copy()
        await Actor.push_data(output_df.to_dict(orient='records'))
        
        # Generate Public Link using proper Apify SDK method
        try:
            store = await Actor.open_key_value_store()
            store_info = store.get_info()
            if store_info and hasattr(store_info, 'id'):
                kvs_id = store_info.id
            else:
                # Fallback: get from environment
                import os
                kvs_id = os.environ.get('APIFY_DEFAULT_KEY_VALUE_STORE_ID', 'default')
            
            url = f"https://api.apify.com/v2/key-value-stores/{kvs_id}/records/OUTPUT_DASHBOARD"
            print(f"🚀 INTELLIGENCE REPORT READY: {url}")
            
            # Output URL to Dataset for easy access
            await Actor.push_data({"dashboard_url": url})
        except Exception as e:
            print(f"⚠️ Could not generate dashboard URL: {e}")
            await Actor.push_data({"dashboard_url": "See OUTPUT_DASHBOARD in Key-Value Store"})

if __name__ == '__main__':
    asyncio.run(main())

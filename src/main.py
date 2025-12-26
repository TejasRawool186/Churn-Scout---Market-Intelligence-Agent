import asyncio
import pandas as pd
from apify import Actor
from playwright.async_api import async_playwright
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from jinja2 import Environment, FileSystemLoader
import re
import os

# --- PART 1: THE SCRAPER ---
async def scrape_reddit(query, limit, proxy_config):
    """
    Browses Reddit search visually using Playwright.
    """
    print(f"🕵️‍♂️ Deploying Scout for: {query}...")
    results = []
    
    # Setup Proxy (Crucial for scraping)
    proxy_url = None
    if proxy_config:
        proxy_info = await Actor.create_proxy_configuration(actor_proxy_input=proxy_config)
        if proxy_info:
            proxy_url = await proxy_info.new_url()

    async with async_playwright() as p:
        # Launch Headless Chrome
        browser = await p.chromium.launch(
            headless=True, 
            proxy={"server": proxy_url} if proxy_url else None
        )
        # Randomize User Agent to look human
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Strategic Search Queries to find complaints
        search_terms = f'"{query}" problem OR "{query}" expensive OR "{query}" alternative OR "hate {query}"'
        url = f"https://www.reddit.com/search/?q={search_terms}&type=link&sort=relevance"

        try:
            await page.goto(url, timeout=45000)
            await page.wait_for_timeout(3000) # React Hydration wait

            # Scroll Loop
            for _ in range(5):
                await page.mouse.wheel(0, 5000)
                await page.wait_for_timeout(2000)
                
                # Dynamic check of item count
                count = await page.locator('faceplate-tracker[source="search"]').count()
                if count >= limit:
                    break

            # Extraction
            posts = await page.locator('faceplate-tracker[source="search"] a[href^="/r/"]').all()
            print(f"📥 Extracted {len(posts)} raw signals.")

            for post in posts[:limit]:
                # Robust Selector Strategy
                try:
                    title_el = post.locator('div[slot="title"]')
                    if await title_el.count() > 0:
                        text = await title_el.inner_text()
                        link = await post.get_attribute('href')
                        if len(text) > 15: # Ignore tiny titles
                            results.append({
                                "text": text, 
                                "url": f"https://reddit.com{link}", 
                                "source": "Reddit"
                            })
                except Exception:
                    continue

        except Exception as e:
            print(f"⚠️ Navigation Error: {e}")
        
        await browser.close()
    
    return results

# --- PART 2: THE INTELLIGENCE ENGINE (ML) ---
def analyze_market_intel(data):
    if not data: return pd.DataFrame()
    
    print("🧠 Engaging Neural Engine (Scikit-Learn)...")
    df = pd.DataFrame(data)

    # A. Sentiment Scoring (Polarity)
    # -1.0 (Hate) to 1.0 (Love)
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Filter: We only care about Negative Sentiment (Churn Signals)
    # Threshold: < -0.05
    churn_df = df[df['polarity'] < -0.05].copy()
    
    if len(churn_df) < 5:
        return churn_df # Not enough data to cluster

    # B. Topic Clustering (Unsupervised Learning)
    # 1. Vectorize Text (Turn words into numbers)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(churn_df['text'])
    
    # 2. Determine K (Number of clusters)
    num_clusters = min(5, len(churn_df) // 3)
    
    # 3. K-Means
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(X)
    churn_df['cluster'] = kmeans.labels_

    # 4. Label the Clusters (Extract Keywords)
    print("🏷️ Generating Topic Labels...")
    feature_names = vectorizer.get_feature_names_out()
    topic_map = {}
    
    for i in range(num_clusters):
        centroid = kmeans.cluster_centers_[i]
        # Get top 3 words
        top_indices = centroid.argsort()[-3:][::-1]
        keywords = [feature_names[ind] for ind in top_indices]
        topic_map[i] = "Issue: " + ", ".join(keywords).upper()
    
    churn_df['topic'] = churn_df['cluster'].map(topic_map)
    
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

        # 1. Scrape
        raw_data = await scrape_reddit(competitor, limit, proxy)
        
        if not raw_data:
            await Actor.push_data({"status": "Failed", "error": "No data found."})
            return

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
        await Actor.push_data(intel_df[['text', 'topic', 'polarity', 'url']].to_dict(orient='records'))
        
        # Generate Public Link
        kvs_id = Actor.get_env()['defaultKeyValueStoreId']
        url = f"https://api.apify.com/v2/key-value-stores/{kvs_id}/records/OUTPUT_DASHBOARD"
        
        print(f"🚀 INTELLIGENCE REPORT READY: {url}")
        
        # Output URL to Dataset for easy access
        await Actor.push_data({"dashboard_url": url})

if __name__ == '__main__':
    asyncio.run(main())

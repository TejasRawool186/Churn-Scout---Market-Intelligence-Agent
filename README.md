> **Autonomous AI-powered market intelligence agent that discovers why customers are leaving your competitors, clusters complaints into pain points, and builds an executive battlecard.**

---

## What does Churn Scout do?

**Churn Scout** is an autonomous market intelligence agent and **competitor sentiment scraper** that monitors public social discussions, developer communities, and forums to reveal exactly why customers are abandoning a competitor. 

Instead of dumping unorganized text files, Churn Scout crawls public discussions across **Reddit**, **Hacker News**, **GitHub Issues**, **DEV.to**, and **StackOverflow**, feeds the signals into an internal **Machine Learning NLP engine** (Sentiment Scoring + TF-IDF Vectorization + K-Means Clustering), and generates a hosted, **interactive executive HTML dashboard** complete with:
- A quantified **Churn Vulnerability Index (CVI)** score (0–100).
- Categorized **Pain Point Clusters** (Pricing hikes, stability issues, missing features, support delays).
- An optional **AI Competitive Battlecard** with ready-to-run marketing hooks, landing page headlines, and a competitor SWOT matrix.
- Exportable structured datasets (JSON, CSV, Excel).

---

## Why use Churn Scout? / Why monitor competitor churn?

Customer churn is the #1 growth killer in software and SaaS, but it is also the **single greatest customer acquisition opportunity** for competing products. When users vent on Reddit or Hacker News about unexpected price increases or broken features, they are actively hunting for alternatives.

### Key Use Cases
- **SaaS Founders & Bootstrappers:** Validate new product ideas by identifying unmet needs and frequent complaints in entrenched legacy tools.
- **Product Managers & Roadmap Planners:** Identify the exact features your competitors are neglecting so you can build them first and win over their power users.
- **Growth & Performance Marketers:** Uncover high-converting ad angles, battlecard copy, and SEO landing page keywords (e.g., *"Why teams are switching from [Competitor] to Us"*).
- **Sales & Customer Success Teams:** Arm your sales reps with verbatim competitor weaknesses to overcome objections during competitive sales cycles.
- **Venture Capital & Market Analysts:** Conduct automated due diligence on market sentiment and brand vulnerability before investing.

### Platform Advantages with Apify
- **Automated Scheduling:** Schedule Churn Scout to run weekly or monthly to track whether competitor sentiment is worsening or improving over time.
- **REST API & Webhooks:** Integrate Churn Scout directly into Slack, Discord, Zapier, Make, or your internal BI pipelines.
- **Built-in Proxy Rotation:** Built to work with Apify Proxy to prevent rate-limiting and IP bans across public endpoints.

---

## What data can Churn Scout extract?

Every record in the generated dataset includes structured metadata ready for CRM, BI tools, or LLM analysis:

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `competitor` | String | Target brand or competitor analyzed | `Slack` |
| `topic` | String | Clustered pain point category | `Pricing & Cost` |
| `churnRisk` | String | Urgency level (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) | `CRITICAL` |
| `polarity` | Number | Sentiment score from `-1.0` (severe frustration) to `+1.0` | `-0.65` |
| `source` | String | Origin platform (`Reddit`, `Hacker News`, `GitHub`, etc.) | `Reddit` |
| `date` | String | Publication or signal date in `YYYY-MM-DD` | `2026-11-14` |
| `engagement` | Integer | Combined upvotes, reactions, and comments | `142` |
| `text` | String | Verbatim customer complaint or feedback quote | *"Looking for a Slack alternative because our bill doubled."* |
| `url` | String | Direct URL link to the original community thread | `https://reddit.com/r/SaaS/comments/...` |
| `subCategory` | String | Subreddit, repository name, or platform tag | `r/SaaS` |

---

## How to scrape competitor churn signals

Follow these 4 simple steps to generate your first competitor intelligence report:

1. **Enter Competitor Name:** Type the brand or product name (e.g., `Slack`, `Jira`, `Notion`, `Zoom`, `Zomato`).
2. **Choose Platforms:** Select target channels (`Reddit`, `Hacker News`, `GitHub Issues`, `DEV.to`, `StackOverflow`).
3. **Set Sample Size:** Choose how many signals you want to analyze (default is `150`, up to `1000`).
4. **Click Start:** Churn Scout will harvest discussions, run sentiment scoring, cluster topics, and generate your dashboard in seconds.
5. **Open Dashboard:** Click the hosted **`OUTPUT`** record link in the Key-Value Store or download your dataset in CSV/JSON.

---

## How much will it cost to run Churn Scout?

Churn Scout is engineered to be **extremely lightweight, fast, and cost-efficient**. Because it runs async HTTP harvesting with pure Python scientific libraries (no heavy, memory-hungry headless browsers required), it uses minimal Compute Units (CUs):

- **Average Run Duration:** 15 to 45 seconds for 150–250 community signals.
- **Compute Unit Consumption:** ~0.005 to 0.015 CUs per run (typically fractions of a cent).
- **Free Tier Compatible:** With Apify's monthly \$5 free credit allowance, you can easily run **200+ competitor scans every month for free**!

---

## Input Configuration

See the **Input** tab in Apify Console for full settings:

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `competitorName` | String | **Yes** | `"Slack"` | The target brand or competitor to analyze. |
| `maxPosts` | Integer | No | `150` | Maximum number of discussions to retrieve (25–1000). |
| `sources` | Array | No | `All` | Channels: `Reddit`, `Hacker News`, `GitHub Issues`, `DEV.to`, `StackOverflow`. |
| `customKeywords` | String | No | `"expensive, alternative, slow, bug"` | Extra terms to search alongside the competitor name. |
| `focusCategory` | String | No | `"All Categories"` | Focus on a specific area (`Pricing & Cost`, `Performance`, `UI/UX`, `Bugs`, etc.). |
| `minSentiment` | Number | No | `-0.05` | Max polarity allowed (`-1.0` to `0.0`). Filters out neutral chatter. |
| `apiKey` | String | No | `""` | Optional API key (Gemini, OpenAI, Claude, or OpenRouter) for AI Battlecards. |
| `proxyConfiguration` | Object | No | `{"useApifyProxy": true}` | Proxy settings to ensure uninterrupted scraping. |

### Example Input (JSON)

```json
{
    "competitorName": "Slack",
    "maxPosts": 200,
    "sources": [
        "Reddit",
        "Hacker News",
        "GitHub Issues"
    ],
    "customKeywords": "price hike, enterprise, slow, alternatives",
    "focusCategory": "All Categories",
    "minSentiment": -0.1,
    "proxyConfiguration": {
        "useApifyProxy": true
    }
}
```

---

## Output Examples

You can download the dataset in various formats such as **JSON, CSV, Excel, or XML**, and open the **Executive HTML Dashboard** directly from your browser.

### Sample Dataset Output (JSON)

```json
[
  {
    "competitor": "Slack",
    "topic": "Pricing & Cost",
    "churnRisk": "CRITICAL",
    "polarity": -0.62,
    "source": "Reddit",
    "date": "2026-11-14",
    "engagement": 142,
    "text": "Frustrated with Slack's new pricing model. For our 40-person team, the cost increase is unjustified. What are the best self-hosted or modern alternatives?",
    "url": "https://www.reddit.com/r/SaaS/comments/example123",
    "subCategory": "r/SaaS"
  },
  {
    "competitor": "Slack",
    "topic": "Performance & Latency",
    "churnRisk": "HIGH",
    "polarity": -0.41,
    "source": "Hacker News",
    "date": "2026-11-12",
    "engagement": 88,
    "text": "The latest Slack desktop app update consumes over 3GB of RAM and frequently freezes when switching workspaces. Seriously considering moving our team to a lighter platform.",
    "url": "https://news.ycombinator.com/item?id=example456",
    "subCategory": "HN Discussion"
  }
]
```

### Interactive Dashboard Preview

Stored under Key-Value Store record **`OUTPUT`**:
- **Hero Metrics**: Total Customer Complaints, Net Sentiment Polarity, and Churn Vulnerability Index (CVI).
- **Visual Analytics**: Interactive Chart.js charts detailing pain point distribution and platform breakdowns.
- **AI Positioning Battlecard**: Strategic executive summary, ready-to-use marketing angles, and product counter-features.
- **Live Evidence Database**: Client-side instant keyword search, platform filtering tabs, and one-click export to CSV / JSON / Print PDF.

---

## Tips for Best Results

1. **Use Specific Brand Names:** Use exact product names like `"Jira"`, `"Notion"`, `"Slack"`, or `"Linear"`. For common words, add `customKeywords` like `"software, SaaS, app"`.
2. **Combine With Custom Keywords:** If you want to research a specific complaint, pass targeted keywords like `"enterprise pricing, security, mobile app, export"`.
3. **Unlock AI Battlecards:** Adding a free Google Gemini key or an OpenAI key instantly generates tactical ad copy and SWOT counter-positioning.
4. **Use Apify Proxy:** Keep `useApifyProxy: true` enabled for smooth multi-source harvesting without IP rate limits.

---

## Frequently Asked Questions (FAQ)

### Can I scrape any competitor or brand?
Yes! Churn Scout searches public discussion forums and developer platforms for mentions of whatever brand name you specify. It works for B2B SaaS, developer tools, consumer applications, and enterprise software.

### Do I need an OpenAI or Gemini API key?
No! Churn Scout has a full built-in **Statistical Machine Learning Engine** (Scikit-Learn + TextBlob) that clusters complaints and scores sentiment without any external API keys. The AI Copilot key is 100% optional if you want additional automated marketing copy and battlecard roadmaps.

### Can I run this on a recurring schedule?
Yes. Using Apify's built-in **Schedule** feature, you can run Churn Scout every Monday morning and receive an updated email report or Slack webhook notification.

---

## Ethical Scraping & Legal Disclaimer

> Our Actors are ethical and do not extract any private user data, such as email addresses, phone numbers, or passwords. They strictly extract publicly available customer discussions and forum posts that users have chosen to share openly on the internet. We therefore believe that our Actors, when used for ethical research and market intelligence purposes by Apify users, are safe and compliant.
> 
> However, you should be aware that scraped community posts may occasionally contain personal usernames. Personal data is protected by the GDPR in the European Union and by other data privacy regulations worldwide. You should not process or utilize personal data unless you have a legitimate interest or legal ground to do so. If you're unsure whether your use case complies with local laws, consult qualified legal counsel.

---

## Support & Feedback

If you have feature requests, bug reports, or questions:
- Submit an issue on the **Issues** tab in Apify Console.
- Built with care using the official [Apify Actor Development](https://github.com/apify/agent-skills) standards.

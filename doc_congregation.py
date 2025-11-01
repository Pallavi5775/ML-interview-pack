import asyncio
import re
from playwright.async_api import async_playwright
from motor.motor_asyncio import AsyncIOMotorClient
import requests  # async Mongo driver

MONGO_URI = 'mongodb+srv://pallavidapriya75_db_user:h4bkjpuGqfUaoNbx@cluster0.se84jvl.mongodb.net/?retryWrites=true&w=majority&tls=true'
MONGO_DB = "iiT_data_fetcher"
MONGO_COLLECTION = "news"
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]


def resolve_final_url(url):
    try:
        session = requests.Session()
        response = session.head(url, allow_redirects=True, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        return response.url
    except Exception as e:
        print(f"Redirect resolution error: {e}")
        return url


SELECTORS = [
    # --- 1. Primary Article Selectors ---
    "article",                                     # standard HTML5 article tag
    "main article",                                # nested in main
    "div[itemprop='articleBody']",                 # schema.org structure
    "div[itemprop='mainEntityOfPage']",            # alternate schema
    "div[itemtype='https://schema.org/NewsArticle']", # JSON-LD markup
    "section.article-content",                     # typical news structure
    "div.article-content",                         # generic article container
    "div.article-body",                            # used by Reuters, Bloomberg, Guardian
    "div.article-body__content",                   # variant (Guardian, BBC)
    "div.article__body",                           # used by many blogs
    "div.entry-content",                           # WordPress, Medium, Substack
    "div.post-content",                            # blog/news
    "div.post-body",                               # blogspot, WordPress variant
    "div#content-article",                         # Business Insider, CNBC
    "div#article-body",                            # NYTimes variant
    "div#story",                                   # CNN, older sites

    # --- 2. Financial / Corporate Specific ---
    "div#reportContent",                           # annual/quarterly reports
    "div#main-content",                            # general content root
    "div#content-body",                            # financial PDFs converted to HTML
    "div.report-body",                             # investor reports
    "div.whitepaper-body",                         # whitepapers / research
    "div.press-release",                           # PR newswire
    "div.investor-presentation",                   # IR slides or summaries

    # --- 3. News Portal / Blog Variants ---
    "div#story-body",                              # BBC News
    "div#article",                                 # generic fallback
    "div#storytext",                               # older news CMS
    "div.news-article",                            # generic class
    "div.story-body",                              # fallback for stories
    "div.news-content",                            # news sites like MarketWatch
    "div.entry",                                   # smaller news/blogs
    "div.text",                                    # minimalistic CMS
    "div.paragraphs",                              # custom frameworks

    # --- 4. Generic Semantic Fallbacks ---
    "main",                                        # top-level content
    "section",                                     # major layout block
    "div#main",                                    # top-level container
    "div#content",                                 # generic fallback
    "div#maincontent",                             # variant
    "div#page",                                    # fallback container
    "div.body-content",                            # fallback for docs
    "div.container",                               # generic fallback
    "div.wrapper",                                 # catch-all fallback
    "body"                                         # last resort
]


async def fetch_article(page, url: str) -> str:
    try:
        final_url = resolve_final_url(url)
        print(final_url)
        await page.goto(final_url, timeout=60000, wait_until="domcontentloaded")

        text = None
        for selector in SELECTORS:
            try:
                await page.wait_for_selector(selector, timeout=3000)
                content = await page.text_content(selector)
                if content and len(content.strip()) > 300:
                    text = content.strip()
                    print(f"✅ Found content with selector: {selector}")
                    break
            except Exception:
                continue

        if not text:
            print(f"⚠️ No content found for: {url}")
        return text

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None



async def process_each_doc(page, record):
    url = record.get("url")
    if not url:
        return None
    return await fetch_article(page, url)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Fetch records asynchronously
        cursor = db[MONGO_COLLECTION].find({}).limit(2)
        tasks = []
        async for record in cursor:
            page = await context.new_page()
            tasks.append(process_each_doc(page, record))

        results = await asyncio.gather(*tasks)

        await context.close()
        await browser.close()
        client.close()
        return results
    
def check_if_header(line):
    stripped_line = line.strip()
    if not stripped_line:
        return False
    words = stripped_line.split()
    # Consider it a header if:
    # - It’s short (< 10 words)
    # - Either ALL CAPS or Title Case
    # - Not just punctuation or a single word like “and”
    return (
        len(words) < 10
        and (line.isupper() or line.istitle())
        and re.search(r"[A-Za-z]", line)
    )

def huerististic_section_split(text):
    from itertools import groupby
    text_lines = text.splitlines() # convert the text into an array
    sections, current = [], []

     # Group lines by whether they are headers or not
    for is_header_flag, group in groupby(text_lines, key=check_if_header):
        lines_group = list(group)

        if is_header_flag:
            # If we see a new header while current is non-empty, flush previous section
            if current:
                sections.append(" ".join(current).strip())
                current = []
            current.extend(lines_group)
        else:
            current.extend(lines_group)

    if current:
        sections.append(" ".join(current).strip())
    sections = [s.strip() for s in sections if len(s.split()) > 5]
    return sections
import spacy
nlp = spacy.load("en_core_web_sm")
import tiktoken


enc = tiktoken.encoding_for_model("gpt-4-turbo")


def count_tokens(text):
    return len(enc.encode(text))


def sentence_token_chunking(text: str, max_tokens: int = 800, overlap_sentences: int = 1):
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks, cur, cur_tokens = [], [], 0
    for i, sent in enumerate(sents):
        sent_tokens = count_tokens(sent)
        if cur_tokens + sent_tokens > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
            # Add overlap sentences for continuity
            cur = sents[max(0, i - overlap_sentences):i]
            cur_tokens = sum(count_tokens(x) for x in cur)
        cur.append(sent)
        cur_tokens += sent_tokens
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks
# Breaks text into spaCy sentences.
# Accumulates them until the chunk reaches a target token size (e.g., 800 tokens).
# Adds overlap for continuity.

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AsyncOpenAI
import asyncio


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def semantic_sentence_split(text: str, threshold: float = 0.6):
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if len(sents) < 3:
      return [text]
    # Get embeddings for each sentence
    res = await client.embeddings.create(model="text-embedding-3-small", input=sents)
    embs = np.array([d.embedding for d in res.data])
    sims = cosine_similarity(embs)
    sections, current = [], [sents[0]]
    for i in range(1, len(sents)):
        if sims[i-1, i] < threshold: # low similarity → boundary
            sections.append(" ".join(current))
            current = []
    current.append(sents[i])
    if current:
       sections.append(" ".join(current))
    return sections


async def smart_continuous_split(text):
        chunks = sentence_token_chunking(text, max_tokens=800)
        if len(chunks) < 3: # continuous, not much punctuation
            print("⚠️ Fallback to semantic split")
            chunks = await semantic_sentence_split(text)
        return chunks

async def process_each_content(res):
    heuristic_sections = huerististic_section_split(res)

    print("/////////////////////////////////////////////////////////////////////////////////////////////////////")
    print(heuristic_sections)

    # If heuristic found enough sections (more than 2), use them
    if len(huerististic_section_split) >= 3:
        print(f"✅ Heuristic splitter found {len(heuristic_sections)} sections.")
        return heuristic_sections
    
    smart_continuous_sections  = smart_continuous_split(res)

    return smart_continuous_sections

async def chunking_embedding_wrapper(results):
    results = [await process_each_content(res) for res in results]
    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    results = [
        """ Jamie Dimon said he would put the chance of a serious fall in the US market at ‘more like 30%’, when 10% is currently priced in. Photograph: Mike Segar/ReutersView image in fullscreenJamie Dimon said he would put the chance of a serious fall in the US market at ‘more like 30%’, when 10% is currently priced in. Photograph: Mike Segar/ReutersJP MorganHead of largest US bank warns of risk of American stock market crashJamie Dimon, chair of JPMorgan Chase, said he was ‘far more worried than others’ about serious market correctionSimon GoodleyThu 9 Oct 2025 12.30 BSTLast modified on Fri 10 Oct 2025 08.20 BSTShareThe chance of the US stock market crashing is far greater than many financiers believe, the head of America’s largest bank has said.Jamie Dimon, who is the chair and chief executive of the giant Wall Street bank JPMorgan Chase, said he was “far more worried than others” about a serious market correction, which he predicted could come in the next six months to two years.“I would give it a higher probability than I think is probably priced in the market and by others,” he told the BBC. “So if the market’s pricing in 10%, I would say it is more like 30%.”Dimon added there were a “lot of things out there” creating an atmosphere of uncertainty, pointing to risks including the geopolitical environment, fiscal spending and the remilitarisation of the world.“All these things cause a lot of issues that we don’t know how to answer,” he said. “So I say the level of uncertainty should be higher in most people’s minds than what I would call normal.”The comments are the latest in a string of warnings that stock markets may be due a correction.On Wednesday, the head of the International Monetary Fund, Kristalina Georgieva, said the world economy had shown surprising resilience in the face of Donald Trump’s trade war, but issued a stark warning about the mounting risks, saying: “Buckle up: uncertainty is the new normal.”“Before anyone heaves a big sigh of relief, please hear this: global resilience has not yet been fully tested. And there are worrying signs the test may come,” she told an audience at the Milken Institute in Washington.Meanwhile, concerns are increasingly being aired that a stock market bubble has been created by high valuations of AI companies, with the Bank of England stating on Wednesday that there is a growing risk of a “sudden correction” in global markets.skip past newsletter promotionSign up to Business TodayFree daily newsletterGet set for the working day – we'll point you to all the business news and analysis you need every morningEnter your email address Sign upPrivacy Notice: Newsletters may contain information about charities, online ads, and content funded by outside parties. If you do not have an account, we will create a guest account for you on theguardian.com to send you this newsletter. You can complete full registration at any time. For more information about how we use your data see our Privacy Policy. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.after newsletter promotionDimon conceded that some of the money being invested in AI would “probably be lost”.He added: “The way I look at it is AI is real; AI in total will pay off – just like cars in total paid off, and TVs in total paid off, but most people involved in them didn’t do well.”Explore more on these topicsJP MorganInternational Monetary Fund (IMF)International tradeBankingGlobal economyJamie DimonnewsShareReuse this content""",
        """ Stocks to watch next week: JPMorgan, TSMC, Infosys, ASML and Bellway    Earnings preview of key companies reporting in the coming week and what to look out for         Vicky McKeever    and Pedro Goncalves     Updated 10 October 2025 10 min read               In this article:         2330.TW          INFY.NS          ASML.AS          BWY.L          JPM                A packed week of earnings is set to provide investors with fresh insights into the state of the global economy, as major companies across various sectors and geographies prepare to report.   In the US, JPMorgan Chase (JPM), the country’s largest bank by assets and market capitalisation, will lead the financial sector’s quarterly updates when it reports on Tuesday.        In Asia, Taiwan Semiconductor Manufacturing Company’s (2330.TW, TSM) results will serve as a barometer for the strength of the AI-driven chip boom, as questions mount over whether the rally has run ahead of fundamentals. In India, investor focus is on Infosys, which will report earnings days after unveiling a £1.75bn share buyback that has sparked renewed interest in the stock.   In Europe, all eyes are on ASML (ASML), the continent’s largest and most valuable tech group, as it updates investors on demand for its advanced chipmaking tools amid a string of target price upgrades by analysts.   Meanwhile, in the UK, Bellway's (BWY.L) full-year results will be closely watched not just for past performance, but for forward guidance and further clarity on the housebuilder’s capital return strategy through to June 2026.        Here's more detail on what to look out for:   JPMorgan Chase (JPM) – Reports third-quarter results on Tuesday, 14 October   JPMorgan Chase is poised to kick off the third quarter earnings season next week, with the US banking giant scheduled to report results on Tuesday, 14 October.      Analysts are forecasting earnings of $4.83 per share on revenues of $44.66bn, marking year-on-year increases of 10.5% and 4.7%, respectively, as the bank continues to benefit from elevated interest rates and resilient consumer activity.   Profit estimates for the quarter have risen 2.1% over the past month and 6.7% over the past quarter, amid the market’s heightened expectations.        Shares in the New York-listed group have tended to react positively to recent earnings reports, ending higher in three of the past four quarters. However, the stock dipped 0.7% in July despite delivering results that topped analyst expectations.   Investor sentiment toward US banking stocks has turned increasingly constructive in recent weeks, bolstered by expectations of accelerating loan demand and signs that credit delinquencies may have peaked. Strengthening deal pipelines and sustained trading activity have added to the sense of optimism, underpinned by a supportive monetary and regulatory backdrop.   Read more: Should you invest in gold?    Wells Fargo (WFC), Goldman Sachs (GS), and Citigroup (C) will also report before the market opens on Tuesday, 14 October.     Story continues    Jamie Dimon, the chair and chief executive of JPMorgan Chase, said he was “far more worried than others” about a serious market correction, which he predicted could come in the next six months to two years. “I would give it a higher probability than I think is probably priced in the market and by others,” he told the BBC. “So if the market’s pricing in 10%, I would say it is more like 30%.”    TSMC (2330.TW, TSM) – Releases third-quarter results on Thursday 16 October The latest monthly sales figures from TSMC (2330.TW, TSM) have already given investors some idea as to how the world's largest contract chipmaker performed in the third quarter.  On Thursday, TSMC reported revenue of TWD330.98bn (£8.12bn) for September. This took its total revenue for the third quarter to TWD989.92bn, according to Yahoo Finance UK's calculations. This beat an LSEG Smart Estimate of TWD973.26bn, based on forecasts from 22 analysts, according to a Reuters report. Converted into US dollars, the third quarter figure came in at $32.43bn, which would be around the middle of TSMC's guidance for the period of $31.8bn to $33bn, which it gives in USD.    Matt Britzman, senior equity analyst at Hargreaves Lansdown, said: "That sets the bar high as we look toward the final quarter, but it is consistent with the upward trajectory established earlier in the year and signals continued strength for the AI trade even amid macro and policy uncertainty." Following the release of its latest sales figures, TSMC's Taipei listed shares (2330.TW) hit a fresh high, with the stock up nearly 34% year-to-date. In the second quarter, TSMC reported a nearly 39% increase in revenue to TWD933.79bn and a 60% jump in net profit to TWD398.27bn. Read more: Is Wall Street ready to venture beyond bitcoin and into altcoins? "TSMC got off to a strong start in 2025, with first-half results breezing past expectations and underscoring the strength of its exposure to AI and high-performance computing," said Britzman. "Meanwhile, tension over tariffs persists, with Washington having floated a mandate that half of chip production occur in the US, a demand that Taiwan didn’t take too kindly."    TSMC has already pledged to scale up its operations in the US, announcing earlier this year that it would increase its investment to $165bn. This included the development of three new fabrication plants, two advanced packaging facilities, as well as a major research and development team centre. Infosys (INFY.NS) - Reports second-quarter results on Thursday, 16 October Infosys, one of India’s largest information technology services companies, is due to report second-quarter results for the period ending September 30 this Thursday, with investors looking for signals on revenue growth, client momentum and margin resilience amid intensifying global competition.  The results will provide a snapshot of performance across the group’s digital services, consulting and technology platforms, at a time when the broader IT services sector continues to navigate a shifting demand landscape in key Western markets. Ahead of the announcement, Infosys’s board approved a share buyback of INR18,000 crore (£1.75bn) on 11 September, at a fixed price of INR1,800 per share.    The buyback will represent 2.41% of Infosys’s equity capital and falls within the 25% threshold of the company’s paid-up share capital, according to a regulatory filing. The programme is nearly double the size of the company’s INR9,300 crore buyback in October 2022, which was conducted via the open market at a maximum price of INR1,850 per share. Earlier buybacks included INR13,000 crore in 2017 and INR8,260 crore in 2019. Analysts and market participants will be monitoring whether the share repurchase signals management confidence in the company’s medium-term outlook, especially given the current challenges facing discretionary IT spending and client decision-making cycles. Technically, some investors see signs of stabilisation in the share price. Nilesh Jain, head of technical and derivatives research at Centrum Broking, noted that the stock appears to have bottomed out. “Although it is too early to confirm a full reversal, we are seeing early signs of bottoming out. I expect a pullback towards INR1,550 in the near term, with support at INR1,440,” he said.    ASML (ASML, ASML.AS) – Reports third-quarter results on Wednesday, 15 October ASML, the Dutch semiconductor equipment giant, is set to report third-quarter earnings on Wednesday, 15 October, with markets focused on key updates related to its High-NA EUV technology, export restrictions tied to China and broader customer investment trends.  The company, which holds a dominant position in the global lithography equipment market, has become a central player in the semiconductor supply chain, particularly as demand for advanced chips accelerates alongside the adoption of AI technologies. Goldman Sachs reiterated its Buy rating on ASML ahead of the results, maintaining a €935 (£814) price target. The bank described ASML as a “clear beneficiary of improved leading-edge logic and memory demand driven by AI,” and pointed to a favourable inflection in four of the six categories tracked in its Europe Semiconductor Capital Equipment (Semicap) monitor.    Goldman’s US semiconductor team also upgraded its growth forecasts for wafer fabrication equipment (WFE) in 2025 and 2026, citing a more constructive capital spending outlook among foundries and memory manufacturers. In that context, the bank believes ASML’s current guidance for 2026 is achievable, noting that a quarterly order intake of €2bn would be sufficient to meet consensus expectations for systems revenue. Read more: The most popular stocks and funds investors bought in September While demand in mature technology nodes remains subdued, Goldman said that inventories in these segments appear to have bottomed, signalling potential for stabilisation. ASML shares have surged roughly 30% over the past month, buoyed by renewed investor enthusiasm for semiconductor equipment names. Despite the rally, Goldman Sachs sees scope for further upside and a potential re-rating, particularly as extreme ultraviolet (EUV) lithography layer adoption picks up in the latter half of the decade and ASML gains share in the WFE market into 2027–28.    On 1 October, UBS (UBS) lifted the price objective on the company’s stock to €940 from €750, while maintaining a Buy rating. Bellway (BWY.L) – Reports full-year results on Tuesday, 14 October Bellway (BWY.L) is set to report full-year results this Tuesday, with investor focus likely to centre less on backward-looking figures and more on forward guidance for the year to June 2026 and a potential overhaul of the group’s capital allocation policy.  Russ Mould, investment director, Danni Hewson, head of financial analysis and Dan Coatsworth, head of markets, all at AJ Bell, write that with much of the backward-looking detail disclosed in its August trading update, the spotlight will be on the company’s current trading performance, the strength of its £1.5bn order book, and its outlook for sales volumes and completions in the new financial year.    Sales per outlet per week improved to 0.57 in the year to June, up from 0.51 a year earlier. Commentary on recent activity will be key for investors, particularly after peer Taylor Wimpey (TW.L) flagged a cooling in demand as buyers await the government’s November budget. Bellway shares have declined roughly 20% over the past 12 months, amid concerns about slowing market momentum. Stocks: Create your watchlist and portfolio AJ Bell said analysts will also look for confirmation of Bellway’s prior forecast for a 5% increase in housing completions in the year to June 2026, which would lift volumes to around 9,000 units, compared with 8,749 in the year just ended. For the 2024 financial year, analysts expect revenue of £2.8bn, up from £2.4bn, with £3bn forecast for the current year. Pre-tax profit is estimated at £276m, rising to £313m in the year ahead, though Mould, Hewson and Coatsworth warned that any further provisions linked to cladding remediation or Building Safety Act compliance could affect final figures.    Other companies reporting next week include: Monday 13 October Tristel (TSTL.L) Fastenal (FAST) Tuesday 14 October Robert Walters (RWA.L) YouGov (YOU.L) Bytes Technology (BYIT.L) Ashmore (ASHM.L) Johnson and Johnson (JNJ) Wells Fargo (WFC) Goldman Sachs GS (GS) Citigroup (C) Domino’s Pizza (DPZ) Wednesday 15 October  PageGroup (PAGE.L) Sanderson Design (SDG.L)    Rio Tinto (RIO.L) Rank (RNK.L) Bank of America (BAC) Abbott Laboratories (ABT) Morgan Stanley (MS) LAM Research (LRCX) Las Vegas Sands (LVS) United Airlines (UAL) Citizens Financial (CFG) Thursday 16 October Whitbread (WTB.L) Croda (CRDA.L) GB Group (GBG.L) Sabre (SBRE.L) Canal+ (CAN.L) Nestlé (NESN.SW) EssilorLuxottica (EL.PA) ABB (ABB.NS) EQT (EQT) Pernod Ricard (RI.PA) Kinnevik (KINV-A.ST) Marsh & McLennan (MMC) Bank of New York Mellon (BK) US Bancorp (USB) Travelers (TRV) Freeport McMoRan (FCX) M&T Bank (MTB) Friday 17 October Yara International (YAR.OL) Tomra (TOM.OL) American Express (AXP) Schlumberger (SLB) Fifth Third Bancorp (FITB) Interpublic (IPG) Autoliv (ALV) You can read Yahoo Finance's full calendar here. Read more:  The most popular stocks and funds investors bought in September  How Instagram affects our spending  UK economic growth slowed between April and June, ONS confirms   Download the Yahoo Finance app, available for Apple and Android.    Terms   and Privacy Policy     Privacy dashboard"""
    ]
    asyncio.run(chunking_embedding_wrapper(results))


    # string_str = """ Jamie Dimon said he would put the chance of a serious fall in the US market at ‘more like 30%’, when 10% is currently priced in. Photograph: Mike Segar/ReutersView image in fullscreenJamie Dimon said he would put the chance of a serious fall in the US market at ‘more like 30%’, when 10% is currently priced in. Photograph: Mike Segar/ReutersJP MorganHead of largest US bank warns of risk of American stock market crashJamie Dimon, chair of JPMorgan Chase, said he was ‘far more worried than others’ about serious market correctionSimon GoodleyThu 9 Oct 2025 12.30 BSTLast modified on Fri 10 Oct 2025 08.20 BSTShareThe chance of the US stock market crashing is far greater than many financiers believe, the head of America’s largest bank has said.Jamie Dimon, who is the chair and chief executive of the giant Wall Street bank JPMorgan Chase, said he was “far more worried than others” about a serious market correction, which he predicted could come in the next six months to two years.“I would give it a higher probability than I think is probably priced in the market and by others,” he told the BBC. “So if the market’s pricing in 10%, I would say it is more like 30%.”Dimon added there were a “lot of things out there” creating an atmosphere of uncertainty, pointing to risks including the geopolitical environment, fiscal spending and the remilitarisation of the world.“All these things cause a lot of issues that we don’t know how to answer,” he said. “So I say the level of uncertainty should be higher in most people’s minds than what I would call normal.”The comments are the latest in a string of warnings that stock markets may be due a correction.On Wednesday, the head of the International Monetary Fund, Kristalina Georgieva, said the world economy had shown surprising resilience in the face of Donald Trump’s trade war, but issued a stark warning about the mounting risks, saying: “Buckle up: uncertainty is the new normal.”“Before anyone heaves a big sigh of relief, please hear this: global resilience has not yet been fully tested. And there are worrying signs the test may come,” she told an audience at the Milken Institute in Washington.Meanwhile, concerns are increasingly being aired that a stock market bubble has been created by high valuations of AI companies, with the Bank of England stating on Wednesday that there is a growing risk of a “sudden correction” in global markets.skip past newsletter promotionSign up to Business TodayFree daily newsletterGet set for the working day – we'll point you to all the business news and analysis you need every morningEnter your email address Sign upPrivacy Notice: Newsletters may contain information about charities, online ads, and content funded by outside parties. If you do not have an account, we will create a guest account for you on theguardian.com to send you this newsletter. You can complete full registration at any time. For more information about how we use your data see our Privacy Policy. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.after newsletter promotionDimon conceded that some of the money being invested in AI would “probably be lost”.He added: “The way I look at it is AI is real; AI in total will pay off – just like cars in total paid off, and TVs in total paid off, but most people involved in them didn’t do well.”Explore more on these topicsJP MorganInternational Monetary Fund (IMF)International tradeBankingGlobal economyJamie DimonnewsShareReuse this content"""
    # listSTR = string_str.split(".")
    # for lists in listSTR:
    #     print(lists)
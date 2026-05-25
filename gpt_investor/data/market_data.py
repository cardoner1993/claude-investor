import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger


def get_company_name(ticker: str) -> str:
    info = yf.Ticker(ticker).info
    return info.get("shortName") or info.get("longName") or ticker


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    return data["Close"].iloc[-1]


def get_news(ticker: str) -> list:
    return yf.Ticker(ticker).news


def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return "No analyst ratings available."

    latest_rating = recommendations.iloc[-1]
    firm = latest_rating.get("Firm", "N/A")
    to_grade = latest_rating.get("To Grade", "N/A")
    action = latest_rating.get("Action", "N/A")

    return f"Latest analyst rating for {ticker}:\nFirm: {firm}\nTo Grade: {to_grade}\nAction: {action}"


def _fetch_article_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)[:2000]
            logger.info("fetch OK  {} chars  {}", len(text), url[:80])
            return text
        logger.debug("fetch SKIP  status={}  {}", resp.status_code, url[:80])
    except Exception as e:
        logger.warning("fetch FAIL  {}  {}", e, url[:80])
    return ""

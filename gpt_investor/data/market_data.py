import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger


def get_company_name(ticker: str) -> str:
    """Resolve a display name for a ticker via yfinance.

    Falls back to the ticker symbol itself when neither name field is present.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    str
        `shortName`, else `longName`, else the ticker.
    """
    info = yf.Ticker(ticker).info
    return info.get("shortName") or info.get("longName") or ticker


def get_current_price(ticker):
    """Latest 1-minute close price for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Most recent close from a 1-day, 1-minute history.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    return data["Close"].iloc[-1]


def get_news(ticker: str) -> list:
    """yfinance news metadata for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    list
        Raw yfinance news items (each a dict with a `content` block).
    """
    return yf.Ticker(ticker).news


def get_analyst_ratings(ticker):
    """Format the latest analyst recommendation as a text blurb.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    str
        Firm / to-grade / action lines, or a "No analyst ratings available."
        message when yfinance returns nothing.
    """
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
    """Scrape up to 2000 chars of visible article text from a URL.

    Strips script/style/nav/footer/header before extracting text. Any
    non-200, non-HTML, or exception returns an empty string.

    Parameters
    ----------
    url : str
        Article URL.

    Returns
    -------
    str
        Trimmed body text, or "" on failure.
    """
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

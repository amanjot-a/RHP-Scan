import pdfplumber
import pandas as pd
import numpy as np
import re
import json
import logging
from collections import Counter

# -----------------------------
# Year normalization helpers
# -----------------------------

def normalize_year_key(s):
    """Extract a 4-digit year (e.g., 2025) from a string and return it as '2025'."""
    if s is None:
        return None
    s = str(s)
    m = re.search(r"(20\d{2})", s)
    return m.group(1) if m else None


def normalize_financials(fin):
    """Convert nested financial dicts to use normalized 4-digit year strings.
    Example: {'Revenue': {'march 31, 2025': 100}} -> {'Revenue': {'2025': 100}}
    """
    out = {}
    for key, sub in fin.items():
        if not isinstance(sub, dict):
            out[key] = sub
            continue
        d = {}
        for k, v in sub.items():
            y = normalize_year_key(k)
            if y:
                d[y] = v
        out[key] = d
    return out


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------
# 1. LOAD PDF & EXTRACT TEXT
# -----------------------------

pdf_path = r"D:\Chrome\New folder (2)\Red Herring Prospectus.pdf"

def is_financial_table(df):
    keywords = [
        "revenue", "income", "sales",
        "profit", "pat", "ebitda",
        "cash flow", "statement of cash flows", "balance sheet",
        "total assets", "assets", "liabilities",
        "shareholder", "statement of profit", "statement of profit and loss",
        "net cash from operating", "cash from operations"
    ]

    text_blob = " ".join(df.astype(str).fillna("").values.flatten()).lower()
    year_matches = re.findall(r"20\d{2}", text_blob)
    year_count = len(set(year_matches))

    # require at least two distinct year columns and presence of financial keywords
    return year_count >= 2 and any(k in text_blob for k in keywords)


def extract_text(pdf_path):
    text = ""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                for table in page.extract_tables():
                    tables.append(pd.DataFrame(table))
    except FileNotFoundError:
        logging.error("PDF not found: %s", pdf_path)
        raise

    financial_tables = [t for t in tables if is_financial_table(t)]
    return text.lower(), financial_tables


# -----------------------------
# 2. FINANCIAL EXTRACTION
# -----------------------------
def extract_financials(text):
    patterns = {
        "revenue": r"revenue.*?(\d[\d,]+)",
        "ebitda": r"ebitda.*?(\d[\d,]+)",
        "pat": r"(profit after tax|net profit).*?(\d[\d,]+)",
        "debt": r"(borrowings|debt).*?(\d[\d,]+)"
    }

    data = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        # matches can be strings or tuples (if the pattern has multiple groups).
        # extract the numeric group (last element if tuple) and sanitize commas
        values = []
        for m in matches:
            num_str = m if isinstance(m, str) else m[-1]
            num_str = num_str.replace(",", "")
            try:
                values.append(int(num_str))
            except ValueError:
                # skip non-numeric matches
                continue
        data[key] = values[:5]

    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# -----------------------------
# 3. RISK ANALYSIS (NLP)
# -----------------------------
def risk_analysis(text):
    risk_keywords = [
        "litigation", "penalty", "regulatory", "dependent",
        "volatile", "uncertain", "loss", "adverse", "risk"
    ]
    words = text.split()
    counts = Counter(words)

    risk_score = sum(counts[k] for k in risk_keywords)

    if risk_score > 300:
        level = "HIGH"
    elif risk_score > 150:
        level = "MEDIUM"
    else:
        level = "LOW"

    return risk_score, level

# -----------------------------
# 4. VALUATION METRICS
# -----------------------------
def valuation_metrics(text):
    eps_matches = re.findall(r"eps.*?([\d,]+\.?\d*)", text)
    price_matches = re.findall(r"price band.*?([\d,]+\.?\d*)", text)

    def pick_number(matches):
        for m in matches:
            s = m.replace(",", "").strip()
            if s and re.match(r"^-?\d+(?:\.\d+)?$", s):
                return s
        return None

    eps_str = pick_number(eps_matches)
    price_str = pick_number(price_matches)

    eps = float(eps_str) if eps_str else None
    price = float(price_str) if price_str else None

    pe = price / eps if eps and price else None

    return {
        "EPS": eps,
        "IPO Price": price,
        "P/E": pe
    }

# -----------------------------
# 5. MANAGEMENT SCORING
# -----------------------------
def management_score(text):
    score = 10
    penalties = ["related party", "resigned", "litigation", "non-compliance"]

    for p in penalties:
        if p in text:
            score -= 1.5

    return max(score, 1)

# -----------------------------
# Additional Robust Helpers
# -----------------------------

def extract_yearwise_financials(tables):
    data = {"Revenue": {}, "PAT": {}, "TotalAssets": {}, "Equity": {}, "EBITDA": {}, "Debt": {}, "Borrowings": {}, "CFO": {}}

    def parse_int(val):
        s = str(val).replace(',', '').strip()
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        s = s.replace('+', '')
        s = s.replace('—', '-')
        if re.match(r'^-?\d+$', s):
            return int(s)
        try:
            return int(float(s))
        except Exception:
            return None

    for df in tables:
        df = df.replace({None: "", np.nan: ""})
        if df.shape[0] < 1:
            continue

        # Try to find the header row by locating a cell containing a year
        header_idx = None
        for i in range(min(3, df.shape[0])):
            row_text = " ".join(str(x) for x in df.iloc[i].values)
            if re.search(r"20\d{2}", row_text):
                header_idx = i
                break

        if header_idx is not None:
            df.columns = [str(c).lower() for c in df.iloc[header_idx]]
            df = df[header_idx+1:]
        else:
            df.columns = [str(c).lower() for c in df.iloc[0]]
            df = df[1:]

        for _, row in df.iterrows():
            row_text = " ".join(row.astype(str)).lower()

            # Revenue
            if re.search(r"\b(revenue|total income|sales)\b", row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        year = yr_match.group(0)
                        data["Revenue"][year] = num

            # PAT
            if re.search(r"\b(profit after tax|profit for the year|net profit|profit.*after.*tax|pat)\b", row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        year = yr_match.group(0)
                        data["PAT"][year] = num

            # Total assets
            if re.search(r"total assets|assets", row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        data["TotalAssets"][yr_match.group(0)] = num

            # Equity / shareholder's funds
            if re.search(r"equity|shareholder" , row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        data["Equity"][yr_match.group(0)] = num

            # EBITDA
            if re.search(r"ebitda|earnings before interest|earnings before interest, tax" , row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        data["EBITDA"][yr_match.group(0)] = num

            # Debt / borrowings
            if re.search(r"borrowings|debt|long-term borrowings|short-term borrowings", row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        data["Debt"][yr_match.group(0)] = data["Debt"].get(yr_match.group(0), 0) + num

            # CFO
            if re.search(r"net cash from operating activities|net cash from operating", row_text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        data["CFO"][yr_match.group(0)] = num

    return data

def cash_flow_quality(tables):
    cfo = {}
    pat = {}

    for df in tables:
        df = df.replace({None: "", np.nan: ""})
        if df.shape[0] < 1:
            continue

        # Find header row with years if possible
        header_idx = None
        for i in range(min(3, df.shape[0])):
            row_text = " ".join(str(x) for x in df.iloc[i].values)
            if re.search(r"20\d{2}", row_text):
                header_idx = i
                break

        if header_idx is not None:
            df.columns = [str(c).lower() for c in df.iloc[header_idx]]
            df = df[header_idx+1:]
        else:
            df.columns = [str(c).lower() for c in df.iloc[0]]
            df = df[1:]

        for _, row in df.iterrows():
            text = " ".join(row.astype(str)).lower()

            def parse_int(val):
                s = str(val).replace(',', '').strip()
                if s.startswith('(') and s.endswith(')'):
                    s = '-' + s[1:-1]
                s = s.replace('+', '')
                s = s.replace('—', '-')
                if re.match(r'^-?\d+$', s):
                    return int(s)
                try:
                    return int(float(s))
                except Exception:
                    return None

            if re.search(r"operat.*cash|cash.*operat", text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        year = yr_match.group(0)
                        cfo[year] = num

            if re.search(r"\b(profit after tax|profit for the year|net profit|profit.*after.*tax|pat)\b", text):
                for col, val in row.items():
                    yr_match = re.search(r"20\d{2}", str(col))
                    num = parse_int(val)
                    if yr_match and num is not None:
                        year = yr_match.group(0)
                        pat[year] = num

    quality = {}
    for year in pat:
        if year in cfo:
            quality[year] = "GOOD" if cfo[year] >= pat[year] else "WEAK"

    return quality


def extract_risk_section(text):
    match = re.search(r"risk factors(.*?)(financial information|management discussion)", text, re.S | re.I)
    return match.group(1) if match else text[:50000]


def advanced_risk_analysis(text):
    risk_text = extract_risk_section(text)

    weights = {
        "litigation": 3,
        "penalty": 3,
        "regulatory": 2,
        "dependency": 2,
        "loss": 2,
        "uncertain": 1,
        "may adversely": 2,
        "no assurance": 2
    }

    score = 0
    for word, weight in weights.items():
        score += risk_text.lower().count(word) * weight

    if score > 120:
        level = "HIGH"
    elif score > 60:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level

# -----------------------------
# Qualitative / Document NLP Helpers
# -----------------------------

def extract_section(text, start_keywords, end_keywords, max_len=8000):
    try:
        starts = "|".join([re.escape(s) for s in start_keywords])
        ends = "|".join([re.escape(e) for e in end_keywords])
        pattern = r"(?:(?:" + starts + r"))(.*?)(?=(?:" + ends + r"))"
        match = re.search(pattern, text, re.S | re.I)
        return match.group(1).strip()[:max_len] if match else ""
    except Exception:
        return ""


def identify_offer_specifics(text):
    section = extract_section(
        text,
        start_keywords=["objects of the issue", "objects of the offer"],
        end_keywords=["basis of issue price", "risk factors", "industry overview"]
    )

    # Flexible price band patterns (e.g., "₹100 to ₹110", "100-110", "Price Band of Rs. 100 - 110")
    price_band_patterns = [
        r"price band.*?[₹Rs\.\s]*?(\d+[\d,]*)\s*(?:to|\-|\–)\s*[₹Rs\.\s]*?(\d+[\d,]*)",
        r"price.*?[₹Rs\.\s]*(\d+[\d,]*)\s*(?:to|\-|\–)\s*[₹Rs\.\s]*(\d+[\d,]*)"
    ]

    price_band = None
    for p in price_band_patterns:
        m = re.search(p, section, re.I)
        if m:
            price_band = [m.group(1).replace(',', ''), m.group(2).replace(',', '')]
            break
    if not price_band:
        for p in price_band_patterns:
            m = re.search(p, text, re.I)
            if m:
                price_band = [m.group(1).replace(',', ''), m.group(2).replace(',', '')]
                break

    issue_size = None
    m = re.search(r"issue size.*?[₹Rs\.\s]*?(\d+[\d,]*)", section, re.I)
    if not m:
        m = re.search(r"issue size.*?[₹Rs\.\s]*?(\d+[\d,]*)", text, re.I)
    if m:
        issue_size = m.group(1).replace(',', '')

    return {
        "UseOfProceedsText": section,
        "PriceBand": price_band,
        "IssueSize": issue_size
    }


def interpret_offer_specifics(data):
    text = (data.get("UseOfProceedsText") or "").lower()

    score = 0
    interpretation = []

    if "debt repayment" in text or "repay" in text:
        interpretation.append("Funds used for debt reduction (risk lowering)")
        score += 1

    if "capital expenditure" in text or "expansion" in text or "capex" in text:
        interpretation.append("Growth-oriented use of proceeds")
        score += 2

    if "working capital" in text:
        interpretation.append("Working capital support (neutral)")
        score += 1

    if "general corporate purposes" in text or "general corporate" in text:
        interpretation.append("Vague use of funds (negative)")
        score -= 1

    return {
        "Summary": interpretation,
        "OfferQualityScore": score
    }


def analyze_business_model(text):
    section = extract_section(
        text,
        ["business overview", "our business", "company overview"],
        ["industry overview", "risk factors"]
    )

    keywords = {
        "asset light": "Scalable business model",
        "subscription": "Recurring revenue",
        "recurring revenue": "Recurring revenue",
        "long-term contract": "Revenue visibility",
        "brand": "Brand-driven moat",
        "technology": "Tech-enabled differentiation",
        "customer concentration": "Customer concentration risk",
        "customer concentration risk": "Customer concentration risk",
        "high attrition": "Workforce/attrition risk",
        "high capex": "Capital intensity",
        "low margin": "Low margin business"
    }

    insights = []
    for k, meaning in keywords.items():
        if k in section.lower():
            insights.append(meaning)

    return {
        "BusinessModelText": section[:1500],
        "Insights": insights
    }


def analyze_industry(text):
    section = extract_section(
        text,
        ["industry overview"],
        ["business overview", "risk factors"]
    )

    growth_words = ["growing", "cagr", "expanding", "growth", "increasing demand", "strong demand"]
    challenges = ["competitive", "regulatory", "price pressure", "disruption", "cyclical", "supply chain"]

    score = 0
    if any(w in section.lower() for w in growth_words):
        score += 2
    if any(w in section.lower() for w in challenges):
        score -= 1

    return {
        "IndustryText": section[:1500],
        "IndustryOutlook": "FAVOURABLE" if score > 0 else "CHALLENGING"
    }


def analyze_strengths_and_strategy(text):
    section = extract_section(
        text,
        ["strengths", "competitive strengths", "our strengths"],
        ["strategy", "business strategy", "risk factors"]
    )

    strategy_section = extract_section(
        text,
        ["strategy", "business strategy"],
        ["risk factors", "financial information"]
    )

    return {
        "Strengths": section[:1200],
        "Strategy": strategy_section[:1200]
    }


def interpret_debt(debt_equity):
    if debt_equity is None:
        return "UNKNOWN"
    if debt_equity > 1:
        return "HIGH LEVERAGE RISK"
    elif debt_equity > 0.5:
        return "MODERATE LEVERAGE"
    else:
        return "LOW DEBT RISK"


def financial_health_summary(roe, cfo_pat, revenue_cagr):
    summary = []

    if roe and roe > 15:
        summary.append("Strong return on equity")

    if cfo_pat is not None and cfo_pat < 1:
        summary.append("Weak cash flow quality")

    if revenue_cagr and revenue_cagr > 15:
        summary.append("High growth company")

    return summary


def analyze_management(text):
    section = extract_section(
        text,
        ["management", "board of directors", "our directors"],
        ["risk factors", "financial information"]
    )

    red_flags = ["litigation", "penalty", "default", "resigned", "fraud", "bankruptcy", "insolvency", "sanction", "suspended", "conflict of interest", "seized", "investigation", "fine"]

    score = 10
    for rf in red_flags:
        if rf in section.lower():
            score -= 2

    return {
        "ManagementText": section[:1200],
        "ManagementScore": score
    }


def analyze_promoter_holding(text):
    match = re.findall(r"promoter.*?(\d+\.?\d*)%", text, re.I)
    if not match:
        match = re.findall(r"promoter holding.*?(\d+\.?\d*)%", text, re.I)
    holding = float(match[0]) if match else None

    interpretation = (
        "HIGH COMMITMENT" if holding and holding > 50 else
        "MODERATE COMMITMENT" if holding and holding > 25 else
        "LOW PROMOTER SKIN IN GAME"
    )

    return {
        "PromoterHolding": holding,
        "Interpretation": interpretation
    }


def analyze_risk_factors(text):
    section = extract_section(
        text,
        ["risk factors"],
        ["financial information", "objects of the issue"]
    )

    weighted_risks = {
        "litigation": 4,
        "penalty": 3,
        "regulatory": 3,
        "dependency": 2,
        "loss": 2,
        "volatile": 1,
        "default": 4,
        "bankruptcy": 5,
        "insolvency": 5,
        "non-compliance": 3,
        "fraud": 5,
        "fine": 3,
        "violation": 3
    }

    score = sum(section.lower().count(k) * w for k, w in weighted_risks.items())

    level = "HIGH" if score > 100 else "MEDIUM" if score > 50 else "LOW"

    return {
        "RiskScore": score,
        "RiskLevel": level
    }


def analyze_legal_risk(text):
    litigation_mentions = re.findall(r"litigation|case|notice|penalty", text, re.I)

    return {
        "LegalRiskMentions": len(litigation_mentions),
        "LegalRiskLevel": "HIGH" if len(litigation_mentions) > 20 else "MODERATE"
    }


def full_qualitative_analysis(text):
    offer = interpret_offer_specifics(identify_offer_specifics(text))
    business = analyze_business_model(text)
    industry = analyze_industry(text)
    strategy = analyze_strengths_and_strategy(text)
    management = analyze_management(text)
    promoter = analyze_promoter_holding(text)
    risk = analyze_risk_factors(text)
    legal = analyze_legal_risk(text)

    return {
        "OfferSpecifics": offer,
        "BusinessModel": business,
        "Industry": industry,
        "StrengthsAndStrategy": strategy,
        "Management": management,
        "PromoterHolding": promoter,
        "RiskAssessment": risk,
        "LegalRisk": legal
    }


def export_dashboard_json(financials, cashflow, risk, valuation, mgmt, verdict, analysis=None):
    output = {
        "Financials": financials,
        "CashFlowQuality": cashflow,
        "Risk": {
            "Score": risk[0],
            "Level": risk[1]
        },
        "Valuation": valuation,
        "ManagementScore": mgmt,
        "FinalVerdict": verdict
    }

    if analysis:
        output["Analysis"] = analysis

    with open("ipo_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

# -----------------------------
# Additional Analyst Models
# -----------------------------

def listing_gain_model(valuation, risk_level, revenue_growth):
    """
    Output: ListingGainScore (0–10) + Signal
    """
    score = 5

    # Valuation effect
    if valuation.get("P/E") and valuation["P/E"] > 40:
        score += 1.5  # hype driven
    elif valuation.get("P/E") and valuation["P/E"] < 20:
        score -= 1

    # Risk tolerance (listing gains ignore risks)
    if risk_level == "HIGH":
        score += 1

    # Growth story
    if revenue_growth and revenue_growth > 20:
        score += 2

    score = min(max(score, 0), 10)

    signal = "HIGH LISTING GAIN POTENTIAL" if score >= 7 else "LIMITED LISTING GAIN"

    return score, signal


def calculate_cagr(values_by_year):
    years = sorted(values_by_year.keys())
    if len(years) < 2:
        return None

    start = values_by_year[years[0]]
    end = values_by_year[years[-1]]
    n = len(years) - 1

    if start <= 0:
        return None

    return round(((end / start) ** (1 / n) - 1) * 100, 2)


def long_term_model(roe, cashflow_quality, pe, risk_level, mgmt_score):
    score = 0

    # ROE quality
    if roe:
        if roe >= 20:
            score += 3
        elif roe >= 15:
            score += 2
        elif roe >= 10:
            score += 1

    # Cash flow discipline
    if cashflow_quality and "WEAK" not in cashflow_quality.values():
        score += 2

    # Valuation sanity
    if pe and pe < 25:
        score += 2
    elif pe and pe > 40:
        score -= 1

    # Governance
    if risk_level == "LOW":
        score += 2
    elif risk_level == "HIGH":
        score -= 2

    # Management
    if mgmt_score >= 7:
        score += 1

    verdict = (
        "STRONG LONG-TERM COMPOUNDER" if score >= 7
        else "AVERAGE LONG-TERM" if score >= 4
        else "WEAK LONG-TERM BET"
    )

    return score, verdict


def dupont_analysis(financials):
    """
    financials must contain:
    revenue, pat, total_assets, equity

    This version normalizes year keys and picks the most recent year where all metrics exist.
    """
    if not financials or not isinstance(financials, dict):
        return None

    fin = normalize_financials(financials)
    revenue = fin.get("Revenue", {})
    pat = fin.get("PAT", {})
    assets = fin.get("TotalAssets", {})
    equity = fin.get("Equity", {})

    # Find years present in all series
    years = set(revenue.keys()) & set(pat.keys()) & set(assets.keys()) & set(equity.keys())
    if not years:
        return None

    latest_year = sorted(years)[-1]

    try:
        rev = revenue[latest_year]
        p = pat[latest_year]
        a = assets[latest_year]
        e = equity[latest_year]

        if rev == 0 or e == 0:
            return None

        net_margin = p / rev
        asset_turnover = rev / a
        leverage = a / e

        roe = net_margin * asset_turnover * leverage * 100

        return {
            "Year": latest_year,
            "NetProfitMargin": round(net_margin * 100, 2),
            "AssetTurnover": round(asset_turnover, 2),
            "FinancialLeverage": round(leverage, 2),
            "ROE": round(roe, 2)
        }
    except Exception:
        return None


# -----------------------------
# IPO Scoring System (Config-driven)
# -----------------------------

# Rubric and thresholds are centralized here for easy tuning and documentation.
IPO_SCORE_CONFIG = {
    'max': {'Quantitative': 45, 'Qualitative': 20, 'ValueChain': 10, 'Disclosure': 15, 'Benchmarking': 10},
    # Quantitative sub-weights (sums to Quantitative max)
    'quant': {
        'ROE': {'excellent': 15, 'good': 10, 'ok': 6},
        'RevenueCAGR': {'high': 12, 'medium': 8, 'low': 4},
        'NetMargin': {'high': 10, 'medium': 6, 'low': 3},
        'DebtEquity': {'low': 4, 'medium': 2},
        'CFO_PAT': {'good': 4, 'ok': 2}
    },
    # Qualitative
    'qual': {'risk_low': 10, 'risk_med': 5, 'mgmt_max': 10, 'promoter_high': 3, 'promoter_mid': 1, 'legal_penalty_threshold': 10, 'legal_penalty': -7},
    # Disclosure
    'disc': {'priceband': 6, 'issue_size': 3, 'business_text_long': 3, 'risk_section': 3},
    # Benchmarking
    'bench': {'roe': 6, 'revcagr': 6, 'debt_equity': 3}
}

# Human-readable rubric (for docs & storing)
IPO_SCORING_RUBRIC = '''
IPO Scoring Rubric (Total 100):
- Quantitative (45): ROE(15) | Revenue CAGR(12) | Net Margin(10) | Debt/Equity(4) | CFO/PAT(4)
- Qualitative (25): Risk Level (0-10) | Management (0-10) | Promoter Holding (0-3) | Legal penalty
- Disclosure (15): Clear Price Band(6) | Issue Size(3) | Business description length(3) | Risk section presence(3)
- Benchmarking (15): ROE vs peers(6) | Revenue CAGR vs peers(6) | Debt/Equity vs peers(3)
Scoring penalties: Significant legal mentions (>10) -> -7 points applied to qualitative component.
'''


def score_quantitative(fin, screener=None, earnings_quality_res=None):
    """Score financial strength using IPO_SCORE_CONFIG."""
    cfg = IPO_SCORE_CONFIG['quant']
    score = 0
    details = {}

    dup = dupont_analysis(fin)
    roe = dup.get('ROE') if dup else None
    details['ROE'] = roe

    if roe is not None:
        if roe >= 20:
            score += cfg['ROE']['excellent']
        elif roe >= 15:
            score += cfg['ROE']['good']
        elif roe >= 10:
            score += cfg['ROE']['ok']

    revenue_cagr = calculate_cagr(fin.get('Revenue', {}))
    details['RevenueCAGR'] = revenue_cagr
    if revenue_cagr is not None:
        if revenue_cagr >= 25:
            score += cfg['RevenueCAGR']['high']
        elif revenue_cagr >= 15:
            score += cfg['RevenueCAGR']['medium']
        elif revenue_cagr >= 8:
            score += cfg['RevenueCAGR']['low']

    y = latest_year(fin.get('Revenue', {}))
    net_margin = None
    if y and fin.get('PAT', {}).get(y) is not None and fin.get('Revenue', {}).get(y):
        net_margin = fin['PAT'][y] / fin['Revenue'][y] * 100
    details['NetMargin'] = round(net_margin, 2) if net_margin is not None else None
    if net_margin is not None:
        if net_margin >= 20:
            score += cfg['NetMargin']['high']
        elif net_margin >= 12:
            score += cfg['NetMargin']['medium']
        elif net_margin >= 7:
            score += cfg['NetMargin']['low']

    debt_equity = None
    if y and fin.get('Debt', {}).get(y) is not None and fin.get('Equity', {}).get(y):
        equity_val = fin.get('Equity', {}).get(y)
        if equity_val:
            debt_equity = fin['Debt'][y] / equity_val
    details['DebtEquity'] = round(debt_equity, 2) if debt_equity is not None else None
    if debt_equity is not None:
        if debt_equity <= 0.5:
            score += cfg['DebtEquity']['low']
        elif debt_equity <= 1:
            score += cfg['DebtEquity']['medium']

    # CFO/PAT latest
    cfo_pat = None
    if earnings_quality_res:
        yrs = sorted(k for k in earnings_quality_res.keys())
        if yrs:
            cfo_pat = earnings_quality_res.get(yrs[-1])
    details['CFO_PAT'] = cfo_pat
    if cfo_pat is not None:
        if cfo_pat >= 1:
            score += cfg['CFO_PAT']['good']
        elif cfo_pat >= 0.6:
            score += cfg['CFO_PAT']['ok']

    return min(score, IPO_SCORE_CONFIG['max']['Quantitative']), details


def score_qualitative(qualitative, risk_level, mgmt_score):
    cfg = IPO_SCORE_CONFIG['qual']
    score = 0
    details = {}

    # Risk level influence
    if risk_level == 'LOW':
        score += cfg['risk_low']
    elif risk_level == 'MEDIUM':
        score += cfg['risk_med']
    details['RiskLevel'] = risk_level

    # Management score scaled to cfg mgmt_max
    mgmt_comp = (mgmt_score / 10.0) * cfg['mgmt_max'] if mgmt_score is not None else 0
    score += mgmt_comp
    details['MgmtComponent'] = round(mgmt_comp, 2)

    # Promoter holding
    promoter = qualitative.get('PromoterHolding', {}) if qualitative else {}
    holding = promoter.get('PromoterHolding') if isinstance(promoter, dict) else None
    if holding is not None:
        if holding > 50:
            score += cfg['promoter_high']
        elif holding > 25:
            score += cfg['promoter_mid']
    details['PromoterHolding'] = holding

    # Legal mentions penalty
    legal = qualitative.get('LegalRisk', {}) if qualitative else {}
    mentions = legal.get('LegalRiskMentions') if isinstance(legal, dict) else 0
    details['LegalMentions'] = mentions
    if mentions > cfg['legal_penalty_threshold']:
        score += cfg['legal_penalty']

    # Boundaries
    return max(min(score, IPO_SCORE_CONFIG['max']['Qualitative']), 0), details


def score_disclosure(qualitative, text):
    cfg = IPO_SCORE_CONFIG['disc']
    score = 0
    details = {}

    offer = qualitative.get('OfferSpecifics', {}) if qualitative else {}
    priceband = offer.get('PriceBand')
    issue_size = offer.get('IssueSize')

    if priceband:
        score += cfg['priceband']
        details['PriceBand'] = True
    else:
        details['PriceBand'] = False

    if issue_size:
        score += cfg['issue_size']
        details['IssueSize'] = True
    else:
        details['IssueSize'] = False

    business_text = (qualitative.get('BusinessModel', {}) or {}).get('BusinessModelText', '')
    if business_text and len(business_text) > 200:
        score += cfg['business_text_long']
        details['BusinessTextLong'] = True
    else:
        details['BusinessTextLong'] = False

    risk_section = qualitative.get('RiskAssessment', {})
    if risk_section and risk_section.get('RiskScore', 0) > 0:
        score += cfg['risk_section']
        details['RiskSection'] = True
    else:
        details['RiskSection'] = False

    return min(score, IPO_SCORE_CONFIG['max']['Disclosure']), details


def score_benchmark(fin, peers=None):
    if not peers:
        return 0, {}
    cfg = IPO_SCORE_CONFIG['bench']
    score = 0
    details = {}

    dup = dupont_analysis(fin)
    roe = dup.get('ROE') if dup else None
    details['ROE'] = roe
    if roe is not None and peers.get('ROE') is not None and roe >= peers.get('ROE'):
        score += cfg['roe']

    revenue_cagr = calculate_cagr(fin.get('Revenue', {}))
    details['RevenueCAGR'] = revenue_cagr
    if revenue_cagr is not None and peers.get('RevenueCAGR') is not None and revenue_cagr >= peers.get('RevenueCAGR'):
        score += cfg['revcagr']

    y = latest_year(fin.get('Revenue', {}))
    debt_equity = None
    if y and fin.get('Debt', {}).get(y) is not None and fin.get('Equity', {}).get(y):
        debt_equity = fin['Debt'][y] / fin['Equity'][y] if fin.get('Equity', {}).get(y) else None
    details['DebtEquity'] = debt_equity
    if debt_equity is not None and peers.get('DebtEquity') is not None and debt_equity <= peers.get('DebtEquity'):
        score += cfg['debt_equity']

    return min(score, IPO_SCORE_CONFIG['max']['Benchmarking']), details


def compute_ipo_score(fin, qualitative, valuation, mgmt_score, risk_level, peers=None, screener=None, earnings_quality_res=None, valuechain=None):
    fin_score, fin_details = score_quantitative(fin, screener=screener, earnings_quality_res=earnings_quality_res)
    qual_score, qual_details = score_qualitative(qualitative, risk_level, mgmt_score)
    disc_score, disc_details = score_disclosure(qualitative, "")
    bench_score, bench_details = score_benchmark(fin, peers=peers)

    # ValueChain contribution scaled to its max
    vc_score = 0
    vc_details = {}
    if valuechain and isinstance(valuechain, dict):
        vc_raw = valuechain.get('aggregate', {}).get('weighted_score', 0)
        # scale 0-10 to config max
        vc_score = round((vc_raw / 10.0) * IPO_SCORE_CONFIG['max'].get('ValueChain', 0), 2)
        vc_details = valuechain

    breakdown = {
        'Quantitative': fin_score,
        'Qualitative': qual_score,
        'ValueChain': vc_score,
        'Disclosure': disc_score,
        'Benchmarking': bench_score
    }

    total = fin_score + qual_score + vc_score + disc_score + bench_score
    total = min(max(int(round(total)), 0), 100)

    if total >= 75:
        verdict = 'STRONG BUY'
    elif total >= 60:
        verdict = 'CONSIDER'
    elif total >= 45:
        verdict = 'NEUTRAL'
    else:
        verdict = 'AVOID'

    return {
        'Score': total,
        'Breakdown': breakdown,
        'ComponentsDetail': {
            'Quantitative': fin_details,
            'Qualitative': qual_details,
            'ValueChain': vc_details,
            'Disclosure': disc_details,
            'Benchmark': bench_details
        },
        'Verdict': verdict,
        'RubricVersion': 'v1',
        'RubricText': IPO_SCORING_RUBRIC
    }


def earnings_quality(pat, cfo):
    # Normalize keys then compute CFO/PAT ratio for overlapping years
    pat_norm = {k: v for k, v in normalize_financials({'PAT': pat}).get('PAT', {}).items()}
    cfo_norm = {k: v for k, v in normalize_financials({'CFO': cfo}).get('CFO', {}).items()}

    quality = {}
    for year in sorted(set(pat_norm.keys()) & set(cfo_norm.keys())):
        if pat_norm[year] == 0:
            quality[year] = None
            continue
        ratio = cfo_norm[year] / pat_norm[year]
        quality[year] = round(ratio, 2)
    return quality


def operating_leverage(ebitda, revenue):
    years = sorted(revenue.keys())
    if len(years) < 2:
        return None

    try:
        growth_rev = (revenue[years[-1]] - revenue[years[-2]]) / revenue[years[-2]]
        growth_ebitda = (ebitda[years[-1]] - ebitda[years[-2]]) / ebitda[years[-2]]
    except Exception:
        return None

    return round(growth_ebitda / growth_rev, 2) if growth_rev != 0 else None


def balance_sheet_risk(debt, equity):
    risk = {}
    for year in debt:
        if year in equity and equity[year] != 0:
            d_e = debt[year] / equity[year]
            risk[year] = "HIGH" if d_e > 1 else "MODERATE" if d_e > 0.5 else "LOW"
    return risk

# -----------------------------
# Screener-style Ratio Engine
# -----------------------------

def latest_year(data):
    if not data:
        return None
    try:
        return sorted(data.keys())[-1]
    except Exception:
        return None


def profitability_ratios(fin):
    try:
        y = latest_year(fin.get("Revenue", {}))
        if not y:
            return {}
        return {
            "Net Profit Margin %": round(fin["PAT"][y] / fin["Revenue"][y] * 100, 2) if fin.get("PAT", {}).get(y) is not None else None,
            "EBITDA Margin %": round(fin["EBITDA"][y] / fin["Revenue"][y] * 100, 2) if fin.get("EBITDA", {}).get(y) is not None else None
        }
    except Exception:
        return {}


def return_ratios(fin):
    try:
        y = latest_year(fin.get("Revenue", {}))
        if not y:
            return {}
        roe = fin["PAT"][y] / fin["Equity"][y] * 100 if fin.get("PAT", {}).get(y) is not None and fin.get("Equity", {}).get(y) else None
        roa = fin["PAT"][y] / fin["TotalAssets"][y] * 100 if fin.get("PAT", {}).get(y) is not None and fin.get("TotalAssets", {}).get(y) else None
        roce = fin["EBITDA"][y] / (fin["TotalAssets"][y] - fin.get("Debt", {}).get(y, 0)) * 100 if fin.get("EBITDA", {}).get(y) is not None and fin.get("TotalAssets", {}).get(y) else None
        return {
            "ROE %": round(roe, 2) if roe is not None else None,
            "ROA %": round(roa, 2) if roa is not None else None,
            "ROCE %": round(roce, 2) if roce is not None else None
        }
    except Exception:
        return {}


def valuation_ratios(fin, market_price=None):
    try:
        y = latest_year(fin.get("PAT", {}))
        if not y:
            return {}
        shares = fin.get("SharesOutstanding")
        if not shares or shares == 0:
            # cannot compute per-share metrics without shares
            eps = None
            book_value = None
        else:
            eps = fin["PAT"][y] / shares if fin.get("PAT", {}).get(y) is not None else None
            book_value = fin["Equity"][y] / shares if fin.get("Equity", {}).get(y) is not None else None

        pe = round(market_price / eps, 2) if market_price and eps else None
        pb = round(market_price / book_value, 2) if market_price and book_value else None

        return {
            "EPS (₹)": round(eps, 2) if eps is not None else None,
            "Book Value (₹)": round(book_value, 2) if book_value is not None else None,
            "P/E": pe,
            "P/B": pb
        }
    except Exception:
        return {}


def leverage_ratios(fin):
    try:
        y = latest_year(fin.get("Debt", {}))
        if not y:
            return {}
        return {
            "Debt / Equity": round(fin["Debt"][y] / fin["Equity"][y], 2) if fin.get("Equity", {}).get(y) else None,
            "Equity Ratio": round(fin["Equity"][y] / fin["TotalAssets"][y], 2) if fin.get("TotalAssets", {}).get(y) else None
        }
    except Exception:
        return {}


def efficiency_ratios(fin):
    try:
        y = latest_year(fin.get("Revenue", {}))
        if not y:
            return {}
        return {
            "Asset Turnover": round(fin["Revenue"][y] / fin["TotalAssets"][y], 2) if fin.get("TotalAssets", {}).get(y) else None
        }
    except Exception:
        return {}


def cashflow_ratios(fin):
    try:
        y = latest_year(fin.get("CFO", {}))
        if not y:
            return {}
        cfo = fin["CFO"][y]
        capex = fin.get("Capex", {}).get(y, 0)
        fcf = cfo - capex
        return {
            "CFO / PAT": round(cfo / fin["PAT"][y], 2) if fin.get("PAT", {}).get(y) else None,
            "Free Cash Flow": round(fcf, 2),
            "FCF / PAT": round(fcf / fin["PAT"][y], 2) if fin.get("PAT", {}).get(y) else None
        }
    except Exception:
        return {}


def growth_cagr(data):
    years = sorted(data.keys())
    if len(years) < 2:
        return None
    start, end = data[years[0]], data[years[-1]]
    n = len(years) - 1
    if start <= 0:
        return None
    return round(((end / start) ** (1/n) - 1) * 100, 2)


def growth_ratios(fin):
    try:
        return {
            "Sales CAGR %": growth_cagr(fin.get("Revenue", {})),
            "Profit CAGR %": growth_cagr(fin.get("PAT", {}))
        }
    except Exception:
        return {}


def screener_ratios(fin, market_price=None):
    # fin is expected normalized year-key dict (strings like '2025')
    try:
        return {
            "Profitability": profitability_ratios(fin),
            "Returns": return_ratios(fin),
            "Valuation": valuation_ratios(fin, market_price),
            "Leverage": leverage_ratios(fin),
            "Efficiency": efficiency_ratios(fin),
            "Cash Flow": cashflow_ratios(fin),
            "Growth": growth_ratios(fin)
        }
    except Exception:
        return {}

# -----------------------------
# 6. FINAL VERDICT
# -----------------------------
def final_verdict(fin_df, risk_level, pe, mgmt_score):
    verdict = "NEUTRAL"

    if risk_level == "HIGH" or mgmt_score < 5:
        verdict = "AVOID"
    elif pe and pe < 20 and mgmt_score >= 7:
        verdict = "INVEST"

    return verdict

# -----------------------------
# 7. RUN PIPELINE
# -----------------------------
def analyze_rhp(pdf_path):
    text, tables = extract_text(pdf_path)

    # Year-wise financials from detected financial tables
    financials_yearwise = extract_yearwise_financials(tables)

    # Cashflow quality and derived CFO values
    cashflow_quality_res = cash_flow_quality(tables)

    # Better risk analysis focused on Risk Factors section
    risk_score, risk_level = advanced_risk_analysis(text)

    valuation = valuation_metrics(text)
    mgmt_score = management_score(text)

    # Analyst models
    revenue_growth = calculate_cagr(financials_yearwise.get("Revenue", {}))
    listing_score, listing_signal = listing_gain_model(valuation, risk_level, revenue_growth)

    dupont_result = dupont_analysis(financials_yearwise)
    roe = dupont_result.get("ROE") if dupont_result else None

    long_term_score, long_term_verdict = long_term_model(roe, cashflow_quality_res, valuation.get("P/E"), risk_level, mgmt_score)

    earnings_quality_result = earnings_quality(financials_yearwise.get("PAT", {}), financials_yearwise.get("CFO", {}))
    operating_lev = operating_leverage(financials_yearwise.get("EBITDA", {}), financials_yearwise.get("Revenue", {}))
    bs_risk = balance_sheet_risk(financials_yearwise.get("Debt", {}), financials_yearwise.get("Equity", {}))

    # Qualitative section extracted and interpreted from the RHP text
    qualitative = full_qualitative_analysis(text)

    # Value-chain analysis to inform strengths & execution risks
    try:
        value_chain = analyze_value_chain(text)
    except Exception:
        value_chain = None

    verdict = final_verdict(
        financials_yearwise,
        risk_level,
        valuation.get("P/E"),
        mgmt_score
    )

    # Normalize financials before analysis so downstream models use consistent years
    financials_norm = normalize_financials(financials_yearwise)

    # Re-run certain calculations on normalized data
    dupont_result = dupont_analysis(financials_norm)
    roe = dupont_result.get("ROE") if dupont_result else None

    earnings_quality_result = earnings_quality(financials_norm.get("PAT", {}), financials_norm.get("CFO", {}))
    operating_lev = operating_leverage(financials_norm.get("EBITDA", {}), financials_norm.get("Revenue", {}))
    bs_risk = balance_sheet_risk(financials_norm.get("Debt", {}), financials_norm.get("Equity", {}))

    analysis = {
        "ListingGain": {"Score": listing_score, "Signal": listing_signal},
        "LongTerm": {"Score": long_term_score, "Verdict": long_term_verdict},
        "DuPont": dupont_result,
        "EarningsQuality": earnings_quality_result,
        "OperatingLeverage": operating_lev,
        "BalanceSheetRisk": bs_risk,
        "Qualitative": qualitative
    }

    # Screener-style ratios (use IPO price as fallback market price if available)
    market_price = valuation.get("IPO Price") or valuation.get("Market Price") or None
    screener = screener_ratios(financials_norm, market_price=market_price)
    analysis["ScreenerRatios"] = screener

    # Include ValueChain into the analysis
    analysis['ValueChain'] = value_chain

    # Compute aggregated IPO Score and attach
    try:
        ipo_score = compute_ipo_score(financials_norm, qualitative, valuation, mgmt_score, risk_level, peers=None, screener=screener, earnings_quality_res=earnings_quality_result, valuechain=value_chain)
        analysis['IPOScore'] = ipo_score
    except Exception:
        ipo_score = None

    result = {
        "Financials": financials_yearwise,
        "CashFlowQuality": cashflow_quality_res,
        "Risk Score": risk_score,
        "Risk Level": risk_level,
        "Valuation": valuation,
        "Management Score": mgmt_score,
        "IPOScore": ipo_score,
        "Final Verdict": verdict,
        "Analysis": analysis
    }

    # Export dashboard-friendly JSON (with analysis block)
    try:
        export_dashboard_json(financials_yearwise, cashflow_quality_res, (risk_score, risk_level), valuation, mgmt_score, verdict, analysis=analysis)
        logging.info("Wrote ipo_analysis.json")
    except Exception as e:
        logging.error("Failed to write JSON: %s", e)

    return result
# -----------------------------
# 8. EXECUTION
# -----------------------------
# -----------------------------
# Print helpers (research-note style)
# -----------------------------
from datetime import datetime

# -----------------------------
# JSON store and safe-update helpers
# -----------------------------
analysis_json = {
    "meta": {
        "generated_at": datetime.now().isoformat()
    }
}


def update_json(root, key, value):
    """Safely update JSON dict without overwriting existing keys.
    If key not present, set it. If present and both are dicts, merge.
    """
    if key not in root:
        root[key] = value
    else:
        if isinstance(root[key], dict) and isinstance(value, dict):
            root[key].update(value)
        else:
            # Do not overwrite non-dict values; keep existing
            pass


def save_analysis_json(store, filename="ipo_analysis.json"):
    # update timestamp
    store.setdefault('meta', {})['generated_at'] = datetime.now().isoformat()
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=4)


def store_quantitative_metrics(store, metrics):
    update_json(store, 'QuantitativeMetrics', metrics)

def print_heading(title):
    print("\n" + "=" * 80)
    print(f"{title.upper()}")
    print("=" * 80)


def print_bullets(items):
    if not items:
        return
    for i in items:
        print(f"• {i}")


def print_offer_specifics(offer):
    print_heading("Offer Specifics")

    if offer.get("Summary"):
        print("Use of Proceeds:")
        print_bullets(offer.get("Summary", []))
    else:
        print("Use of Proceeds: Not clearly disclosed")

    print(f"\nOffer Quality Score: {offer.get('OfferQualityScore', 'N/A')} / 5")


def print_and_store_offer(offer, store):
    print_offer_specifics(offer)
    try:
        update_json(store, 'OfferSpecifics', offer)
    except Exception:
        pass


def print_business_model(business):
    print_heading("Company Overview & Business Model")

    insights = business.get("Insights", [])
    if insights:
        print("Key Business Characteristics:")
        print_bullets(insights)
    else:
        print("Business characteristics could not be clearly identified.")

    print("\nBusiness Description (excerpt):")
    desc = business.get("BusinessModelText", "") or ""
    print((desc[:600] + "...") if len(desc) > 600 else desc)


def print_and_store_business(business, store):
    print_business_model(business)
    try:
        update_json(store, 'BusinessModel', {
            'Insights': business.get('Insights'),
            'Description': business.get('BusinessModelText')
        })
    except Exception:
        pass


def print_industry_analysis(industry):
    print_heading("Industry Overview")

    print(f"Industry Outlook: {industry.get('IndustryOutlook', 'UNKNOWN')}\n")
    print("Industry Commentary (excerpt):")
    txt = industry.get("IndustryText", "") or ""
    print((txt[:600] + "...") if len(txt) > 600 else txt)


def print_and_store_industry(industry, store):
    print_industry_analysis(industry)
    try:
        update_json(store, 'IndustryAnalysis', {
            'Outlook': industry.get('IndustryOutlook'),
            'Commentary': industry.get('IndustryText')
        })
    except Exception:
        pass


# -----------------------------
# Value Chain Analysis
# -----------------------------

VALUE_CHAIN_KEYWORDS = {
    'inbound_logistics': ['supply', 'supplier', 'procure', 'procurement', 'raw material', 'inventory', 'inventory management', 'warehouse', 'transport', 'port', 'logistic'],
    'operations': ['manufactur', 'production', 'plant', 'capacity', 'throughput', 'operat', 'facility', 'processing', 'assembly'],
    'outbound_logistics': ['distribution', 'distribut', 'dealer', 'channel', 'ship', 'dispatch', 'delivery', 'freight', 'logistics'],
    'marketing_sales': ['marketing', 'sales', 'distribution network', 'channel partners', 'brand', 'retail', 'e-commerce', 'advertis', 'promotion', 'market share'],
    'service': ['service', 'after-sales', 'warranty', 'customer support', 'maintenance', 'spares', 'installation']
}

SUPPORT_KEYWORDS = {
    'infrastructure': ['infrastructure', 'plant', 'facility', 'estate'],
    'hr': ['hr', 'human resource', 'talent', 'attrition', 'attrit', 'training', 'compensation'],
    'technology': ['technology', 'it', 'software', 'automation', 'digital', 'erps', 'erp', 'r&d', 'r and d', 'innovation'],
    'procurement': ['procurement', 'sourcing', 'vendor', 'supplier']
}


def analyze_value_chain(text):
    """Analyze value-chain by extracting evidence sentences and scoring each activity (0-10).
    Returns a dict with activities, support functions and an aggregate weighted score.
    """
    if not text:
        return {
            'activities': {},
            'support': {},
            'aggregate': {'weighted_score': 0, 'notes': ['No text']}
        }

    # split into sentences
    sents = re.split(r'(?<=[\.\?\!])\s+', text)

    activities = {}

    for act, kws in VALUE_CHAIN_KEYWORDS.items():
        matches = []
        reasons = []
        for sent in sents:
            lowered = sent.lower()
            for kw in kws:
                if kw in lowered:
                    matches.append(sent.strip())
                    reasons.append(f"Mention of '{kw}'")
                    break
        # dedupe and keep top evidence
        evidence = list(dict.fromkeys(matches))[:5]
        # rough scoring: 0-10 based on evidence count and presence of positive signals
        base = min(len(evidence) * 2, 10)
        # boost where positive adjectives present
        boosts = sum(1 for ev in evidence if re.search(r'(?i)modern|state[- ]of[- ]the[- ]art|automati|efficient|integrat|backward integration|vertical integration', ev))
        score = min(10, base + boosts)
        activities[act] = {
            'score': score,
            'evidence': evidence,
            'reasons': reasons
        }

    support = {}
    for act, kws in SUPPORT_KEYWORDS.items():
        matches = []
        reasons = []
        for sent in sents:
            lowered = sent.lower()
            for kw in kws:
                if kw in lowered:
                    matches.append(sent.strip())
                    reasons.append(f"Mention of '{kw}'")
                    break
        evidence = list(dict.fromkeys(matches))[:4]
        score = min(10, len(evidence) * 2)
        support[act] = {'score': score, 'evidence': evidence, 'reasons': reasons}

    # Weighted aggregate: operations weight double
    weights = {
        'inbound_logistics': 1,
        'operations': 2,
        'outbound_logistics': 1,
        'marketing_sales': 1.2,
        'service': 1
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(activities[a]['score'] * weights.get(a, 1) for a in activities)
    agg_score = round((weighted_sum / total_weight), 2)  # in 0-10 scale

    notes = []
    # add interpretive notes
    if activities['operations']['score'] >= 7:
        notes.append('Operations appear well-described and capable.')
    if activities['marketing_sales']['score'] >= 6:
        notes.append('Go-to-market channels appear present and described.')
    if activities['inbound_logistics']['score'] <= 2:
        notes.append('Supply-chain visibility appears limited.')

    return {'activities': activities, 'support': support, 'aggregate': {'weighted_score': agg_score, 'notes': notes}}


def print_value_chain(vc_result):
    print("\n" + "="*80)
    print("VALUE CHAIN ANALYSIS")
    print("="*80)
    for act, info in vc_result["activities"].items():
        print(f"\n{act.replace('_',' ').title()}  — Score: {info['score']}/10")
        # show short evidence
        for ev in info["evidence"]:
            print(f"  - {ev[:180]}...")
        # show top reasons
        for r in info["reasons"][:3]:
            print(f"    * {r}")
    print("\nAggregate Value Chain Score:", vc_result["aggregate"]["weighted_score"])
    for n in vc_result["aggregate"]["notes"]:
        print(" -", n)


def print_and_store_value_chain(vc_result, store):
    try:
        print_value_chain(vc_result)
    except Exception:
        pass
    try:
        update_json(store, 'ValueChain', vc_result)
    except Exception:
        pass


def print_strengths_and_strategy(strategy):
    print_heading("Strengths & Strategy")

    if strategy.get("Strengths"):
        print("Strengths (as stated by company):")
        s = strategy.get("Strengths", "") or ""
        print((s[:600] + "...") if len(s) > 600 else s)

    if strategy.get("Strategy"):
        print("\nGrowth Strategy:")
        st = strategy.get("Strategy", "") or ""
        print((st[:600] + "...") if len(st) > 600 else st)


def print_and_store_management(mgmt, store):
    print_management_review(mgmt)
    try:
        update_json(store, 'Management', {
            'Score': mgmt.get('ManagementScore'),
            'Notes': mgmt.get('ManagementText')
        })
    except Exception:
        pass


def print_and_store_promoter(promoter, store):
    print_promoter_holding(promoter)
    try:
        update_json(store, 'PromoterHolding', promoter)
    except Exception:
        pass


def print_financial_summary(fin_summary):
    print_heading("Financial Analysis Summary")

    if fin_summary:
        print_bullets(fin_summary)
    else:
        print("No strong financial signals identified.")


def print_management_review(mgmt):
    print_heading("Management & Governance Review")

    print(f"Management Quality Score: {mgmt.get('ManagementScore', 'N/A')} / 10")

    print("\nManagement Commentary (excerpt):")
    m = mgmt.get("ManagementText", "") or ""
    print((m[:600] + "...") if len(m) > 600 else m)


def print_promoter_holding(promoter):
    print_heading("Promoter Shareholding")

    holding = promoter.get("PromoterHolding") if isinstance(promoter, dict) else None

    if holding is not None:
        print(f"Promoter Holding: {holding}%")
        print(f"Interpretation: {promoter.get('Interpretation')}")
    else:
        print("Promoter holding data not clearly available.")


def print_risk_assessment(risk, legal):
    print_heading("Risk Assessment")

    print(f"Overall Risk Level: {risk.get('RiskLevel', 'UNKNOWN')}")
    print(f"Risk Score: {risk.get('RiskScore', 'N/A')}")

    print("\nLegal & Regulatory Risk:")
    print(f"Mentions Found: {legal.get('LegalRiskMentions', 0)}")
    print(f"Risk Level: {legal.get('LegalRiskLevel', 'UNKNOWN')}")


def print_and_store_risk(risk, legal, store):
    print_risk_assessment(risk, legal)
    try:
        update_json(store, 'RiskAssessment', {
            'Overall': risk,
            'Legal': legal
        })
    except Exception:
        pass


def print_ipo_score(score_dict):
    print_heading("IPO SCORE")
    if not score_dict:
        print("IPO Score not available.")
        return
    print(f"Score: {score_dict.get('Score')} / 100")
    print(f"Verdict: {score_dict.get('Verdict')}")
    print("Breakdown:")
    for k, v in (score_dict.get('Breakdown') or {}).items():
        print(f" - {k}: {v}")


def print_and_store_ipo_score(score_dict, store):
    print_ipo_score(score_dict)
    try:
        update_json(store, 'IPOScore', score_dict)
    except Exception:
        pass


def print_ipo_rubric():
    print_heading("IPO SCORING RUBRIC")
    print(IPO_SCORING_RUBRIC)


def print_and_store_ipo_rubric(store):
    try:
        print_ipo_rubric()
    except Exception:
        pass
    try:
        update_json(store, 'IPOScoringRubric', {
            'Version': 'v1',
            'Text': IPO_SCORING_RUBRIC,
            'Config': IPO_SCORE_CONFIG
        })
    except Exception:
        pass


def print_full_analysis(result):
    """Accepts either the full `analyze_rhp` result dict or a qualitative dict.
    Prints a research-note style summary highlighting interpretations.
    """
    qual = result.get('Analysis', {}).get('Qualitative') if isinstance(result, dict) and 'Analysis' in result else result

    # If user passed full result, derive a short financial summary
    fin_summary = []
    try:
        if isinstance(result, dict) and 'Analysis' in result:
            analysis = result['Analysis']
            dup = analysis.get('DuPont')
            if dup and dup.get('ROE'):
                roe = dup.get('ROE')
            else:
                roe = None

            eq = analysis.get('EarningsQuality', {})
            # CFO/PAT latest
            cfo_pat = None
            if eq:
                yrs = sorted(k for k in eq.keys())
                cfo_pat = eq.get(yrs[-1])

            rev_cagr = None
            fin = result.get('Financials', {})
            if fin:
                rev_cagr = calculate_cagr(fin.get('Revenue', {}))

            fin_summary = financial_health_summary(roe, cfo_pat, rev_cagr)
    except Exception:
        fin_summary = []

    # Defensive defaults
    if not qual:
        print("No qualitative analysis available to print.")
        return

    print_offer_specifics(qual.get('OfferSpecifics', {}))
    print_business_model(qual.get('BusinessModel', {}))
    print_industry_analysis(qual.get('Industry', {}))
    print_strengths_and_strategy(qual.get('StrengthsAndStrategy', {}))
    print_financial_summary(fin_summary)
    print_management_review(qual.get('Management', {}))
    print_promoter_holding(qual.get('PromoterHolding', {}))
    print_risk_assessment(qual.get('RiskAssessment', {}), qual.get('LegalRisk', {}))


def print_and_store_full_analysis(result, store):
    """Prints full analysis and stores the 'Analysis' block safely into `store`. Accepts either
    the full pipeline result dict or the Qualitative dict directly.
    """
    try:
        # If full result dict provided, extract Qualitative
        qual = result.get('Qualitative') if isinstance(result, dict) and 'Qualitative' in result else result if isinstance(result, dict) else {}
        # Use existing print_full_analysis to print
        print_full_analysis(result if isinstance(result, dict) else {'Analysis': {'Qualitative': qual}})
    except Exception:
        try:
            print_full_analysis(result)
        except Exception:
            pass

    try:
        # store under Analysis. If given full result, store whole block; otherwise store as Qualitative
        if isinstance(result, dict) and 'Analysis' in result:
            update_json(store, 'Analysis', result['Analysis'])
        else:
            update_json(store, 'Analysis', {'Qualitative': result})
    except Exception:
        pass


if __name__ == "__main__":
    pdf_path = r"D:\Chrome\New folder (2)\Red Herring Prospectus.pdf"

    try:
        output = analyze_rhp(pdf_path)

        print("\nFINAL IPO VERDICT:", output["Final Verdict"])
        print("RISK LEVEL:", output["Risk Level"])
        print("MANAGEMENT SCORE:", output["Management Score"])
        print("VALUATION:", output["Valuation"])
        print("\nExported: ipo_analysis.json")

        # Print research-note style summary and store structured qualitative + quantitative JSON
        try:
            # Print and store qualitative analysis into analysis_json
            print_and_store_full_analysis(output.get('Analysis', {}).get('Qualitative', {}), analysis_json)

            # Build quantitative metrics and store them
            financials_norm = normalize_financials(output.get('Financials', {}))
            dup = output.get('Analysis', {}).get('DuPont')
            roe_val = dup.get('ROE') if dup else None

            screener_val = output.get('Analysis', {}).get('ScreenerRatios', {}).get('Valuation', {})
            pe_val = screener_val.get('P/E') if screener_val and screener_val.get('P/E') is not None else output.get('Valuation', {}).get('P/E')

            # Debt/Equity from normalized financials
            debt_equity = None
            debt = financials_norm.get('Debt', {})
            equity = financials_norm.get('Equity', {})
            common_years = set(debt.keys()) & set(equity.keys())
            if common_years:
                y = sorted(common_years)[-1]
                if equity[y] != 0:
                    debt_equity = round(debt[y] / equity[y], 2)

            metrics = {
                "ROE": roe_val,
                "P_E": pe_val,
                "Debt_Equity": debt_equity,
                "ListingGainScore": output.get('Analysis', {}).get('ListingGain', {}).get('Score'),
                "LongTermScore": output.get('Analysis', {}).get('LongTerm', {}).get('Score'),
                "IPO_Score": output.get('Analysis', {}).get('IPOScore', {}).get('Score'),
                "ValueChainScore": output.get('Analysis', {}).get('ValueChain', {}).get('aggregate', {}).get('weighted_score')
            }

            store_quantitative_metrics(analysis_json, metrics)

            # Save the scoring rubric and final structured JSON
            try:
                update_json(analysis_json, 'IPOScoringRubric', {
                    'Version': 'v1',
                    'Text': IPO_SCORING_RUBRIC,
                    'Config': IPO_SCORE_CONFIG
                })
            except Exception:
                pass

            # Print and store value-chain analysis into the structured JSON
            try:
                print_and_store_value_chain(output.get('Analysis', {}).get('ValueChain', {}), analysis_json)
            except Exception:
                pass

            save_analysis_json(analysis_json)
            print("\nSaved structured analysis JSON: ipo_analysis.json")

        except Exception as e:
            logging.error("Failed to print and store full analysis: %s", e)

    except FileNotFoundError:
        print("PDF not found. Please check the path:", pdf_path)
    except Exception as e:
        print("Analysis failed:", e)

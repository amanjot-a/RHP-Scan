import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="IPO RHP Analyzer", layout="wide")

st.title("IPO RHP Analyzer — Dashboard")

try:
    with open("ipo_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    st.error("ipo_analysis.json not found. Run the analysis first.")
    st.stop()

st.header("Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Final Verdict", data.get('FinalVerdict', 'N/A'))
    st.metric("Risk Level", data.get('Risk', {}).get('Level', 'N/A'))
with col2:
    st.metric("Management Score", data.get('ManagementScore', 'N/A'))
    st.metric("Listing Gain Signal", data.get('Analysis', {}).get('ListingGain', {}).get('Signal', 'N/A'))
with col3:
    st.metric("Long-Term Verdict", data.get('Analysis', {}).get('LongTerm', {}).get('Verdict', 'N/A'))

st.header("Financials — Year-wise")
fin = data.get('Financials', {})
if fin:
    df_rev = pd.Series(fin.get('Revenue', {}), name='Revenue').sort_index()
    df_pat = pd.Series(fin.get('PAT', {}), name='PAT').sort_index()
    st.subheader('Revenue')
    st.bar_chart(df_rev)
    st.subheader('PAT')
    st.bar_chart(df_pat)
else:
    st.info('No structured financials found.')

st.header('Analysis')
analysis = data.get('Analysis', {})
st.write(analysis)

st.header('DuPont (if available)')
dup = analysis.get('DuPont')
if dup:
    st.table(pd.DataFrame([dup]))
else:
    st.info('DuPont not available — check financials contain Total Assets and Equity for the same year')

st.header('Earnings Quality')
eq = analysis.get('EarningsQuality')
if eq:
    st.table(pd.DataFrame([eq]))
else:
    st.info('No earnings quality data (requires PAT and CFO overlap)')

st.caption('Run: python RHP.py to refresh ipo_analysis.json; then refresh this page.')

"""
=============================================================================
  TERM PROJECT — Topic Modelling of 2024 Ghana Election Media Coverage
  Streamlit Dashboard  |  Sections (c) · (d) · (e-i) · (e-ii) · (e-iii) · (e-iv)
  + Word Clouds  |  + LDA Matrix Displays (K1 · K2 · K3)
=============================================================================
  Run:
      pip install streamlit pandas scikit-learn matplotlib seaborn wordcloud numpy scipy
      streamlit run app.py
  Place all  *_final_preprocessed.csv  files in the same folder as app.py.
=============================================================================
"""

import io
import colorsys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ghana 2024 Election — LDA Topic Modelling",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --navy:    #1A3A5C;
    --gold:    #C9A84C;
    --crimson: #C0392B;
    --teal:    #1A7A6E;
    --slate:   #4A5568;
    --bg:      #F4F2EE;
    --card:    #FFFFFF;
    --state:   #2471A3;
    --priv:    #E67E22;
    --npp:     #0057A8;
    --ndc:     #009A44;
}
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg) !important;
}

/* ── Hero ──────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0b1e30 0%, #1A3A5C 55%, #1A7A6E 100%);
    border-radius: 18px;
    padding: 52px 56px 44px;
    margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:""; position:absolute; top:-70px; right:-70px;
    width:280px; height:280px; border-radius:50%;
    background:rgba(201,168,76,.13);
}
.hero::after {
    content:""; position:absolute; bottom:-90px; left:28%;
    width:360px; height:360px; border-radius:50%;
    background:rgba(26,122,110,.16);
}
.hero h1 {
    font-family:'Playfair Display',serif;
    font-size:2.7rem; font-weight:900; color:#fff;
    line-height:1.18; margin:0 0 12px;
    position:relative; z-index:1;
}
.hero p {
    font-size:1.05rem; color:rgba(255,255,255,.76);
    margin:0; position:relative; z-index:1; max-width:650px;
}
.hero-badge {
    display:inline-block;
    background:rgba(201,168,76,.2); border:1px solid rgba(201,168,76,.5);
    color:#C9A84C; font-family:'IBM Plex Mono',monospace;
    font-size:.74rem; letter-spacing:.1em;
    padding:4px 14px; border-radius:20px; margin-bottom:18px;
    position:relative; z-index:1;
}

/* ── Section headers ───────────────────────────── */
.sec-hdr {
    font-family:'Playfair Display',serif;
    font-size:1.7rem; font-weight:700; color:var(--navy);
    border-left:4px solid var(--gold);
    padding-left:14px; margin:36px 0 20px; line-height:1.25;
}
.sub-hdr {
    font-size:1rem; font-weight:600; color:var(--slate);
    margin:26px 0 10px; text-transform:uppercase; letter-spacing:.06em;
}

/* ── Metric cards ──────────────────────────────── */
.metric-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
.metric-card {
    flex:1; min-width:155px;
    background:var(--card); border-radius:14px;
    padding:20px 22px;
    box-shadow:0 2px 14px rgba(26,58,92,.08);
    border-top:3px solid var(--gold);
}
.metric-card .val {
    font-family:'Playfair Display',serif;
    font-size:2rem; font-weight:700; color:var(--navy); line-height:1;
}
.metric-card .lbl {
    font-size:.77rem; color:var(--slate);
    margin-top:5px; text-transform:uppercase; letter-spacing:.07em;
}

/* ── Insight / warn boxes ──────────────────────── */
.insight-box {
    background:linear-gradient(135deg,#eaf4fb,#f0faf8);
    border-left:4px solid var(--teal);
    border-radius:0 10px 10px 0;
    padding:16px 20px; margin:14px 0;
    font-size:.95rem; color:var(--navy);
}
.warn-box {
    background:#fff9ee; border-left:4px solid var(--gold);
    border-radius:0 10px 10px 0;
    padding:14px 18px; margin:12px 0;
    font-size:.92rem; color:#7a5c0a;
}

/* ── Tags ──────────────────────────────────────── */
.tag {
    display:inline-block; padding:3px 10px;
    border-radius:20px; font-size:.74rem;
    font-weight:600; letter-spacing:.05em; margin:2px 3px;
}
.tag-state   { background:#dbeeff; color:#1A3A5C; }
.tag-private { background:#fde8d0; color:#8a3a00; }
.tag-narrow  { background:#fde8e8; color:#7a0000; }
.tag-broad   { background:#d4f0eb; color:#0a4a40; }
.tag-npp     { background:#d6e8ff; color:#003070; }
.tag-ndc     { background:#d4f0e0; color:#004020; }
.tag-neutral { background:#ececec; color:#444; }

/* ── Topic table ───────────────────────────────── */
.topic-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.topic-table th {
    background:var(--navy); color:#fff;
    padding:10px 14px; text-align:left;
    font-size:.79rem; letter-spacing:.05em; text-transform:uppercase;
}
.topic-table td { padding:9px 14px; border-bottom:1px solid #eee; vertical-align:top; }
.topic-table tr:nth-child(even) td { background:#f9f8f5; }
.topic-table tr:hover td { background:#f0f4f8; }

/* ── Fancy divider ─────────────────────────────── */
.fancy-div {
    height:2px;
    background:linear-gradient(90deg,var(--gold),transparent);
    margin:28px 0; border:none;
}

/* ── Sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] { background:#0f2035 !important; }
[data-testid="stSidebar"] * { color:rgba(255,255,255,.88) !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12) !important; }

/* ── Figure images ─────────────────────────────── */
.stImage img {
    border-radius:10px;
    box-shadow:0 4px 20px rgba(26,58,92,.13);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CONSTANTS  (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
NAVY          = "#1A3A5C"
GOLD          = "#C9A84C"
CRIMSON       = "#C0392B"
TEAL          = "#1A7A6E"
SLATE         = "#4A5568"
BG_WHITE      = "#FAFAFA"
STATE_COLOR   = "#2471A3"
PRIVATE_COLOR = "#E67E22"
NPP_COLOR     = "#0057A8"
NDC_COLOR     = "#009A44"

CUSTOM_PAL = [
    "#1A3A5C","#2471A3","#5DADE2","#1A7A6E","#28B463",
    "#C9A84C","#E67E22","#C0392B","#8E44AD","#4A5568",
]
CLOUD_COLORS = CUSTOM_PAL  # one per topic

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.facecolor":  BG_WHITE,
    "axes.facecolor":    BG_WHITE,
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=155, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

def show_fig(fig, caption=""):
    st.image(fig_to_buf(fig), caption=caption, use_container_width=True)

def metric_html(val, label):
    return (f'<div class="metric-card">'
            f'<div class="val">{val}</div>'
            f'<div class="lbl">{label}</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 16px;">
        <div style="font-size:2.2rem;">🗳️</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                    font-weight:700;line-height:1.35;margin-top:8px;">
            Ghana 2024<br>Election Coverage
        </div>
        <div style="font-size:.7rem;opacity:.5;margin-top:4px;
                    font-family:'IBM Plex Mono',monospace;letter-spacing:.08em;">
            LDA TOPIC MODELLING PIPELINE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    PAGES = [
        "🏠  Overview & Pipeline",
        "⚙️  (c) Modelling",
        "🏷️  (d) Labelling",
        "☁️  Word Clouds",
        "📊  (e-i) Dominant Topics",
        "📰  (e-ii) Agenda by Outlet",
        "🔍  (e-iii) Narrow vs. Broad",
        "⚖️  (e-iv) Party Framing",
        "🔢  LDA Matrices",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.76rem;opacity:.55;line-height:1.75;">
        <b>Outlets</b><br>
        Citinewsroom · Daily Guide<br>
        Ghanaian Times · The Chronicle<br>
        Ghana News Agency · GBC<br><br>
        <b>Model</b><br>
        LDA · k=10 · sklearn<br>
        online learning · 40 iter
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL  (cached — runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting LDA model — this runs once…")
def load_and_fit():
    Citynewsroom      = pd.read_csv("Citynewsroom_final_preprocessed.csv")
    Dailyguide        = pd.read_csv("Dailyguide_final_preprocessed.csv")
    Ghanaiantimes     = pd.read_csv("Ghanaiantimes_final_preprocessed.csv")
    Thechronicle      = pd.read_csv("Thechronicle_final_preprocessed.csv")
    ghana_news_agency = pd.read_csv("ghana_news_agency_final_preprocessed.csv")
    gbc_news          = pd.read_csv("gbc_news_final_preprocessed.csv")

    combined = pd.concat(
        [Citynewsroom, Dailyguide, Ghanaiantimes, Thechronicle,
         ghana_news_agency, gbc_news],
        ignore_index=True,
    )
    # Normalise source → lowercase
    combined["source"] = combined["source"].astype(str).str.strip().str.lower()

    # Normalise outlet_type
    STATE_KW = ["state", "public"]
    combined["outlet_type_clean"] = (
        combined["outlet_type"].astype(str).str.lower()
        .apply(lambda x: "State-owned" if any(k in x for k in STATE_KW) else "Private")
    )

    combined.drop_duplicates(subset="url", keep="first", inplace=True)
    combined["processed_text"] = combined["processed_text"].fillna("").astype(str)
    combined = combined[combined["processed_text"].str.strip().str.len() > 20].copy()
    combined.reset_index(drop=True, inplace=True)

    # Vectorise
    vectorizer = CountVectorizer(
        max_df=0.90, min_df=5, max_features=5000, ngram_range=(1, 2)
    )
    dtm   = vectorizer.fit_transform(combined["processed_text"])
    vocab = np.array(vectorizer.get_feature_names_out())

    # Term frequencies
    term_freq = np.asarray(dtm.sum(axis=0)).flatten()

    # Fit LDA
    N_TOPICS = 10
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, doc_topic_prior=0.1, topic_word_prior=0.01,
        learning_method="online", learning_decay=0.7,
        max_iter=40, random_state=42, n_jobs=-1,
    )
    doc_topic_matrix = lda.fit_transform(dtm)

    # Labels (from notebook cell 31)
    TOPIC_LABELS = {
        1:  "Media Platforms & Digital Engagement",
        2:  "National Development & Economic Policy",
        3:  "Electoral Integrity, Security & Peaceful Process",
        4:  "Presidential Leadership & International Relations",
        5:  "Parliamentary Governance & Constitutional Affairs",
        6:  "Constituency-Level Election Results & Vote Collation",
        7:  "Party Politics: NDC vs. NPP Campaign Dynamics",
        8:  "Media Institutions & Broadcasting Operations",
        9:  "Social Cohesion, Youth Engagement & Peace Advocacy",
        10: "Illegal Mining (Galamsey) & Environmental Regulation",
    }
    TOPIC_SHORT = {
        1:  "Digital Media",         2:  "Economy & Policy",
        3:  "Electoral Process",     4:  "Presidential Leadership",
        5:  "Parliament & Gov.",     6:  "Election Results",
        7:  "Party Politics",        8:  "Media Operations",
        9:  "Peace & Social Coh.",  10: "Galamsey & Environment",
    }

    combined["dominant_topic"]       = doc_topic_matrix.argmax(axis=1) + 1
    combined["dominant_topic_label"] = combined["dominant_topic"].map(TOPIC_LABELS)
    combined["topic_confidence"]     = doc_topic_matrix.max(axis=1).round(4)

    topic_counts = combined["dominant_topic"].value_counts().reindex(
        range(1, N_TOPICS + 1), fill_value=0
    )

    return (combined, lda, vectorizer, dtm, vocab, term_freq,
            doc_topic_matrix, TOPIC_LABELS, TOPIC_SHORT, topic_counts, N_TOPICS)


# ─── Load ───────────────────────────────────────────────────────────────────
try:
    (combined, lda, vectorizer, dtm, vocab, term_freq,
     doc_topic_matrix, TOPIC_LABELS, TOPIC_SHORT, topic_counts, N_TOPICS) = load_and_fit()
    DATA_OK = True
except Exception as exc:
    DATA_OK  = False
    DATA_ERR = str(exc)

# ─── Static outlet / analysis constants ─────────────────────────────────────
OUTLETS         = ["citinewsroom", "daily guide", "ghanaian times",
                   "the chronicle", "ghana news agency", "gbc"]
STATE_OUTLETS   = ["ghanaian times", "ghana news agency", "gbc"]
PRIVATE_OUTLETS = ["citinewsroom", "daily guide", "the chronicle"]
OUTLET_DISPLAY  = {
    "citinewsroom":      "Citinewsroom",
    "daily guide":       "Daily Guide",
    "ghanaian times":    "Ghanaian Times",
    "the chronicle":     "The Chronicle",
    "ghana news agency": "Ghana News Agency",
    "gbc":               "GBC",
}

NARROW_TOPICS = {4, 7, 8, 9}
BROAD_TOPICS  = {1, 2, 3, 5, 6, 10}
NPP_ALIGNED   = {3, 6, 8}
NDC_ALIGNED   = {2, 5, 10}
SHARED        = {1, 4, 7, 9}

VOTER_PRIORITIES = {
    "Economy & Cost of Living":    66,
    "Unemployment / Jobs":         68,
    "Corruption / Accountability": 15,
    "Infrastructure (Roads)":      22,
    "Healthcare":                  13,
    "Peace & Security":             4,
}
TOPIC_VOTER_MAP = {
    1: "Economy & Cost of Living",   2: "Corruption / Accountability",
    3: "Economy & Cost of Living",   4: "Unemployment / Jobs",
    5: "Corruption / Accountability",6: "Infrastructure (Roads)",
    7: "Healthcare",                 8: "Peace & Security",
    9: "Unemployment / Jobs",       10: "Infrastructure (Roads)",
}

# ─── Derived values (only when data loaded) ──────────────────────────────────
if DATA_OK:
    available_sources = set(combined["source"].unique())
    OUTLETS_FILTERED  = [o for o in OUTLETS      if o in available_sources]
    STATE_FILTERED    = [o for o in STATE_OUTLETS if o in available_sources]
    PRIV_FILTERED     = [o for o in PRIVATE_OUTLETS if o in available_sources]

    def outlet_topic_pct(outlet):
        sub = combined[combined["source"] == outlet]
        if sub.empty:
            return {t: 0.0 for t in range(1, N_TOPICS + 1)}
        return {t: sub[sub["dominant_topic"] == t].shape[0] / len(sub) * 100
                for t in range(1, N_TOPICS + 1)}

    outlet_pcts = {o: outlet_topic_pct(o) for o in OUTLETS_FILTERED}

    heat_df2 = pd.DataFrame(outlet_pcts, index=range(1, N_TOPICS + 1)).T
    heat_df2.columns = [TOPIC_SHORT[c] for c in heat_df2.columns]
    heat_df2.index   = [OUTLET_DISPLAY.get(o, o) for o in heat_df2.index]

    CLASSIFICATION = {t: ("Narrow" if t in NARROW_TOPICS else "Broad")
                      for t in range(1, N_TOPICS + 1)}
    combined["coverage_type"] = combined["dominant_topic"].map(CLASSIFICATION)
    total_narrow   = int((combined["coverage_type"] == "Narrow").sum())
    total_broad    = int((combined["coverage_type"] == "Broad").sum())
    total_articles = len(combined)

    PARTY_MAP = {t: "NPP-aligned"    for t in NPP_ALIGNED}
    PARTY_MAP.update({t: "NDC-aligned"    for t in NDC_ALIGNED})
    PARTY_MAP.update({t: "Shared/Neutral" for t in SHARED})
    combined["party_frame"] = combined["dominant_topic"].map(PARTY_MAP)
    npp_count = int((combined["party_frame"] == "NPP-aligned").sum())
    ndc_count = int((combined["party_frame"] == "NDC-aligned").sum())
    neu_count = int((combined["party_frame"] == "Shared/Neutral").sum())

    state_avg   = pd.DataFrame({o: outlet_pcts[o] for o in STATE_FILTERED}).mean(axis=1) if STATE_FILTERED else pd.Series(dtype=float)
    private_avg = pd.DataFrame({o: outlet_pcts[o] for o in PRIV_FILTERED}).mean(axis=1)  if PRIV_FILTERED  else pd.Series(dtype=float)
    divergence  = (private_avg - state_avg).sort_values() if (not state_avg.empty and not private_avg.empty) else pd.Series(dtype=float)

    voter_total = sum(VOTER_PRIORITIES.values())
    voter_norm  = {k: v / voter_total * 100 for k, v in VOTER_PRIORITIES.items()}
    media_agg   = {}
    for concern in VOTER_PRIORITIES:
        matched = [t for t, c in TOPIC_VOTER_MAP.items() if c == concern]
        media_agg[concern] = sum(int(topic_counts.loc[t]) for t in matched) / total_articles * 100
    media_norm = {k: media_agg.get(k, 0.0) for k in voter_norm}
    gaps       = {c: media_norm[c] - voter_norm[c] for c in voter_norm}

    # Matrix helpers
    SAMPLE_PER_TOPIC = 3
    sample_idx = []
    for t in range(1, N_TOPICS + 1):
        t_rows = combined[combined["dominant_topic"] == t].index.tolist()
        sample_idx.extend(t_rows[:SAMPLE_PER_TOPIC])
    sample_idx = sorted(set(sample_idx))[:30]

    row_labels = [
        f"{combined.loc[i,'dominant_topic']:02d}| "
        f"{str(combined.loc[i,'title'])[:35]:35s} "
        f"[{OUTLET_DISPLAY.get(combined.loc[i,'source'], combined.loc[i,'source'])[:4]}]"
        for i in sample_idx
    ]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview & Pipeline":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge"> TERM PROJECT · 2026</div>
        <h1>2024 Ghana Election<br>Media Coverage Analysis</h1>
        <p>A full LDA topic modelling pipeline examining how six major Ghanaian
        news outlets framed the 2024 presidential and parliamentary elections —
        from data ingestion through narrative framing and voter priority alignment.</p>
    </div>
    """, unsafe_allow_html=True)

    if not DATA_OK:
        st.error(
            f"⚠️ Could not load data. Place all `*_final_preprocessed.csv` files "
            f"in the same folder as `app.py`.\n\nError: `{DATA_ERR}`"
        )
        st.stop()

    # Metrics
    perp = lda.perplexity(dtm)
    st.markdown(
        '<div class="metric-row">'
        + metric_html(f"{len(combined):,}", "Total Articles")
        + metric_html(str(len(OUTLETS_FILTERED)), "Outlets Loaded")
        + metric_html(str(N_TOPICS), "LDA Topics (k)")
        + metric_html(f"{perp:.0f}", "Held-out Perplexity")
        + metric_html(f"{total_broad/total_articles*100:.0f}%", "Broad Coverage")
        + metric_html(f"{total_narrow/total_articles*100:.0f}%", "Narrow Coverage")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-hdr">Pipeline Architecture</div>', unsafe_allow_html=True)
    steps = [
        ("1","Scrape","Six outlets scraped: Citinewsroom · Daily Guide · Ghanaian Times · The Chronicle · GNA · GBC"),
        ("2","Pre-process","Tokenise, lemmatise, remove stopwords → `processed_text`"),
        ("3","Vectorise","CountVectorizer — 5,000 terms, unigrams + bigrams"),
        ("4","Fit LDA","k=10 · α=0.1 · β=0.01 · online · 40 iterations · random_state=42"),
        ("5","Top Words","Top-10 words per topic from `lda.components_`"),
        ("6","Label","Human-readable labels derived from top words + article inspection"),
        ("7","e-i","Dominant topic analysis across full corpus"),
        ("8","e-ii","Issue agenda comparison: state vs. private outlets"),
        ("9","e-iii","Narrow vs. Broad coverage (Patterson 1993; Iyengar & Kinder 1987)"),
        ("10","e-iv","Party narrative framing + voter priority mirror test"),
    ]
    cols = st.columns(2)
    for idx, (num, title, desc) in enumerate(steps):
        accent = "#C9A84C" if int(num) <= 6 else "#1A7A6E"
        bg_dot  = "#1A3A5C" if int(num) <= 6 else "#1A7A6E"
        with cols[idx % 2]:
            st.markdown(f"""
            <div style="background:#fff;border-radius:10px;padding:14px 18px;
                        margin-bottom:12px;border-left:3px solid {accent};
                        box-shadow:0 2px 8px rgba(26,58,92,.07);">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
                <span style="background:{bg_dot};color:#fff;border-radius:50%;
                             width:24px;height:24px;display:flex;align-items:center;
                             justify-content:center;font-size:.72rem;font-weight:700;
                             flex-shrink:0;">{num}</span>
                <span style="font-weight:600;color:#1A3A5C;font-size:.95rem;">{title}</span>
              </div>
              <div style="font-size:.82rem;color:#4A5568;padding-left:34px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Topic Labels at a Glance</div>', unsafe_allow_html=True)
    rows_html = ""
    for t in range(1, N_TOPICS + 1):
        cnt  = int(topic_counts.loc[t])
        pct  = cnt / total_articles * 100
        conf = combined[combined["dominant_topic"] == t]["topic_confidence"].mean()
        nb   = "NARROW" if t in NARROW_TOPICS else "BROAD"
        nb_c = "tag-narrow" if nb == "NARROW" else "tag-broad"
        pa   = "NPP" if t in NPP_ALIGNED else ("NDC" if t in NDC_ALIGNED else "SHARED")
        pa_c = "tag-npp" if pa == "NPP" else ("tag-ndc" if pa == "NDC" else "tag-neutral")
        rows_html += (
            f"<tr><td><b>T{t:02d}</b></td>"
            f"<td>{TOPIC_LABELS[t]}</td>"
            f"<td><b>{cnt:,}</b> <span style='color:#888'>({pct:.1f}%)</span></td>"
            f"<td>{conf:.3f}</td>"
            f"<td><span class='tag {nb_c}'>{nb}</span></td>"
            f"<td><span class='tag {pa_c}'>{pa}</span></td></tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr>
        <th>T#</th><th>Label</th><th>Articles</th>
        <th>Mean Conf.</th><th>Coverage Type</th><th>Party Alignment</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown('<hr class="fancy-div">', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <b>How to navigate:</b> Use the sidebar to move between sections.
    Each section corresponds to one part of the project pipeline.
    All figures are generated live from the fitted LDA model — no pre-saved images required.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (c) MODELLING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  (c) Modelling":
    st.markdown('<div class="sec-hdr">⚙️ Section (c) — Modelling</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    # Parameters side-by-side
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="sub-hdr">CountVectorizer Parameters</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        <b>max_df = 0.90</b> — terms in &gt;90 % of docs removed (too common)<br>
        <b>min_df = 5</b> — terms in &lt;5 docs dropped (too rare)<br>
        <b>max_features = 5,000</b> — vocabulary capped at 5,000 most frequent terms<br>
        <b>ngram_range = (1, 2)</b> — unigrams + bigrams (e.g. "john mahama", "illegal mining")
        </div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-row">'
            + metric_html(f"{dtm.shape[0]:,}", "Documents")
            + metric_html(f"{dtm.shape[1]:,}", "Vocabulary Terms")
            + metric_html(f"{dtm.nnz/(dtm.shape[0]*dtm.shape[1])*100:.3f}%", "Matrix Density")
            + "</div>", unsafe_allow_html=True
        )

    with c2:
        st.markdown('<div class="sub-hdr">LDA Hyperparameters</div>', unsafe_allow_html=True)
        params_df = pd.DataFrame({
            "Parameter": ["n_components (k)", "doc_topic_prior α", "topic_word_prior β",
                          "learning_method", "learning_decay", "max_iter", "random_state"],
            "Value":     [N_TOPICS, 0.1, 0.01, "online", 0.7, 40, 42],
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="metric-row" style="margin-top:12px;">'
            + metric_html(f"{lda.perplexity(dtm):.0f}", "Held-out Perplexity")
            + "</div>", unsafe_allow_html=True
        )

    # Top-20 terms
    st.markdown('<div class="sub-hdr">Top 20 Most Frequent Corpus Terms</div>', unsafe_allow_html=True)
    top20_idx = term_freq.argsort()[-20:][::-1]
    top20_df  = pd.DataFrame({
        "Term": vocab[top20_idx],
        "Corpus Frequency": term_freq[top20_idx].astype(int),
    })
    st.dataframe(top20_df, hide_index=True, use_container_width=True)

    st.markdown('<div class="sub-hdr">Figure c-1 — Overall Topic Distribution</div>', unsafe_allow_html=True)
    vals_c1   = [int(topic_counts.loc[t]) for t in range(1, N_TOPICS + 1)]
    labels_c1 = [f"T{t:02d}: {TOPIC_SHORT[t]}" for t in range(1, N_TOPICS + 1)]
    colors_c1 = plt.cm.Blues_r(np.linspace(0.28, 0.85, N_TOPICS))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.barh(labels_c1, vals_c1, color=colors_c1, edgecolor="white", height=0.72)
    for bar, v in zip(bars, vals_c1):
        ax.text(bar.get_width() + max(vals_c1)*0.008, bar.get_y() + bar.get_height()/2,
                f"{v:,}  ({v/total_articles*100:.1f}%)",
                va="center", fontsize=9, color=NAVY)
    ax.set_xlabel("Number of Articles", fontsize=11)
    ax.set_xlim(0, max(vals_c1) * 1.4)
    ax.set_title(f"Figure c-1: LDA Topic Distribution — All Outlets Combined\n"
                 f"n = {total_articles:,} articles · k = {N_TOPICS} topics",
                 fontsize=11, fontweight="bold", color=NAVY, pad=10)
    plt.tight_layout()
    show_fig(fig)

    st.markdown('<div class="sub-hdr">Figure c-2 — Top 12 Keywords per Topic</div>', unsafe_allow_html=True)
    TAB10 = plt.cm.tab10(np.linspace(0, 1, N_TOPICS))
    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 9))
    axes2 = axes2.flatten()
    for i in range(N_TOPICS):
        comp  = lda.components_[i]
        top12 = comp.argsort()[-12:]
        words  = vocab[top12]
        scores = comp[top12] / comp.sum() * 100
        axes2[i].barh(words, scores, color=TAB10[i], edgecolor="white")
        axes2[i].set_title(f"T{i+1:02d}: {TOPIC_SHORT[i+1]}", fontsize=8.5,
                           fontweight="bold", color=NAVY)
        axes2[i].tick_params(labelsize=7.5)
    fig2.suptitle("Figure c-2: Top 12 Keywords per LDA Topic\n"
                  "(normalised word weight within topic component)",
                  fontsize=12, fontweight="bold", color=NAVY, y=1.01)
    plt.tight_layout()
    show_fig(fig2)

    st.markdown('<div class="sub-hdr">Figure c-3 — Stacked Topic Distribution by Outlet</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    x3     = np.arange(len(OUTLETS_FILTERED))
    bottom3= np.zeros(len(OUTLETS_FILTERED))
    out_labels = [OUTLET_DISPLAY.get(o, o) for o in OUTLETS_FILTERED]
    for t_i in range(N_TOPICS):
        topic_num = t_i + 1
        vals_t = [
            combined[combined["source"] == o]["dominant_topic"].eq(topic_num).sum()
            / len(combined[combined["source"] == o]) * 100
            for o in OUTLETS_FILTERED
        ]
        ax3.bar(x3, vals_t, bottom=bottom3, label=f"T{topic_num}: {TOPIC_SHORT[topic_num]}",
                color=TAB10[t_i], edgecolor="white", width=0.62)
        bottom3 += np.array(vals_t)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(out_labels, fontsize=11)
    ax3.set_ylabel("% of Outlet Articles", fontsize=11)
    ax3.set_ylim(0, 112)
    ax3.set_title("Figure c-3: Stacked Topic Distribution by Outlet (%)",
                  fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax3.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    plt.tight_layout()
    show_fig(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (d) LABELLING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏷️  (d) Labelling":
    st.markdown('<div class="sec-hdr">🏷️ Section (d) — Topic Labelling</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    Labels derived by: (1) inspecting the top-20 words per topic component,
    (2) reading 10–15 sampled articles per cluster, and (3) cross-referencing
    with the known 2024 Ghana election context and campaign themes.
    </div>""", unsafe_allow_html=True)

    rows_html = ""
    for i, component in enumerate(lda.components_):
        top10  = ", ".join(vocab[component.argsort()[-10:][::-1]])
        t      = i + 1
        cnt    = int(topic_counts.loc[t])
        pct    = cnt / total_articles * 100
        conf   = combined[combined["dominant_topic"] == t]["topic_confidence"].mean()
        rows_html += (
            f"<tr>"
            f"<td style='font-family:IBM Plex Mono,monospace;font-weight:700;'>T{t:02d}</td>"
            f"<td><b>{TOPIC_LABELS[t]}</b></td>"
            f"<td style='font-size:.8rem;color:#555;'>{top10}</td>"
            f"<td style='text-align:right;'>{cnt:,} <span style='color:#888'>({pct:.1f}%)</span></td>"
            f"<td style='text-align:right;'>{conf:.3f}</td>"
            f"</tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr>
        <th>T#</th><th>Label</th><th>Top 10 Words</th>
        <th style='text-align:right;'>Articles</th>
        <th style='text-align:right;'>Mean Conf.</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown('<div class="sub-hdr">Figure d-1 — Outlet × Topic Heatmap</div>', unsafe_allow_html=True)
    heat_d = {}
    for outlet in OUTLETS_FILTERED:
        sub = combined[combined["source"] == outlet]
        heat_d[OUTLET_DISPLAY.get(outlet, outlet)] = {
            TOPIC_SHORT[t]: sub[sub["dominant_topic"] == t].shape[0] / len(sub) * 100
            for t in range(1, N_TOPICS + 1)
        }
    heat_d_df = pd.DataFrame(heat_d).T

    cmap_d = sns.light_palette("#2C7BB6", as_cmap=True, n_colors=10)
    fig4, ax4 = plt.subplots(figsize=(16, max(4, len(OUTLETS_FILTERED) * 0.85)))
    sns.heatmap(heat_d_df, annot=True, fmt=".1f", cmap=cmap_d,
                linewidths=0.6, linecolor="white",
                annot_kws={"size": 8.5, "weight": "bold"}, ax=ax4,
                cbar_kws={"label": "% of outlet articles", "shrink": 0.85})
    ax4.set_title("Figure d-1: Topic Distribution Heatmap by Outlet (%)\n"
                  "Each cell = % of that outlet's articles dominated by that topic",
                  fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax4.tick_params(axis="x", labelsize=8.5, rotation=38)
    ax4.tick_params(axis="y", labelsize=10, rotation=0)
    plt.tight_layout()
    show_fig(fig4)

    # Downloads
    st.markdown('<div class="sub-hdr">Download Outputs</div>', unsafe_allow_html=True)
    summary_rows = []
    for t in range(1, N_TOPICS + 1):
        comp  = lda.components_[t - 1]
        top10 = ", ".join(vocab[comp.argsort()[-10:][::-1]])
        cnt   = int(topic_counts.loc[t])
        conf  = combined[combined["dominant_topic"] == t]["topic_confidence"].mean()
        summary_rows.append({
            "Topic #": f"T{t:02d}", "Label": TOPIC_LABELS[t],
            "Top 10 Words": top10, "Article Count": cnt,
            "% Corpus": round(cnt / total_articles * 100, 1),
            "Mean Conf.": round(float(conf), 3),
        })
    summary_df = pd.DataFrame(summary_rows)
    labelled_df = combined[["source","outlet_type_clean","title","url",
                             "dominant_topic","dominant_topic_label",
                             "topic_confidence","processed_text"]]
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("⬇️  topic_summary_table.csv",
                           summary_df.to_csv(index=False).encode(),
                           "topic_summary_table.csv", "text/csv")
    with dc2:
        st.download_button("⬇️  labelled_corpus.csv",
                           labelled_df.to_csv(index=False).encode(),
                           "labelled_corpus.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: WORD CLOUDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "☁️  Word Clouds":
    st.markdown('<div class="sec-hdr">☁️ Word Clouds per Topic</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    One word cloud per LDA topic generated from <code>lda.components_</code>
    (topic-word weights). Word size ∝ term's probability within the topic.
    Each cloud uses a colour palette matched to its topic number.
    </div>""", unsafe_allow_html=True)

    # Individual topic selector
    sel_topic = st.selectbox(
        "Select a specific topic to inspect (or view all below):",
        ["All Topics"] + [f"T{t:02d}: {TOPIC_SHORT[t]}" for t in range(1, N_TOPICS + 1)],
    )

    def make_wc_for_topic(t_idx):
        """Generate a WordCloud image array for topic index t_idx (0-based)."""
        base_color = CLOUD_COLORS[t_idx % len(CLOUD_COLORS)]
        r_h = int(base_color[1:3], 16) / 255
        g_h = int(base_color[3:5], 16) / 255
        b_h = int(base_color[5:7], 16) / 255
        h, s, v = colorsys.rgb_to_hsv(r_h, g_h, b_h)

        def _cf(word, font_size, position, orientation, random_state=None, **kw):
            brightness = 0.55 + 0.40 * (font_size / 100)
            r, g, b = colorsys.hsv_to_rgb(h, max(0.3, s), min(1.0, brightness))
            return (int(r * 255), int(g * 255), int(b * 255))

        weights = {vocab[i]: float(lda.components_[t_idx, i]) for i in range(len(vocab))}
        return WordCloud(
            width=620, height=390, background_color="white",
            max_words=60, prefer_horizontal=0.85,
            color_func=_cf, collocations=False, random_state=42,
        ).generate_from_frequencies(weights)

    if sel_topic != "All Topics":
        t_num = int(sel_topic[1:3])
        wc    = make_wc_for_topic(t_num - 1)
        fig_s, ax_s = plt.subplots(figsize=(10, 5), facecolor=BG_WHITE)
        ax_s.imshow(wc, interpolation="bilinear")
        ax_s.axis("off")
        ax_s.set_title(f"T{t_num:02d}: {TOPIC_LABELS[t_num]}",
                       fontsize=13, fontweight="bold",
                       color=CLOUD_COLORS[(t_num - 1) % len(CLOUD_COLORS)])
        plt.tight_layout()
        show_fig(fig_s)
        # Show top-15 words as a table
        comp    = lda.components_[t_num - 1]
        top15   = comp.argsort()[-15:][::-1]
        wc_tbl  = pd.DataFrame({
            "Term": vocab[top15],
            "Topic Weight": comp[top15].round(2),
            "Norm. P(term|topic) %": (comp[top15] / comp.sum() * 100).round(4),
        })
        st.markdown(f'<div class="sub-hdr">Top 15 Terms — T{t_num:02d}: {TOPIC_SHORT[t_num]}</div>',
                    unsafe_allow_html=True)
        st.dataframe(wc_tbl, hide_index=True, use_container_width=True)

    else:
        st.markdown('<div class="sub-hdr">All Topics — 2 × 5 Grid</div>', unsafe_allow_html=True)
        n_cols, n_rows = 5, 2
        fig_wc, axes_wc = plt.subplots(n_rows, n_cols,
                                        figsize=(n_cols * 4.6, n_rows * 3.8),
                                        facecolor=BG_WHITE)
        axes_wc = axes_wc.flatten()
        for t_idx in range(N_TOPICS):
            wc_img = make_wc_for_topic(t_idx)
            axes_wc[t_idx].imshow(wc_img, interpolation="bilinear")
            axes_wc[t_idx].axis("off")
            axes_wc[t_idx].set_facecolor(BG_WHITE)
            axes_wc[t_idx].set_title(
                f"T{t_idx+1:02d}: {TOPIC_SHORT[t_idx+1]}",
                fontsize=9.5, fontweight="bold",
                color=CLOUD_COLORS[t_idx % len(CLOUD_COLORS)], pad=5,
            )
        for ax_wc in axes_wc[N_TOPICS:]:
            ax_wc.set_visible(False)
        fig_wc.suptitle(
            "LDA Topic Word Clouds — All Outlets Combined\n"
            "2024 Ghana Election Media Coverage  ·  Word size ∝ topic-word weight",
            fontsize=13, fontweight="bold", color=NAVY, y=1.01,
        )
        plt.tight_layout()
        show_fig(fig_wc)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (e-i) DOMINANT TOPICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  (e-i) Dominant Topics":
    st.markdown('<div class="sec-hdr">📊 Section (e-i) — Dominant Topics</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    <b>Research question:</b> What were the most prominent topics covered across
    the entire election period? Do they align with the major pre-election survey
    issues such as the economy, unemployment, infrastructure, and corruption?
    </div>""", unsafe_allow_html=True)

    top1_t = int(topic_counts.idxmax())
    top2_t = int(topic_counts.nlargest(2).index[1])
    top3_t = int(topic_counts.nlargest(3).index[2])
    st.markdown(
        '<div class="metric-row">'
        + metric_html(f"T{top1_t:02d}", "Top Topic")
        + metric_html(f"T{top2_t:02d}", "2nd Topic")
        + metric_html(f"T{top3_t:02d}", "3rd Topic")
        + metric_html(f"{topic_counts.max()/total_articles*100:.1f}%", "Top Topic Share")
        + metric_html(f"{combined['topic_confidence'].mean():.3f}", "Mean Confidence")
        + metric_html(f"{combined['topic_confidence'].median():.3f}", "Median Confidence")
        + "</div>", unsafe_allow_html=True
    )

    # Tabular summary
    st.markdown('<div class="sub-hdr">Topic Distribution Table</div>', unsafe_allow_html=True)
    tbl_rows = ""
    for t in topic_counts.sort_values(ascending=False).index:
        cnt  = int(topic_counts.loc[t])
        pct  = cnt / total_articles * 100
        conf = combined[combined["dominant_topic"] == t]["topic_confidence"].mean()
        bar  = "▓" * int(pct / 1.5)
        tbl_rows += (
            f"<tr><td><b>T{t:02d}</b></td><td>{TOPIC_LABELS[t]}</td>"
            f"<td style='text-align:right;'>{cnt:,}</td>"
            f"<td style='text-align:right;'>{pct:.1f}%</td>"
            f"<td style='text-align:right;'>{conf:.3f}</td>"
            f"<td style='color:{NAVY};font-size:.75rem;'>{bar}</td></tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr>
        <th>T#</th><th>Label</th>
        <th style='text-align:right;'>Articles</th>
        <th style='text-align:right;'>%</th>
        <th style='text-align:right;'>Mean Conf.</th>
        <th>Proportion</th>
    </tr></thead>
    <tbody>{tbl_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    # Figure e-i(a) — Annotated bar
    st.markdown('<div class="sub-hdr">Figure e-i(a) — Annotated Dominant Topics Bar</div>', unsafe_allow_html=True)
    sorted_topics = topic_counts.sort_values(ascending=True)
    labels_sorted = [f"T{t:02d}: {TOPIC_SHORT[t]}" for t in sorted_topics.index]
    vals_sorted   = sorted_topics.values.astype(int)
    colors_e1     = [CUSTOM_PAL[(t - 1) % 10] for t in sorted_topics.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(labels_sorted, vals_sorted, color=colors_e1,
                   edgecolor="white", height=0.72, linewidth=0.8)
    for bar, v, t_idx in zip(bars, vals_sorted, sorted_topics.index):
        conf = combined[combined["dominant_topic"] == t_idx]["topic_confidence"].mean()
        pct  = v / total_articles * 100
        ax.text(bar.get_width() + max(vals_sorted) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)  conf={conf:.2f}",
                va="center", fontsize=8.5, color=NAVY, fontweight="500")
    ax.set_xlabel("Number of Articles", fontsize=11, labelpad=8)
    ax.set_xlim(0, max(vals_sorted) * 1.47)
    ax.set_title(
        f"Figure e-i(a): LDA Dominant Topics — All Outlets Combined\n"
        f"2024 Ghana Election Media Coverage · n = {total_articles:,} articles\n"
        "Annotated: article count · corpus share (%) · mean topic confidence",
        fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.tick_params(axis="y", labelsize=9.5)
    plt.tight_layout()
    show_fig(fig)

    # Figure e-i(b) — Radar
    st.markdown('<div class="sub-hdr">Figure e-i(b) — Topic Prominence Radar</div>', unsafe_allow_html=True)
    angles   = np.linspace(0, 2 * np.pi, N_TOPICS, endpoint=False).tolist()
    angles  += angles[:1]
    pcts_rad = [float(topic_counts.loc[t]) / total_articles * 100 for t in range(1, N_TOPICS + 1)]
    pcts_rad+= pcts_rad[:1]

    fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax_r.plot(angles, pcts_rad, color=NAVY, linewidth=2.2)
    ax_r.fill(angles, pcts_rad, color=NAVY, alpha=0.18)
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels([TOPIC_SHORT[t] for t in range(1, N_TOPICS + 1)],
                          fontsize=8.5, color=SLATE)
    ax_r.set_rlabel_position(15)
    ax_r.set_yticks([5, 10, 15, 20])
    ax_r.set_yticklabels(["5%", "10%", "15%", "20%"], fontsize=8, color=SLATE)
    ax_r.grid(color=SLATE, linestyle="--", linewidth=0.6, alpha=0.4)
    ax_r.set_title("Figure e-i(b): Topic Prominence Radar\n"
                   "(% of corpus dominated by each topic)",
                   fontsize=12, fontweight="bold", color=NAVY, pad=20)
    plt.tight_layout()
    rc1, rc2, rc3 = st.columns([1, 2, 1])
    with rc2:
        show_fig(fig_r)

    st.markdown(f"""
    <div class="warn-box">
    <b>Key Finding:</b>
    The top three dominant topics are T{top1_t:02d} ({TOPIC_LABELS[top1_t]}),
    T{top2_t:02d} ({TOPIC_LABELS[top2_t]}), and T{top3_t:02d} ({TOPIC_LABELS[top3_t]}).
    Together they account for
    {(topic_counts.loc[top1_t]+topic_counts.loc[top2_t]+topic_counts.loc[top3_t])/total_articles*100:.1f}%
    of all articles, reflecting the election's central campaign themes.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (e-ii) AGENDA BY OUTLET
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📰  (e-ii) Agenda by Outlet":
    st.markdown('<div class="sec-hdr">📰 Section (e-ii) — Issue Agenda by Outlet</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    <b>Research question:</b> How does the "issue agenda" vary by outlet?
    Compare state-owned outlets (Ghanaian Times · GNA · GBC) with private/online outlets
    (Citinewsroom · Daily Guide · The Chronicle) on topics like corruption,
    economic hardship, and galamsey. Are any topics unique to specific outlets?
    </div>""", unsafe_allow_html=True)

    # Top-3 table
    st.markdown('<div class="sub-hdr">Top-3 Topics per Outlet</div>', unsafe_allow_html=True)
    rows_h = ""
    for outlet in OUTLETS_FILTERED:
        sub   = combined[combined["source"] == outlet]
        top3  = sub["dominant_topic"].value_counts().head(3)
        otype = "STATE" if outlet in STATE_OUTLETS else "PRIVATE"
        oc    = "tag-state" if otype == "STATE" else "tag-private"
        lbls  = [f"T{t}: {TOPIC_SHORT[t]} ({cnt/len(sub)*100:.1f}%)"
                 for t, cnt in top3.items()]
        while len(lbls) < 3:
            lbls.append("—")
        dname = OUTLET_DISPLAY.get(outlet, outlet)
        rows_h += (
            f"<tr><td><b>{dname}</b> <span class='tag {oc}'>{otype}</span></td>"
            f"<td>{lbls[0]}</td><td>{lbls[1]}</td><td>{lbls[2]}</td></tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr><th>Outlet</th><th>#1 Topic</th><th>#2 Topic</th><th>#3 Topic</th></tr></thead>
    <tbody>{rows_h}</tbody>
    </table>""", unsafe_allow_html=True)

    # Figure e-ii(a) — Heatmap
    st.markdown('<div class="sub-hdr">Figure e-ii(a) — Annotated Outlet Topic Heatmap</div>', unsafe_allow_html=True)
    cmap_h = sns.light_palette("#2C7BB6", as_cmap=True, n_colors=10)
    fig_h, ax_h = plt.subplots(figsize=(17, max(4.5, len(OUTLETS_FILTERED) * 0.88)))
    vmax_h = float(heat_df2.values.max()) if heat_df2.values.size > 0 else 100.0
    sns.heatmap(heat_df2, annot=True, fmt=".1f", cmap=cmap_h,
                linewidths=0.8, linecolor="white",
                annot_kws={"size": 9, "weight": "semibold"}, ax=ax_h,
                cbar_kws={"label": "% of outlet articles", "shrink": 0.85},
                vmin=0, vmax=vmax_h)
    ax_h.set_title(
        "Figure e-ii(a): Topic Distribution Heatmap by Outlet (%)\n"
        "Each cell = % of that outlet's articles dominated by that topic",
        fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax_h.set_xlabel("LDA Topics", fontsize=11)
    ax_h.set_ylabel("Media Outlets", fontsize=11)
    ax_h.tick_params(axis="x", labelsize=9, rotation=42)
    ax_h.tick_params(axis="y", labelsize=10, rotation=0)
    for i, outlet in enumerate(OUTLETS_FILTERED):
        tag   = "STATE" if outlet in STATE_OUTLETS else "PRIVATE"
        color = STATE_COLOR if outlet in STATE_OUTLETS else PRIVATE_COLOR
        ax_h.annotate(f"[{tag}]", xy=(-0.01, i + 0.5),
                      xycoords=("axes fraction", "data"),
                      ha="right", va="center", fontsize=8, color=color, fontweight="bold")
    plt.tight_layout()
    show_fig(fig_h)

    # Figure e-ii(b) — Grouped bars
    st.markdown('<div class="sub-hdr">Figure e-ii(b) — Top-3 Topics per Outlet</div>', unsafe_allow_html=True)
    n_out  = len(OUTLETS_FILTERED)
    fig_b, axes_arr = plt.subplots(1, n_out, figsize=(4 * n_out, 6), sharey=False)
    axes_arr = np.atleast_1d(axes_arr)
    for ax_i, outlet in zip(axes_arr, OUTLETS_FILTERED):
        sub   = combined[combined["source"] == outlet]
        if sub.empty:
            ax_i.set_visible(False); continue
        top3  = sub["dominant_topic"].value_counts().head(3)
        total_o = len(sub)
        t_lbls = [f"T{t}\n{TOPIC_SHORT[t]}" for t in top3.index]
        vals   = [cnt / total_o * 100 for cnt in top3.values]
        while len(t_lbls) < 3:
            t_lbls.append("—\n—"); vals.append(0.0)
        cmap_o = plt.cm.Blues if outlet in STATE_OUTLETS else plt.cm.Oranges
        bar_c  = [cmap_o(0.45 + 0.2 * i) for i in range(3)]
        clr    = STATE_COLOR if outlet in STATE_OUTLETS else PRIVATE_COLOR
        bars_b = ax_i.bar(range(3), vals, color=bar_c, edgecolor="white", width=0.65)
        for b, v in zip(bars_b, vals):
            if v > 0:
                ax_i.text(b.get_x() + b.get_width()/2, v + 0.3,
                          f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=NAVY)
        ax_i.set_xticks(range(3))
        ax_i.set_xticklabels(t_lbls, fontsize=7.5, ha="center")
        ax_i.set_ylabel("% of Outlet Articles" if outlet == OUTLETS_FILTERED[0] else "", fontsize=9)
        otype = "STATE-OWNED" if outlet in STATE_OUTLETS else "PRIVATE"
        ax_i.set_title(f"{OUTLET_DISPLAY.get(outlet,outlet)}\n[{otype}]",
                       fontsize=10, fontweight="bold", color=clr, pad=6)
        ax_i.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 50)
    fig_b.suptitle("Figure e-ii(b): Top 3 Topics per Outlet\n"
                   "Blue = state-owned  ·  Orange = private / online",
                   fontsize=11, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    show_fig(fig_b)

    # Figure e-ii(c) — Divergence lollipop
    if not divergence.empty:
        st.markdown('<div class="sub-hdr">Figure e-ii(c) — State vs. Private Agenda Divergence</div>',
                    unsafe_allow_html=True)
        fig_d, ax_d = plt.subplots(figsize=(12, 7))
        for y_pos, (tidx, val) in enumerate(divergence.items()):
            color = PRIVATE_COLOR if val > 0 else STATE_COLOR
            ax_d.plot([0, val], [y_pos, y_pos], color=color, lw=2.2, solid_capstyle="round")
            ax_d.scatter(val, y_pos, color=color, s=110, zorder=5)
            ax_d.text(val + (0.35 if val >= 0 else -0.35), y_pos,
                      f"{val:+.1f}pp", va="center",
                      ha="left" if val >= 0 else "right",
                      fontsize=8.8, color=color, fontweight="600")
        ax_d.axvline(0, color=SLATE, lw=1.2, linestyle="--", alpha=0.7)
        ax_d.set_yticks(range(N_TOPICS))
        ax_d.set_yticklabels([f"T{tidx:02d}: {TOPIC_SHORT[tidx]}"
                               for tidx in divergence.index], fontsize=9)
        ax_d.set_xlabel("Percentage-point difference  (Private − State)", fontsize=10.5)
        ax_d.set_title("Figure e-ii(c): State vs. Private Outlet Agenda Divergence\n"
                       "Positive = private outlets cover MORE  ·  Negative = state outlets cover MORE",
                       fontsize=12, fontweight="bold", color=NAVY, pad=10)
        ax_d.legend(handles=[
            mpatches.Patch(color=PRIVATE_COLOR, label="Private / Online over-represents"),
            mpatches.Patch(color=STATE_COLOR,   label="State-owned over-represents"),
        ], fontsize=9, loc="lower right")
        plt.tight_layout()
        show_fig(fig_d)

        max_priv_t  = int(divergence[divergence > 0].idxmax()) if (divergence > 0).any() else None
        max_state_t = int(divergence[divergence < 0].idxmin()) if (divergence < 0).any() else None
        if max_priv_t and max_state_t:
            st.markdown(f"""
            <div class="warn-box">
            <b>Key Finding:</b> Private outlets over-represent
            <b>T{max_priv_t:02d}: {TOPIC_SHORT[max_priv_t]}</b>
            by {divergence.max():.1f} pp relative to state-owned outlets.
            State-owned outlets devote comparatively more space to
            <b>T{max_state_t:02d}: {TOPIC_SHORT[max_state_t]}</b>
            ({divergence.min():.1f} pp gap), reflecting different editorial mandates.
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (e-iii) NARROW vs BROAD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  (e-iii) Narrow vs. Broad":
    st.markdown('<div class="sec-hdr">🔍 Section (e-iii) — Narrow vs. Broad Coverage</div>',
                unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    narrow_pct = total_narrow / total_articles * 100
    broad_pct  = total_broad  / total_articles * 100

    st.markdown("""
    <div class="insight-box">
    <b>Framework (Patterson 1993; Iyengar & Kinder 1987):</b><br>
    <span class="tag tag-narrow">NARROW</span>
    = process / horse-race — polling, candidate rallies, voting logistics, results collation<br>
    <span class="tag tag-broad">BROAD</span>
    = substance / policy — economy, illegal mining, education, governance
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="metric-row">'
        + metric_html(f"{narrow_pct:.1f}%", "Narrow Coverage")
        + metric_html(f"{broad_pct:.1f}%", "Broad Coverage")
        + metric_html(f"1 : {total_broad/max(total_narrow,1):.2f}", "Narrow : Broad Ratio")
        + metric_html(str(len(NARROW_TOPICS)), "Narrow Topics")
        + metric_html(str(len(BROAD_TOPICS)), "Broad Topics")
        + "</div>", unsafe_allow_html=True
    )

    # Classification table
    rationale = {
        1: "Civic discourse — policy substance",   2: "Campaign policy positions",
        3: "Incumbent record — policy",            4: "Electoral process logistics",
        5: "Environmental policy substance",        6: "Economic policy substance",
        7: "Voting day operations & security",      8: "Electoral integrity procedures",
        9: "Unity appeals & leadership posture",   10: "Constituency-level races",
    }
    rows_nb = ""
    for t in range(1, N_TOPICS + 1):
        tag  = "NARROW" if t in NARROW_TOPICS else "BROAD"
        tc   = "tag-narrow" if tag == "NARROW" else "tag-broad"
        rows_nb += (
            f"<tr><td><b>T{t:02d}</b></td><td>{TOPIC_LABELS[t]}</td>"
            f"<td><span class='tag {tc}'>{tag}</span></td>"
            f"<td style='font-size:.82rem;color:#555;'>{rationale[t]}</td></tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr><th>T#</th><th>Label</th><th>Type</th><th>Rationale</th></tr></thead>
    <tbody>{rows_nb}</tbody>
    </table>""", unsafe_allow_html=True)

    # Figure e-iii(a) — Bar + donut
    st.markdown('<div class="sub-hdr">Figure e-iii(a) — Classification Overview</div>', unsafe_allow_html=True)
    fig10 = plt.figure(figsize=(15, 7))
    gs10  = gridspec.GridSpec(1, 2, figure=fig10, width_ratios=[1.4, 1])
    ax_l  = fig10.add_subplot(gs10[0])
    ax_r  = fig10.add_subplot(gs10[1])

    tc_list = sorted([(t, int(topic_counts.loc[t])) for t in range(1, N_TOPICS + 1)],
                     key=lambda x: x[1], reverse=True)
    lbls_e3 = [f"T{t:02d}: {TOPIC_SHORT[t]}" for t, _ in tc_list]
    cnts_e3 = [c for _, c in tc_list]
    clrs_e3 = [CRIMSON if CLASSIFICATION[t] == "Narrow" else TEAL for t, _ in tc_list]

    bars_e3 = ax_l.barh(lbls_e3, cnts_e3, color=clrs_e3, edgecolor="white", height=0.72)
    for bar, v, (t, _) in zip(bars_e3, cnts_e3, tc_list):
        tag2 = "NARROW" if CLASSIFICATION[t] == "Narrow" else "BROAD"
        ax_l.text(bar.get_width() + max(cnts_e3)*0.01,
                  bar.get_y() + bar.get_height()/2,
                  f"{v:,}  [{tag2}]", va="center", fontsize=8.5,
                  color=CRIMSON if tag2 == "NARROW" else TEAL)
    ax_l.set_xlabel("Number of Articles", fontsize=10)
    ax_l.set_title("Topic Volume by Coverage Type", fontsize=10.5, fontweight="bold", color=NAVY)
    ax_l.set_xlim(0, max(cnts_e3) * 1.5)
    ax_l.legend(handles=[
        mpatches.Patch(color=CRIMSON, label="Narrow (process-focused)"),
        mpatches.Patch(color=TEAL,    label="Broad (policy/substance)"),
    ], fontsize=9)

    if total_narrow + total_broad > 0:
        wedges, _, autos = ax_r.pie(
            [total_narrow, total_broad],
            labels=["Narrow\n(process)", "Broad\n(policy)"],
            colors=[CRIMSON, TEAL], autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2.5},
            textprops={"fontsize": 11})
        for at in autos:
            at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
        ax_r.add_patch(plt.Circle((0, 0), 0.60, fc=BG_WHITE))
        ratio_txt = f"1 : {total_broad/max(total_narrow,1):.1f}\nN : B"
        ax_r.text(0, 0, ratio_txt, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=NAVY)
    ax_r.set_title("Narrow / Broad Ratio\n(All outlets combined)",
                   fontsize=10.5, fontweight="bold", color=NAVY, pad=8)
    fig10.suptitle("Figure e-iii(a): Narrow vs. Broad Election Coverage Classification\n"
                   "Narrow = process-focused  ·  Broad = policy/substance",
                   fontsize=12, fontweight="bold", color=NAVY)
    plt.tight_layout()
    show_fig(fig10)

    # Figure e-iii(b) — Ratio by outlet
    st.markdown('<div class="sub-hdr">Figure e-iii(b) — Narrow / Broad Ratio by Outlet</div>',
                unsafe_allow_html=True)
    outlet_nb = {}
    for outlet in OUTLETS_FILTERED:
        sub = combined[combined["source"] == outlet]
        if len(sub) > 0:
            n = (sub["coverage_type"] == "Narrow").sum() / len(sub) * 100
            b = (sub["coverage_type"] == "Broad").sum()  / len(sub) * 100
            outlet_nb[outlet] = {"Narrow": n, "Broad": b}
        else:
            outlet_nb[outlet] = {"Narrow": 0.0, "Broad": 0.0}

    fig11, ax11 = plt.subplots(figsize=(11, 5.5))
    x11  = np.arange(len(OUTLETS_FILTERED))
    nv   = [outlet_nb[o]["Narrow"] for o in OUTLETS_FILTERED]
    bv   = [outlet_nb[o]["Broad"]  for o in OUTLETS_FILTERED]
    w11  = 0.36
    b1e3 = ax11.bar(x11 - w11/2, nv, w11, label="Narrow (process)",
                    color=CRIMSON, edgecolor="white")
    b2e3 = ax11.bar(x11 + w11/2, bv, w11, label="Broad (policy)",
                    color=TEAL, edgecolor="white")
    for b, v in zip(list(b1e3) + list(b2e3), nv + bv):
        if v > 0:
            ax11.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%",
                      ha="center", va="bottom", fontsize=9, color=NAVY)
    ax11.set_xticks(x11)
    ax11.set_xticklabels([OUTLET_DISPLAY.get(o, o) for o in OUTLETS_FILTERED], fontsize=10.5)
    ax11.set_ylabel("% of Outlet Articles", fontsize=10.5)
    ax11.set_ylim(0, max(nv + bv) * 1.26 if max(nv + bv) > 0 else 100)
    ax11.legend(fontsize=10)
    ax11.set_title("Figure e-iii(b): Narrow vs. Broad Coverage Ratio by Outlet",
                   fontsize=12, fontweight="bold", color=NAVY, pad=10)
    plt.tight_layout()
    show_fig(fig11)

    st.markdown(f"""
    <div class="warn-box">
    <b>Key Finding:</b> Broad (policy/substance) coverage accounts for
    <b>{broad_pct:.1f}%</b> of all articles — a Narrow : Broad ratio of
    <b>1 : {total_broad/max(total_narrow,1):.2f}</b>. This suggests Ghanaian
    election media in 2024 leaned substantially toward policy substance over
    horse-race process reporting, a positive indicator for democratic discourse quality.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: (e-iv) PARTY FRAMING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚖️  (e-iv) Party Framing":
    st.markdown('<div class="sec-hdr">⚖️ Section (e-iv) — Party Framing & Voter Mirror</div>',
                unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    <b>Research questions:</b><br>
    1. What topics did the incumbent NPP focus on vs. the opposition NDC?<br>
    2. Do the media's top issues mirror voter priorities (Global Info Analytics / Musa Danquah, Dec 2023)?<br>
    3. Were critical public issues under- or over-covered relative to voter concern weights?
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="metric-row">'
        + metric_html(f"{npp_count/total_articles*100:.1f}%", "NPP-Aligned Articles")
        + metric_html(f"{ndc_count/total_articles*100:.1f}%", "NDC-Aligned Articles")
        + metric_html(f"{neu_count/total_articles*100:.1f}%", "Neutral / Shared")
        + metric_html(f"+{max(gaps.values()):.1f}pp", "Largest Over-Coverage")
        + metric_html(f"{min(gaps.values()):.1f}pp", "Largest Under-Coverage")
        + "</div>", unsafe_allow_html=True
    )

    # Party alignment table
    rationale_pa = {
        3: "NPP's record in office, Akufo-Addo legacy",
        6: "Economic policy & anti-corruption pledges",
        8: "Incumbent emphasis on peaceful process",
        2: "Mahama campaign trail & NDC policy promises",
        5: "Galamsey — major NDC attack line against NPP",
        10:"Constituency races: NDC grassroots mobilisation",
        1: "Civic/media discourse — both parties",
        4: "EC processes — neutral electoral administration",
        7: "Voting day logistics — neutral",
        9: "Unity/leadership appeals — both parties",
    }
    rows_pa = ""
    for t in sorted(list(NPP_ALIGNED) + list(NDC_ALIGNED) + list(SHARED)):
        pa   = "NPP-aligned" if t in NPP_ALIGNED else ("NDC-aligned" if t in NDC_ALIGNED else "Shared/Neutral")
        pc   = "tag-npp" if t in NPP_ALIGNED else ("tag-ndc" if t in NDC_ALIGNED else "tag-neutral")
        cnt  = int(topic_counts.loc[t])
        pct  = cnt / total_articles * 100
        rows_pa += (
            f"<tr><td><b>T{t:02d}</b></td><td>{TOPIC_LABELS[t]}</td>"
            f"<td><span class='tag {pc}'>{pa}</span></td>"
            f"<td style='text-align:right;'>{cnt:,} ({pct:.1f}%)</td>"
            f"<td style='font-size:.81rem;color:#555;'>{rationale_pa.get(t,'')}</td></tr>"
        )
    st.markdown(f"""
    <table class="topic-table">
    <thead><tr>
        <th>T#</th><th>Label</th><th>Alignment</th>
        <th style='text-align:right;'>Articles</th><th>Rationale</th>
    </tr></thead>
    <tbody>{rows_pa}</tbody>
    </table>""", unsafe_allow_html=True)

    # Figure e-iv(a) — Stacked bar + donut
    st.markdown('<div class="sub-hdr">Figure e-iv(a) — Party Narrative Framing</div>', unsafe_allow_html=True)
    outlet_frame = {}
    for outlet in OUTLETS_FILTERED:
        sub = combined[combined["source"] == outlet]
        n   = len(sub)
        if n > 0:
            outlet_frame[outlet] = {
                "NPP-aligned":    sub[sub["party_frame"] == "NPP-aligned"].shape[0]    / n * 100,
                "NDC-aligned":    sub[sub["party_frame"] == "NDC-aligned"].shape[0]    / n * 100,
                "Shared/Neutral": sub[sub["party_frame"] == "Shared/Neutral"].shape[0] / n * 100,
            }
        else:
            outlet_frame[outlet] = {"NPP-aligned":0.0, "NDC-aligned":0.0, "Shared/Neutral":0.0}
    frame_df = pd.DataFrame(outlet_frame).T

    fig12, axes12 = plt.subplots(1, 2, figsize=(16, 6))
    ax12a = axes12[0]
    x12   = np.arange(len(OUTLETS_FILTERED))
    bot12 = np.zeros(len(OUTLETS_FILTERED))
    for col, clr in zip(["NPP-aligned","NDC-aligned","Shared/Neutral"],
                        [NPP_COLOR, NDC_COLOR, "#AAAAAA"]):
        vals_col = [frame_df.loc[o, col] if o in frame_df.index else 0 for o in OUTLETS_FILTERED]
        bars12   = ax12a.bar(x12, vals_col, bottom=bot12, color=clr, label=col,
                             edgecolor="white", width=0.6)
        for b, v, b0 in zip(bars12, vals_col, bot12):
            if v > 4.5:
                ax12a.text(b.get_x() + b.get_width()/2, b0 + v/2,
                           f"{v:.1f}%", ha="center", va="center",
                           fontsize=8.5, color="white", fontweight="bold")
        bot12 += np.array(vals_col)
    ax12a.set_xticks(x12)
    ax12a.set_xticklabels([OUTLET_DISPLAY.get(o, o) for o in OUTLETS_FILTERED], fontsize=10)
    ax12a.set_ylabel("% of Outlet Articles", fontsize=10)
    ax12a.set_ylim(0, 110)
    ax12a.legend(fontsize=9, loc="upper right")
    ax12a.set_title("Party Narrative Framing by Outlet", fontsize=10.5, fontweight="bold", color=NAVY)

    ax12b = axes12[1]
    w12, _, a12 = ax12b.pie(
        [npp_count, ndc_count, neu_count],
        labels=["NPP-aligned","NDC-aligned","Neutral / Shared"],
        colors=[NPP_COLOR, NDC_COLOR, "#AAAAAA"],
        autopct="%1.1f%%", startangle=120,
        wedgeprops={"edgecolor":"white","linewidth":2.5},
        textprops={"fontsize":10})
    for at in a12:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    ax12b.add_patch(plt.Circle((0,0), 0.55, fc=BG_WHITE))
    ax12b.set_title("Overall Party Frame Balance\n(All outlets combined)",
                    fontsize=10.5, fontweight="bold", color=NAVY)
    fig12.suptitle("Figure e-iv(a): NPP vs. NDC Narrative Framing in Election Coverage",
                   fontsize=11, fontweight="bold", color=NAVY)
    plt.tight_layout()
    show_fig(fig12)

    # Figure e-iv(b) — Mirror test
    st.markdown('<div class="sub-hdr">Figure e-iv(b) — Media Agenda vs. Voter Priorities</div>',
                unsafe_allow_html=True)
    concerns   = list(voter_norm.keys())
    voter_vals = [voter_norm[c] for c in concerns]
    media_vals = [media_norm[c] for c in concerns]

    fig13, ax13 = plt.subplots(figsize=(13, 6))
    x13, w13 = np.arange(len(concerns)), 0.36
    b1m = ax13.bar(x13 - w13/2, voter_vals, w13,
                   label="Voter Priorities (Global Info Analytics, Dec 2023)",
                   color=GOLD, edgecolor="white", alpha=0.9)
    b2m = ax13.bar(x13 + w13/2, media_vals, w13,
                   label="Media Coverage Share (%)",
                   color=NAVY, edgecolor="white", alpha=0.9)
    for b, v in zip(list(b1m) + list(b2m), voter_vals + media_vals):
        if v > 0:
            ax13.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%",
                      ha="center", va="bottom", fontsize=8.5, color=NAVY)
    for i, c in enumerate(concerns):
        gap13 = media_norm[c] - voter_norm[c]
        if abs(gap13) > 4:
            ax13.annotate("", xy=(i + w13/2, media_vals[i]),
                          xytext=(i - w13/2, voter_vals[i]),
                          arrowprops=dict(arrowstyle="->",
                                         color=CRIMSON if gap13 < 0 else TEAL, lw=1.6))
    ax13.set_xticks(x13)
    ax13.set_xticklabels(concerns, fontsize=9, rotation=20, ha="right")
    ax13.set_ylabel("% Share", fontsize=10.5)
    ax13.set_ylim(0, max(voter_vals + media_vals) * 1.27)
    ax13.set_title("Figure e-iv(b): Media Agenda vs. Voter Priorities — Mirror Test\n"
                   "Gold = voter concern weights  ·  Navy = media topic share  ·  Arrows = >4 pp gap",
                   fontsize=11, fontweight="bold", color=NAVY, pad=10)
    ax13.legend(fontsize=9.5)
    plt.tight_layout()
    show_fig(fig13)

    # Figure e-iv(c) — Coverage gap
    st.markdown('<div class="sub-hdr">Figure e-iv(c) — Coverage Gap Analysis</div>', unsafe_allow_html=True)
    gaps_sorted = dict(sorted(gaps.items(), key=lambda x: x[1]))
    colors_gap  = [CRIMSON if v < 0 else TEAL for v in gaps_sorted.values()]
    fig14, ax14 = plt.subplots(figsize=(12, 6))
    bars14 = ax14.barh(list(gaps_sorted.keys()), list(gaps_sorted.values()),
                       color=colors_gap, edgecolor="white", height=0.62)
    for bar, v in zip(bars14, gaps_sorted.values()):
        if v != 0:
            ax14.text(v + (0.2 if v >= 0 else -0.2),
                      bar.get_y() + bar.get_height()/2,
                      f"{v:+.1f} pp", va="center",
                      ha="left" if v >= 0 else "right",
                      fontsize=9.5, color=CRIMSON if v < 0 else TEAL, fontweight="600")
    ax14.axvline(0, color=SLATE, lw=1.5, linestyle="-", alpha=0.6)
    ax14.set_xlabel("Coverage Gap  (Media % − Voter Concern %)", fontsize=10.5)
    ax14.set_title("Figure e-iv(c): Media Coverage Gap Analysis\n"
                   "Teal = over-represented  ·  Red = under-represented vs voter priorities",
                   fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax14.legend(handles=[
        mpatches.Patch(color=TEAL,    label="Over-represented (media > voter priority)"),
        mpatches.Patch(color=CRIMSON, label="Under-represented (media < voter priority)"),
    ], fontsize=9.5, loc="lower right")
    max_g = max(abs(min(gaps_sorted.values())), abs(max(gaps_sorted.values())))
    ax14.set_xlim(-max_g * 1.28, max_g * 1.28)
    plt.tight_layout()
    show_fig(fig14)

    most_over  = max(gaps, key=lambda x: gaps[x])
    most_under = min(gaps, key=lambda x: gaps[x])
    imbalance  = abs(npp_count - ndc_count) / total_articles * 100
    st.markdown(f"""
    <div class="warn-box">
    <b>Key Findings:</b><br>
    • <b>Most over-represented:</b> {most_over} (+{max(gaps.values()):.1f} pp) —
      media devoted disproportionate attention here relative to voter-ranked priority.<br>
    • <b>Most under-represented:</b> {most_under} ({min(gaps.values()):.1f} pp) —
      a significant voter concern that received comparatively less editorial space.<br>
    • NPP vs. NDC narrative balance: <b>{npp_count/total_articles*100:.1f}%</b> vs.
      <b>{ndc_count/total_articles*100:.1f}%</b> —
      {'near-parity between incumbent and opposition framing.' if imbalance < 5 else 'a notable imbalance in incumbent vs. opposition coverage framing.'}
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LDA MATRICES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔢  LDA Matrices":
    st.markdown('<div class="sec-hdr">🔢 LDA Matrix Displays</div>', unsafe_allow_html=True)
    if not DATA_OK:
        st.error("Data not loaded."); st.stop()

    st.markdown("""
    <div class="insight-box">
    Three analytical matrices for the combined corpus:
    <b>K1</b> — Document-Topic Matrix (stratified sample, 30 docs × 10 topics) ·
    <b>K2</b> — Document-Term Matrix (top-40 terms × same 30 docs) ·
    <b>K3</b> — Term-Topic Matrix (top-30 discriminative terms × 10 topics).<br>
    Full matrices are available for CSV download below.
    </div>""", unsafe_allow_html=True)

    tab_k1, tab_k2, tab_k3 = st.tabs(["K1 · Document-Topic", "K2 · Document-Term", "K3 · Term-Topic"])

    # ── K1: Document-Topic ──────────────────────────────────────────────────
    with tab_k1:
        st.markdown('<div class="sub-hdr">Figure K1 — Document-Topic Matrix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        Shape: n_docs × k_topics. Each row sums to 1.0.
        Sample: 3 documents per dominant topic (stratified), capped at 30.
        Row label = <b>dominant_topic | truncated title | [outlet]</b>.
        </div>""", unsafe_allow_html=True)

        dt_sample  = doc_topic_matrix[sample_idx, :]
        col_labels = [f"T{t:02d}\n{TOPIC_SHORT[t][:12]}" for t in range(1, N_TOPICS + 1)]
        dt_df      = pd.DataFrame(dt_sample, index=row_labels, columns=col_labels)

        fig_k1, ax_k1 = plt.subplots(figsize=(18, max(10, len(sample_idx) * 0.45)))
        sns.heatmap(dt_df, annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 7, "weight": "bold"}, ax=ax_k1,
                    cbar_kws={"label": "Topic proportion (0–1)", "shrink": 0.80},
                    vmin=0, vmax=1)
        ax_k1.set_title(
            "Figure K1: Document-Topic Matrix (stratified sample, n=30)\n"
            "Cell = proportion attributed to topic  ·  Each row sums to 1.0\n"
            "Row: dominant topic # | title | [outlet abbrev.]",
            fontsize=11, fontweight="bold", color=NAVY, pad=10)
        ax_k1.set_xlabel("LDA Topics", fontsize=10)
        ax_k1.set_ylabel("Documents", fontsize=9)
        ax_k1.tick_params(axis="x", labelsize=8, rotation=0)
        ax_k1.tick_params(axis="y", labelsize=7.5, rotation=0)
        plt.tight_layout()
        show_fig(fig_k1)

        # Metrics
        st.markdown(
            '<div class="metric-row">'
            + metric_html(f"{doc_topic_matrix.shape[0]:,}", "Total Documents")
            + metric_html(f"{doc_topic_matrix.shape[1]}", "Topics")
            + metric_html(f"{doc_topic_matrix.max(axis=1).mean():.3f}", "Mean Max-Topic Prob.")
            + metric_html(f"{doc_topic_matrix.max(axis=1).min():.3f}", "Min Max-Topic Prob.")
            + "</div>", unsafe_allow_html=True
        )

        # CSV download
        dt_full = pd.DataFrame(
            doc_topic_matrix,
            columns=[f"T{t:02d}_{TOPIC_SHORT[t]}" for t in range(1, N_TOPICS + 1)],
        )
        dt_full.insert(0, "source",           combined["source"].values)
        dt_full.insert(1, "dominant_topic",   combined["dominant_topic"].values)
        dt_full.insert(2, "topic_confidence", combined["topic_confidence"].values)
        st.download_button(
            "⬇️  Download full Document-Topic matrix CSV",
            dt_full.to_csv(index=True, index_label="doc_id").encode(),
            "matrix_document_topic.csv", "text/csv",
        )

    # ── K2: Document-Term ───────────────────────────────────────────────────
    with tab_k2:
        st.markdown('<div class="sub-hdr">Figure K2 — Document-Term Matrix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        Full DTM shape: n_docs × vocab_size (sparse).
        Displayed: top-40 corpus terms × the same 30 sampled documents.
        Cell = raw term count (0 = term absent).
        </div>""", unsafe_allow_html=True)

        TOP_TERMS    = 40
        top_term_idx = term_freq.argsort()[-TOP_TERMS:][::-1]
        term_cols    = vocab[top_term_idx]
        dtm_sub      = np.asarray(dtm[sample_idx, :][:, top_term_idx].todense())
        dtm_df       = pd.DataFrame(dtm_sub, index=row_labels, columns=term_cols)

        fig_k2, ax_k2 = plt.subplots(figsize=(22, max(10, len(sample_idx) * 0.45)))
        sns.heatmap(dtm_df, annot=True, fmt="g", cmap="YlOrBr",
                    linewidths=0.3, linecolor="white",
                    annot_kws={"size": 6.5, "weight": "bold"}, ax=ax_k2,
                    cbar_kws={"label": "Raw term count", "shrink": 0.80})
        ax_k2.set_title(
            f"Figure K2: Document-Term Matrix (top-{TOP_TERMS} terms × sampled {len(sample_idx)} docs)\n"
            "Cell = raw term count  ·  0 = term absent from document",
            fontsize=11, fontweight="bold", color=NAVY, pad=10)
        ax_k2.set_xlabel(f"Terms (top-{TOP_TERMS} by corpus frequency)", fontsize=10)
        ax_k2.set_ylabel("Documents", fontsize=9)
        ax_k2.tick_params(axis="x", labelsize=7.5, rotation=45)
        ax_k2.tick_params(axis="y", labelsize=7.5, rotation=0)
        plt.tight_layout()
        show_fig(fig_k2)

        st.markdown(
            '<div class="metric-row">'
            + metric_html(f"{dtm.shape[0]:,}", "Total Documents")
            + metric_html(f"{dtm.shape[1]:,}", "Vocabulary Terms")
            + metric_html(f"{dtm.nnz/(dtm.shape[0]*dtm.shape[1])*100:.3f}%", "Matrix Density")
            + metric_html(f"{term_freq.max():.0f}", "Max Term Freq.")
            + "</div>", unsafe_allow_html=True
        )

        dtm_export = pd.DataFrame(
            np.asarray(dtm[:, top_term_idx].todense()), columns=term_cols
        )
        dtm_export.insert(0, "source",         combined["source"].values)
        dtm_export.insert(1, "dominant_topic", combined["dominant_topic"].values)
        st.download_button(
            f"⬇️  Download Document-Term matrix CSV (top-{TOP_TERMS} terms)",
            dtm_export.to_csv(index=True, index_label="doc_id").encode(),
            "matrix_document_term_top40.csv", "text/csv",
        )

    # ── K3: Term-Topic ──────────────────────────────────────────────────────
    with tab_k3:
        st.markdown('<div class="sub-hdr">Figure K3 — Term-Topic Matrix</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        Source: <code>lda.components_</code> (shape k × vocab), row-normalised so
        each topic sums to 1.0. Displayed as P(term|topic) × 100 (%).
        Terms shown = union of top-15 per topic (≤ 30 total), sorted by corpus frequency.
        </div>""", unsafe_allow_html=True)

        topic_word_norm = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        TOP_WORDS_TT    = 15
        union_idx = set()
        for t_i in range(N_TOPICS):
            top_i = lda.components_[t_i].argsort()[-TOP_WORDS_TT:][::-1]
            union_idx.update(top_i.tolist())
        union_idx  = sorted(union_idx, key=lambda i: -term_freq[i])[:30]
        union_terms = vocab[np.array(union_idx)]
        tt_pct     = topic_word_norm[:, np.array(union_idx)] * 100
        row_lbl_k3 = [f"T{t:02d}: {TOPIC_SHORT[t]}" for t in range(1, N_TOPICS + 1)]
        tt_df      = pd.DataFrame(tt_pct, index=row_lbl_k3, columns=union_terms)

        fig_k3, ax_k3 = plt.subplots(figsize=(22, 7))
        sns.heatmap(tt_df, annot=True, fmt=".2f", cmap="Greens",
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 7, "weight": "bold"}, ax=ax_k3,
                    cbar_kws={"label": "P(term|topic) × 100 (%)", "shrink": 0.80})
        ax_k3.set_title(
            f"Figure K3: Term-Topic Matrix (top-{len(union_terms)} discriminative terms)\n"
            "Cell = P(term|topic) × 100  ·  Rows = LDA topics  ·  Columns = vocabulary terms\n"
            "Each topic row normalised to sum to 100% across full vocabulary",
            fontsize=11, fontweight="bold", color=NAVY, pad=10)
        ax_k3.set_xlabel("Vocabulary Terms", fontsize=10)
        ax_k3.set_ylabel("LDA Topics", fontsize=10)
        ax_k3.tick_params(axis="x", labelsize=8, rotation=45)
        ax_k3.tick_params(axis="y", labelsize=9, rotation=0)
        plt.tight_layout()
        show_fig(fig_k3)

        st.markdown(
            '<div class="metric-row">'
            + metric_html(f"{lda.components_.shape[0]}", "Topics")
            + metric_html(f"{lda.components_.shape[1]:,}", "Vocabulary Terms")
            + metric_html(str(len(union_terms)), "Terms Displayed")
            + metric_html(f"{topic_word_norm.max()*100:.3f}%", "Max P(term|topic)")
            + "</div>", unsafe_allow_html=True
        )

        tt_full = pd.DataFrame(
            topic_word_norm * 100,
            index  =[f"T{t:02d}_{TOPIC_SHORT[t]}" for t in range(1, N_TOPICS + 1)],
            columns=vocab,
        )
        st.download_button(
            "⬇️  Download full Term-Topic matrix CSV",
            tt_full.to_csv(index=True, index_label="topic").encode(),
            "matrix_term_topic.csv", "text/csv",
        )
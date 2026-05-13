"""
Football Scouting & Analytics
================================
General football analytics powered by Understat data.
Covers EPL, La Liga, Bundesliga, Serie A, Ligue 1.
"""

import ast
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("understat_data")

AVAILABLE_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
LEAGUE_LABELS = {
    "EPL":        "Premier League",
    "La_liga":    "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "Serie A",
    "Ligue_1":    "Ligue 1",
}

DARK_BG  = "#0e1117"
CARD_BG  = "#1a1d27"
ACCENT   = "#5fffb0"
TEXT     = "#f0f0f0"
SUBTLE   = "#8899aa"
RED      = "#e63946"
BLUE     = "#457b9d"
GOLD     = "#f4d03f"
PURPLE   = "#9b59b6"
ORANGE   = "#e67e22"

SITUATION_COLORS = {
    "OpenPlay":   BLUE,
    "FromCorner": GOLD,
    "SetPiece":   ORANGE,
    "DirectFreekick": PURPLE,
    "Penalty":    RED,
}
SHOT_TYPE_MARKERS = {
    "RightFoot": "circle",
    "LeftFoot":  "diamond",
    "Head":      "square",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Football Scout | CoachFPL", page_icon="⚽", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top:1.2rem; }}
  .metric-card {{
    background:{CARD_BG}; border-radius:10px; padding:14px 18px;
    margin-bottom:8px; border-left:3px solid {ACCENT};
  }}
  .metric-card h4    {{ color:{ACCENT}; margin:0 0 4px; font-size:11px; letter-spacing:1px; text-transform:uppercase; }}
  .metric-card p     {{ color:{TEXT}; margin:0; font-size:22px; font-weight:700; }}
  .metric-card small {{ color:{SUBTLE}; font-size:11px; }}
  .section-label {{ font-size:11px; font-weight:800; color:{ACCENT}; letter-spacing:1px; margin-bottom:4px; }}
  .scout-header {{
    background:linear-gradient(135deg,#1a1d27 0%,#0d1b2a 100%);
    border:1px solid #2a2d3a; border-radius:12px; padding:20px 24px; margin-bottom:16px;
  }}
  .scout-header h2 {{ color:{TEXT}; margin:0 0 8px; font-size:26px; letter-spacing:1px; }}
  .scout-tag   {{ display:inline-block; background:#2a2d3a; color:{ACCENT}; border-radius:4px; padding:3px 8px; font-size:11px; font-weight:700; margin-right:6px; }}
  .scout-stat  {{ display:inline-block; color:{SUBTLE}; font-size:12px; margin-right:16px; }}
  .scout-stat b {{ color:{TEXT}; }}
  .sim-card {{
    background:{CARD_BG}; border:1px solid #2a2d3a; border-radius:8px;
    padding:10px 14px; margin-bottom:6px;
  }}
  .sim-card-name  {{ color:{TEXT};   font-size:13px; font-weight:700; }}
  .sim-card-team  {{ color:{SUBTLE}; font-size:11px; }}
  .sim-score {{ float:right; color:{ACCENT}; font-weight:800; font-size:14px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_old_players_format(df: pd.DataFrame, season: str, league: str) -> pd.DataFrame:
    """Handle early-season CSVs where each row has a dict-string in 'players' column."""
    rows = []
    for _, row in df.iterrows():
        raw = str(row.get("players", ""))
        # strip outer quotes if present
        raw = raw.strip('"\'')
        try:
            d = ast.literal_eval(raw)
            d["season"] = season
            d["league"] = league
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def load_league_players(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_players.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Detect broken format (early scraper versions stored dict strings)
    if "players" in df.columns and "success" in df.columns:
        df = _parse_old_players_format(df, season, league)
        if df.empty:
            return None
    for c in ["xG","xA","npxG","npxA","npg","goals","assists","time",
              "shots","key_passes","xGChain","xGBuildup"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=600)
def load_league_table(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_table.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.rename(columns={
        "Team":"team","W":"wins","D":"draws","L":"loses",
        "PTS":"actual_pts","NPxG":"npxG","NPxGA":"npxGA",
        "NPxGD":"npxGD","G":"scored","GA":"missed",
    })
    for c in ["xG","xGA","npxG","npxGA","npxGD","xPTS","actual_pts",
              "wins","draws","loses","PPDA","OPPDA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["xGD"]  = (df["xG"]  - df["xGA"]).round(2)
    df["luck"] = (df["actual_pts"] - df["xPTS"]).round(2)
    return df

@st.cache_data(ttl=600)
def load_team_history(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_team_history.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "ppda" in df.columns:
        def _parse(val):
            try:
                d = ast.literal_eval(str(val))
                a, b = float(d.get("att",0)), float(d.get("def",0))
                return round(a/b, 2) if b > 0 else None
            except Exception:
                return None
        df["ppda_coef"] = df["ppda"].apply(_parse)
    return df

@st.cache_data(ttl=600)
def load_player_shots(player_id) -> pd.DataFrame | None:
    p = DATA_DIR / "players" / str(player_id) / "player_shots.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["X","Y","xG","minute"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def available_seasons(league: str) -> list[str]:
    d = DATA_DIR / league
    if not d.exists():
        return []
    return sorted([s.name for s in d.iterdir()
                   if s.is_dir() and (s / "league_table.csv").exists()], reverse=True)

def available_leagues() -> list[str]:
    return [lg for lg in AVAILABLE_LEAGUES if (DATA_DIR / lg).exists()]

def fmt(val, d=2):
    try:
        return f"{float(val):.{d}f}"
    except Exception:
        return str(val)

def plotly_dark(**kw):
    return dict(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                font=dict(color=TEXT, family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=50, b=40), **kw)

def colour_val(val, threshold=0):
    try:
        v = float(val)
        if v > threshold: return f"color:{ACCENT}"
        if v < threshold: return f"color:{RED}"
    except Exception:
        pass
    return ""

def pct_color(p: float) -> str:
    if p >= 80: return ACCENT
    if p >= 60: return "#2ecc71"
    if p >= 40: return GOLD
    if p >= 20: return ORANGE
    return RED

def calc_percentile(value, series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return 50.0
    return float((s < value).sum() / len(s) * 100)

def enrich_player_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mins = df["time"].replace(0, np.nan)
    p90  = mins / 90

    df["npxG90"]       = (df["npxG"]       / p90).round(2)
    df["xA90"]         = (df["xA"]         / p90).round(2)
    df["goals90"]      = (df["goals"]       / p90).round(2)
    df["assists90"]    = (df["assists"]     / p90).round(2)
    df["goal_diff"]    = (df["goals"] - df["xG"]).round(2)
    df["assist_diff"]  = (df["assists"] - df["xA"]).round(2)
    df["npxg_xa90"]    = (df["npxG90"] + df["xA90"]).round(2)

    if "shots" in df.columns:
        df["shots90"]       = (df["shots"] / p90).round(2)
        df["npxg_per_shot"] = (df["npxG"] / df["shots"].replace(0, np.nan)).round(2)
        df["goals_per_shot"]= (df["goals"] / df["shots"].replace(0, np.nan)).round(2)

    if "key_passes" in df.columns:
        df["kp90"] = (df["key_passes"] / p90).round(2)

    if "npg" in df.columns:
        df["npg90"]   = (df["npg"]   / p90).round(2)
        df["npg_diff"]= (df["npg"]   - df["npxG"]).round(2)

    if "xGChain" in df.columns:
        df["xGChain90"]   = (df["xGChain"]   / p90).round(2)

    if "xGBuildup" in df.columns:
        df["xGBuildup90"] = (df["xGBuildup"] / p90).round(2)

    # Involvement = xGChain - npxG (contribution beyond own shots/assists)
    if "xGChain90" in df.columns and "npxG90" in df.columns:
        df["involvement90"] = (df["xGChain90"] - df["npxG90"]).round(2)

    return df


# ── Pitch drawing ─────────────────────────────────────────────────────────────

def pitch_shapes_half():
    """Shapes for the attacking half of the pitch (x: 0.5→1.0 in Understat coords)."""
    # Understat coords: X=0 (left goal) to X=1 (right goal), Y=0→1 top to bottom
    # We'll show attacking half: X from 0.5 to 1.0
    return [
        # Pitch outline (attacking half)
        dict(type="rect", x0=0.5, y0=0, x1=1.01, y1=1,
             line=dict(color="white", width=2), fillcolor="#2d5a27"),
        # Penalty box
        dict(type="rect", x0=0.83, y0=0.21, x1=1.0, y1=0.79,
             line=dict(color="white", width=1.5), fillcolor="rgba(0,0,0,0)"),
        # 6-yard box
        dict(type="rect", x0=0.94, y0=0.36, x1=1.0, y1=0.64,
             line=dict(color="white", width=1.2), fillcolor="rgba(0,0,0,0)"),
        # Goal
        dict(type="rect", x0=1.0, y0=0.42, x1=1.03, y1=0.58,
             line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0.1)"),
        # Centre line
        dict(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
             line=dict(color="white", width=1.5, dash="dot")),
        # Penalty spot
        dict(type="circle", x0=0.879, y0=0.486, x1=0.881, y1=0.514,
             line=dict(color="white", width=1.5), fillcolor="white"),
        # Penalty arc
        dict(type="path",
             path="M 0.83 0.35 Q 0.76 0.5 0.83 0.65",
             line=dict(color="white", width=1.2)),
    ]

def pitch_layout_half():
    return dict(
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#2d5a27",
        font=dict(color=TEXT),
        xaxis=dict(range=[0.47, 1.06], visible=False, scaleanchor="y", scaleratio=0.68),
        yaxis=dict(range=[-0.02, 1.02], visible=False),
        margin=dict(l=10, r=10, t=45, b=10),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚽ Football Scout</p>', unsafe_allow_html=True)
    st.markdown("General football analytics across Europe's top 5 leagues.")
    st.divider()

    leagues = available_leagues()
    if not leagues:
        st.error("No data found.")
        st.stop()

    league = st.selectbox("League", leagues, format_func=lambda x: LEAGUE_LABELS.get(x, x))
    seasons = available_seasons(league)
    if not seasons:
        st.error(f"No seasons for {league}.")
        st.stop()

    season = st.selectbox("Season", seasons, format_func=lambda s: f"{s}/{str(int(s)+1)[2:]}")
    st.divider()

    st.markdown('<p class="section-label">Season Compare</p>', unsafe_allow_html=True)
    compare_seasons = st.multiselect(
        "Seasons to compare", seasons,
        default=seasons[:5] if len(seasons) >= 5 else seasons,
        help="Used in the Season Trends tab",
    )
    st.divider()
    min_minutes = st.slider("Min minutes (players)", 200, 2000, 900, 100)


# ── Header ────────────────────────────────────────────────────────────────────
df_main = load_league_table(league, season)
c1, c2 = st.columns([3,1])
with c1:
    st.title("⚽ Football Scout")
    st.caption(f"{LEAGUE_LABELS.get(league,league)}  ·  {season}/{str(int(season)+1)[2:]}")
with c2:
    if df_main is not None:
        st.metric("Teams", len(df_main))

if df_main is None:
    st.error("No data for this season.")
    st.stop()

tab_teams, tab_players, tab_press, tab_trends = st.tabs([
    "🏟️ Team xG", "👤 Player Scouting", "⚡ Pressing", "📈 Season Trends",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEAM xG
# ══════════════════════════════════════════════════════════════════════════════
with tab_teams:
    df = df_main.copy().sort_values("xGD", ascending=False)

    k1,k2,k3,k4 = st.columns(4)
    for col,label,val,sub in [
        (k1,"Best Attack (xG)",  df.nlargest(1,"xG").iloc[0]["team"],    f'{fmt(df.nlargest(1,"xG").iloc[0]["xG"])} xG'),
        (k2,"Best Defence (xGA)",df.nsmallest(1,"xGA").iloc[0]["team"],  f'{fmt(df.nsmallest(1,"xGA").iloc[0]["xGA"])} xGA'),
        (k3,"Luckiest Team",     df.nlargest(1,"luck").iloc[0]["team"],  f'+{fmt(df.nlargest(1,"luck").iloc[0]["luck"])} pts'),
        (k4,"Unluckiest Team",   df.nsmallest(1,"luck").iloc[0]["team"], f'{fmt(df.nsmallest(1,"luck").iloc[0]["luck"])} pts'),
    ]:
        col.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{val}</p><small>{sub}</small></div>',
                     unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<p class="section-label">xG Difference — ranked</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=df["xGD"], y=df["team"], orientation="h",
            marker_color=[ACCENT if v>=0 else RED for v in df["xGD"]],
            text=[f"{v:.2f}" for v in df["xGD"]], textposition="outside",
            textfont=dict(size=10, color=TEXT),
        ))
        fig.update_layout(**plotly_dark(height=520),
            xaxis=dict(title="xGD", gridcolor="#2a2d3a", zeroline=True, zerolinecolor=SUBTLE),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<p class="section-label">Luck Table — xPTS vs Actual</p>', unsafe_allow_html=True)
        dl = df.sort_values("luck")
        fig2 = go.Figure(go.Bar(
            x=dl["luck"], y=dl["team"], orientation="h",
            marker_color=[RED if v<0 else ACCENT for v in dl["luck"]],
            text=[f"{v:.2f}" for v in dl["luck"]], textposition="outside",
            textfont=dict(size=10, color=TEXT),
        ))
        fig2.update_layout(**plotly_dark(height=520),
            xaxis=dict(title="Pts − xPTS", gridcolor="#2a2d3a", zeroline=True, zerolinecolor=SUBTLE),
            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-label">Full xG Table</p>', unsafe_allow_html=True)
    show_cols = [c for c in ["team","wins","draws","loses","xG","xGA","xGD","npxG","npxGA","xPTS","actual_pts","luck"] if c in df.columns]
    df_show = df[show_cols].round(2).reset_index(drop=True)
    df_show.insert(0,"#",range(1,len(df_show)+1))
    num_cols = [c for c in df_show.columns if c not in ["#","team"]]
    styled = (df_show.style
        .map(lambda v: colour_val(v),             subset=["xGD"]  if "xGD"  in df_show.columns else [])
        .map(lambda v: colour_val(v, threshold=3), subset=["luck"] if "luck" in df_show.columns else [])
        .format({c:"{:.2f}" for c in num_cols})
        .set_properties(**{"background-color":CARD_BG,"color":TEXT,"border-color":"#2a2d3a"}))
    st.dataframe(styled, use_container_width=True, height=420)

    st.divider()
    st.markdown('<p class="section-label">xG vs xGA — Team Quadrant</p>', unsafe_allow_html=True)
    mx,my = df["xG"].median(), df["xGA"].median()
    fig3 = go.Figure()
    fig3.add_shape(type="line", x0=mx,x1=mx, y0=df["xGA"].min()-2,y1=df["xGA"].max()+2,
                   line=dict(color=SUBTLE,dash="dot",width=1))
    fig3.add_shape(type="line", x0=df["xG"].min()-2,x1=df["xG"].max()+2, y0=my,y1=my,
                   line=dict(color=SUBTLE,dash="dot",width=1))
    fig3.add_trace(go.Scatter(
        x=df["xG"], y=df["xGA"], mode="markers+text",
        text=df["team"], textposition="top center", textfont=dict(size=9,color=TEXT),
        marker=dict(size=14,color=df["xGD"],
                    colorscale=[[0,RED],[0.5,SUBTLE],[1,ACCENT]],
                    showscale=True, colorbar=dict(title="xGD",tickfont=dict(color=TEXT)),
                    line=dict(width=1,color=DARK_BG)),
        hovertemplate="<b>%{text}</b><br>xG: %{x:.2f}<br>xGA: %{y:.2f}<extra></extra>",
    ))
    for txt,x,y in [
        ("Good attack\nGood defence", df["xG"].max()-1, df["xGA"].min()+1),
        ("Good attack\nBad defence",  df["xG"].max()-1, df["xGA"].max()-1),
        ("Bad attack\nGood defence",  df["xG"].min()+1, df["xGA"].min()+1),
        ("Bad attack\nBad defence",   df["xG"].min()+1, df["xGA"].max()-1),
    ]:
        fig3.add_annotation(x=x,y=y,text=txt,showarrow=False,
                            font=dict(size=9,color=SUBTLE),align="center")
    fig3.update_layout(**plotly_dark(height=480),
        xaxis=dict(title="xG (attack)",gridcolor="#2a2d3a"),
        yaxis=dict(title="xGA (defence — lower is better)",gridcolor="#2a2d3a"))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PLAYER SCOUTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_players:
    df_p_raw = load_league_players(league, season)
    if df_p_raw is None:
        st.warning("No player data for this season.")
        st.stop()

    df_p = enrich_player_df(df_p_raw)
    df_p = df_p[df_p["time"].fillna(0) >= min_minutes].copy()

    name_col = next((c for c in ["player_name","name"]  if c in df_p.columns), None)
    team_col = next((c for c in ["team_title","team"]   if c in df_p.columns), None)
    pos_col  = next((c for c in ["position","pos"]      if c in df_p.columns), None)

    if pos_col:
        all_pos   = sorted(df_p[pos_col].dropna().unique().tolist())
        positions = st.multiselect("Filter by position", all_pos, default=all_pos, key="pos_filter")
        if positions:
            df_p = df_p[df_p[pos_col].isin(positions)]

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">npxG / 90  vs  xA / 90</p>', unsafe_allow_html=True)
    if name_col and not df_p.empty:
        custom = list(zip(df_p[name_col], df_p[team_col] if team_col else [""]*len(df_p)))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_p["npxG90"], y=df_p["xA90"], mode="markers",
            marker=dict(size=df_p["time"]/35,
                        color=df_p["npxG90"],
                        colorscale=[[0,"#1a1d27"],[0.5,BLUE],[1,ACCENT]],
                        showscale=True, colorbar=dict(title="npxG/90",tickfont=dict(color=TEXT)),
                        opacity=0.8, line=dict(width=0.5,color=DARK_BG)),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>npxG/90: %{x:.2f}<br>xA/90: %{y:.2f}<extra></extra>",
        ))
        for _,row in df_p.nlargest(12,"npxG90").iterrows():
            fig_sc.add_annotation(x=row["npxG90"],y=row["xA90"],
                text=row[name_col].split()[-1],showarrow=False,
                font=dict(size=8,color=TEXT),yshift=10)
        fig_sc.add_hline(y=df_p["xA90"].median(),   line=dict(color=SUBTLE,dash="dot",width=1))
        fig_sc.add_vline(x=df_p["npxG90"].median(), line=dict(color=SUBTLE,dash="dot",width=1))
        fig_sc.update_layout(**plotly_dark(height=500),
            xaxis=dict(title="npxG per 90",gridcolor="#2a2d3a"),
            yaxis=dict(title="xA per 90",  gridcolor="#2a2d3a"))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    cl, cr = st.columns(2)
    if name_col and team_col:
        with cl:
            st.markdown('<p class="section-label">Top 15 — npxG / 90</p>', unsafe_allow_html=True)
            t = df_p.nlargest(15,"npxG90")[[name_col,team_col,"npxG90","xA90","time"]].round(2)
            t.columns=["Player","Team","npxG/90","xA/90","Mins"]
            t.insert(0,"#",range(1,len(t)+1))
            st.dataframe(t.style.format({"npxG/90":"{:.2f}","xA/90":"{:.2f}","Mins":"{:.0f}"})
                .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                use_container_width=True,height=420,hide_index=True)
        with cr:
            st.markdown('<p class="section-label">Top 15 — xA / 90</p>', unsafe_allow_html=True)
            t2 = df_p.nlargest(15,"xA90")[[name_col,team_col,"xA90","npxG90","time"]].round(2)
            t2.columns=["Player","Team","xA/90","npxG/90","Mins"]
            t2.insert(0,"#",range(1,len(t2)+1))
            st.dataframe(t2.style.format({"xA/90":"{:.2f}","npxG/90":"{:.2f}","Mins":"{:.0f}"})
                .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                use_container_width=True,height=420,hide_index=True)

    st.divider()
    st.markdown('<p class="section-label">Over / Underperformance — Goals vs xG</p>', unsafe_allow_html=True)
    df_perf = df_p.dropna(subset=["goal_diff"]).copy()
    df_chart= pd.concat([df_perf.nlargest(10,"goal_diff"),df_perf.nsmallest(10,"goal_diff")]).drop_duplicates().sort_values("goal_diff")
    fig_perf= go.Figure(go.Bar(
        x=df_chart["goal_diff"],
        y=df_chart[name_col] if name_col else df_chart.index,
        orientation="h",
        marker_color=[ACCENT if v>=0 else RED for v in df_chart["goal_diff"]],
        text=[f"{v:.2f}" for v in df_chart["goal_diff"]],textposition="outside",
        textfont=dict(size=9,color=TEXT),
        hovertemplate="<b>%{y}</b><br>Goals − xG: %{x:.2f}<extra></extra>",
    ))
    fig_perf.update_layout(**plotly_dark(height=460),
        xaxis=dict(title="Goals − xG",gridcolor="#2a2d3a",zeroline=True,zerolinecolor=SUBTLE),
        yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig_perf, use_container_width=True)
    st.caption("🟢 Overperformers may regress  ·  🔴 Underperformers may improve")


    # ══════════════════════════════════════════════════════════════════════════
    #  PLAYER SCOUTING CARD
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-label">🎴 Player Scouting Card</p>', unsafe_allow_html=True)

    if not name_col:
        st.info("Player name column not found.")
    else:
        all_names = sorted(df_p[name_col].dropna().unique().tolist())
        sc1, sc2 = st.columns([2,1])
        with sc1:
            scout_player = st.selectbox("Select player", ["— select —"] + all_names, key="scout_player")
        with sc2:
            compare_mode = st.radio("Compare vs", ["All outfield","Same position"], horizontal=True, key="scout_compare")

        if scout_player and scout_player != "— select —":
            prows = df_p[df_p[name_col] == scout_player]
            if prows.empty:
                st.info("Not enough minutes this season.")
            else:
                p = prows.iloc[0]
                pool = df_p[df_p[pos_col] == p[pos_col]] if (compare_mode=="Same position" and pos_col and pd.notna(p.get(pos_col))) else df_p

                p_name    = p[name_col]
                p_team    = p[team_col] if team_col and pd.notna(p.get(team_col)) else ""
                p_pos     = p[pos_col]  if pos_col  and pd.notna(p.get(pos_col))  else ""
                p_mins    = int(p["time"])    if pd.notna(p["time"]) else 0
                p_games   = int(p["games"])   if "games"   in p.index and pd.notna(p.get("games"))   else "—"
                p_goals   = int(p["goals"])   if pd.notna(p.get("goals",  np.nan)) else "—"
                p_assists = int(p["assists"]) if pd.notna(p.get("assists",np.nan)) else "—"
                sl        = f"{season}/{str(int(season)+1)[2:]}"

                st.markdown(f"""
                <div class="scout-header">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                      <span class="scout-tag">{p_pos}</span>
                      <span class="scout-tag" style="color:{SUBTLE}">{sl}</span>
                      <h2>{p_name}</h2>
                      <span class="scout-stat">🏟️ <b>{p_team}</b></span>
                      <span class="scout-stat">⏱ <b>{p_mins:,}</b> mins</span>
                      <span class="scout-stat">🎮 <b>{p_games}</b> games</span>
                      <span class="scout-stat">⚽ <b>{p_goals}</b> goals</span>
                      <span class="scout-stat">🎯 <b>{p_assists}</b> assists</span>
                    </div>
                    <div style="color:{SUBTLE};font-size:11px;text-align:right">
                      Percentile vs<br><b style="color:{TEXT}">{compare_mode}</b><br>
                      {LEAGUE_LABELS.get(league,league)} · {len(pool)} players
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── All metric categories ─────────────────────────────────────
                # (label, column, higher_is_better)
                CATEGORIES = {
                    "⚡ SHOOTING": [
                        ("npxG / 90",         "npxG90",        True),
                        ("Goals / 90",         "goals90",       True),
                        ("np Goals / 90",      "npg90",         True),
                        ("Shots / 90",         "shots90",       True),
                        ("npxG per Shot",      "npxg_per_shot", True),
                        ("Goals per Shot",     "goals_per_shot",True),
                    ],
                    "🎯 FINISHING": [
                        ("Goals vs xG",        "goal_diff",     True),
                        ("np Goals vs npxG",   "npg_diff",      True),
                        ("Assists vs xA",      "assist_diff",   True),
                    ],
                    "🔑 CREATING": [
                        ("xA / 90",            "xA90",          True),
                        ("Assists / 90",       "assists90",     True),
                        ("Key Passes / 90",    "kp90",          True),
                    ],
                    "📊 INVOLVEMENT": [
                        ("npxG + xA / 90",     "npxg_xa90",     True),
                        ("xG Chain / 90",      "xGChain90",     True),
                        ("xG Buildup / 90",    "xGBuildup90",   True),
                        ("Involvement / 90",   "involvement90", True),
                    ],
                    "📋 VOLUME": [
                        ("Games Played",       "games",         True),
                        ("Minutes",            "time",          True),
                        ("Yellow Cards",       "yellow_cards",  False),
                    ],
                }

                def build_card_chart(cat_name, metric_defs):
                    rows = []
                    for label, col, higher in metric_defs:
                        if col not in pool.columns or col not in p.index:
                            continue
                        val = p[col]
                        if pd.isna(val):
                            continue
                        raw_pct = calc_percentile(val, pool[col])
                        pct = raw_pct if higher else (100 - raw_pct)
                        rows.append((label, val, pct))
                    if not rows:
                        return None

                    names  = [r[0] for r in rows]
                    vals   = [r[1] for r in rows]
                    pcts   = [r[2] for r in rows]
                    h      = max(120, len(rows)*50 + 65)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=names, x=[100]*len(rows), orientation="h",
                        marker_color="rgba(255,255,255,0.07)",
                        showlegend=False, hoverinfo="skip", width=0.55,
                    ))
                    fig.add_trace(go.Bar(
                        y=names, x=pcts, orientation="h",
                        marker_color=[pct_color(p_) for p_ in pcts],
                        showlegend=False, width=0.55,
                        hovertemplate=[f"<b>{n}</b><br>Value: {v:.2f}<br>Percentile: {int(round(pc))}<extra></extra>"
                                       for n,v,pc in zip(names,vals,pcts)],
                    ))
                    for nm,v,pc in zip(names,vals,pcts):
                        fig.add_annotation(y=nm, x=-1,   text=f"{v:.2f}", xanchor="right", showarrow=False, font=dict(color=TEXT,size=10))
                        fig.add_annotation(y=nm, x=101,  text=str(int(round(pc))), xanchor="left", showarrow=False, font=dict(color=pct_color(pc),size=11))
                    fig.update_layout(
                        barmode="overlay",
                        title=dict(text=cat_name, font=dict(color=ACCENT,size=12), x=0),
                        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                        font=dict(color=TEXT), height=h,
                        margin=dict(l=165,r=55,t=40,b=10),
                        xaxis=dict(range=[-20,115],visible=False),
                        yaxis=dict(tickfont=dict(size=10),tickcolor=TEXT,autorange="reversed"),
                    )
                    return fig

                # Two-column card layout
                cats  = list(CATEGORIES.items())
                col_a, col_b = st.columns(2)
                for i,(cname,mets) in enumerate(cats):
                    fig_c = build_card_chart(cname, mets)
                    if fig_c:
                        (col_a if i%2==0 else col_b).plotly_chart(fig_c, use_container_width=True)

                # Overall score
                all_pcts = []
                for _,mets in CATEGORIES.items():
                    for _,col,higher in mets:
                        if col and col in pool.columns and col in p.index and not pd.isna(p.get(col)):
                            pct = calc_percentile(p[col], pool[col])
                            all_pcts.append(pct if higher else 100-pct)
                if all_pcts:
                    overall = np.mean(all_pcts)
                    st.markdown(
                        f'<p style="color:{SUBTLE};font-size:12px;margin-top:4px">'
                        f'Overall percentile average: <b style="color:{pct_color(overall)}">{overall:.0f}</b>/100 '
                        f'— vs <b style="color:{TEXT}">{len(pool)}</b> players ({compare_mode.lower()})</p>',
                        unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PLAYER SIMILARITY
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-label">🔍 Player Similarity</p>', unsafe_allow_html=True)
    st.caption("Find players with the most similar statistical profile using cosine similarity.")

    SIM_FEATURES = [c for c in ["npxG90","xA90","goals90","assists90","shots90",
                                 "kp90","npxg_per_shot","npxg_xa90","xGChain90","xGBuildup90"]
                    if c in df_p.columns]

    if not name_col or len(SIM_FEATURES) < 3:
        st.info("Not enough metrics available for similarity.")
    else:
        sim_col1, sim_col2, sim_col3 = st.columns([2,1,1])
        with sim_col1:
            sim_player = st.selectbox("Select player", ["— select —"] + sorted(df_p[name_col].dropna().unique().tolist()), key="sim_player")
        with sim_col2:
            n_similar = st.slider("Similar players", 5, 20, 10, key="sim_n")
        with sim_col3:
            cross_league = st.checkbox("Cross-league search", value=False, key="sim_cross",
                                       help="Search across all scraped leagues/seasons")

        if sim_player and sim_player != "— select —":
            # Build comparison pool
            if cross_league:
                cross_frames = []
                for lg in available_leagues():
                    for ss in available_seasons(lg):
                        df_s = load_league_players(lg, ss)
                        if df_s is not None:
                            df_s = enrich_player_df(df_s)
                            df_s = df_s[df_s["time"].fillna(0) >= min_minutes]
                            df_s["_league"] = lg
                            df_s["_season"] = ss
                            cross_frames.append(df_s)
                sim_pool = pd.concat(cross_frames, ignore_index=True) if cross_frames else df_p
                sim_pool["_league_season"] = sim_pool.get("_league","") + " " + sim_pool.get("_season","")
            else:
                sim_pool = df_p.copy()
                sim_pool["_league_season"] = f"{LEAGUE_LABELS.get(league,league)} {season}"

            feat_df = sim_pool[SIM_FEATURES + ([name_col] if name_col else [])].dropna(subset=SIM_FEATURES)
            if name_col not in feat_df.columns:
                st.info("Name column missing.")
            elif sim_player not in feat_df[name_col].values:
                st.info("Player not found in pool with enough minutes.")
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(feat_df[SIM_FEATURES].values)
                idx_player = feat_df[name_col].tolist().index(sim_player)
                player_vec = X[idx_player].reshape(1,-1)
                sims = cosine_similarity(player_vec, X)[0]
                feat_df = feat_df.copy()
                feat_df["_sim"] = sims
                # Exclude self, get top N
                similar = (feat_df[feat_df[name_col] != sim_player]
                           .nlargest(n_similar, "_sim")
                           .reset_index(drop=True))

                st.markdown(f'<p style="color:{SUBTLE};font-size:12px">Most similar to <b style="color:{TEXT}">{sim_player}</b> — based on {len(SIM_FEATURES)} statistical metrics</p>', unsafe_allow_html=True)

                ncols = 2
                rows_per_col = (n_similar + 1) // 2
                col_l, col_r = st.columns(2)
                for i, row in similar.iterrows():
                    sim_pct  = int(round(row["_sim"] * 100))
                    sim_color= pct_color(sim_pct)
                    p_name_s = row[name_col]
                    p_team_s = row[team_col] if team_col and team_col in row.index else ""
                    league_s = row.get("_league_season","")
                    card_html = f"""
                    <div class="sim-card">
                      <span class="sim-score">{sim_pct}%</span>
                      <span class="sim-card-name">#{i+1} {p_name_s}</span><br>
                      <span class="sim-card-team">{p_team_s}{"  ·  " + league_s if cross_league else ""}</span>
                    </div>"""
                    (col_l if i < rows_per_col else col_r).markdown(card_html, unsafe_allow_html=True)

                # Radar comparison
                if not similar.empty:
                    st.divider()
                    st.markdown('<p class="section-label">Radar Comparison — top 5 similar</p>', unsafe_allow_html=True)
                    radar_players = [sim_player] + similar.head(5)[name_col].tolist()
                    radar_feats   = SIM_FEATURES[:8]  # cap at 8 for readability

                    fig_radar = go.Figure()
                    all_rows = pd.concat([feat_df[feat_df[name_col]==sim_player], similar.head(5)])
                    for _, r in all_rows.iterrows():
                        pname = r[name_col]
                        vals_r= [float(r[f]) if not pd.isna(r.get(f)) else 0 for f in radar_feats]
                        is_selected = (pname == sim_player)
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals_r + [vals_r[0]],
                            theta=radar_feats + [radar_feats[0]],
                            mode="lines",
                            name=pname,
                            line=dict(width=3 if is_selected else 1.2,
                                      color=ACCENT if is_selected else None),
                            opacity=1.0 if is_selected else 0.6,
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor=CARD_BG,
                            radialaxis=dict(visible=True, color=SUBTLE, gridcolor="#2a2d3a"),
                            angularaxis=dict(color=TEXT, gridcolor="#2a2d3a"),
                        ),
                        paper_bgcolor=DARK_BG, font=dict(color=TEXT),
                        height=480, showlegend=True,
                        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color=TEXT)),
                        margin=dict(l=60,r=60,t=40,b=40),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SHOT MAP
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-label">🗺️ Shot Map</p>', unsafe_allow_html=True)

    if not name_col:
        st.info("Name column not found.")
    else:
        sm1, sm2 = st.columns([2,1])
        with sm1:
            shot_player = st.text_input("Search player for shot map", placeholder="e.g. Salah, Haaland…")
        with sm2:
            shot_season_all = st.checkbox("All seasons", value=False, key="shot_all_seasons")

        if shot_player:
            results = df_p[df_p[name_col].str.contains(shot_player, case=False, na=False)]
            if results.empty:
                st.info(f"No player found matching '{shot_player}'.")
            else:
                chosen = results.iloc[0]
                if "id" not in results.columns:
                    st.info("No player ID available for shot map.")
                else:
                    pid = chosen["id"]
                    df_shots = load_player_shots(pid)

                    if df_shots is None or df_shots.empty:
                        st.info("No shot data found for this player.")
                    else:
                        df_shots = df_shots.dropna(subset=["X","Y","xG"])

                        # Season filter
                        if not shot_season_all and "season" in df_shots.columns:
                            df_shots_show = df_shots[df_shots["season"].astype(str) == str(season)]
                            season_label  = f"{season}/{str(int(season)+1)[2:]}"
                        else:
                            df_shots_show = df_shots
                            season_label  = "All seasons"

                        if df_shots_show.empty:
                            st.info(f"No shots recorded for {chosen[name_col]} in {season_label}.")
                        else:
                            # ── Stats row ─────────────────────────────────────
                            total_shots = len(df_shots_show)
                            goals_df    = df_shots_show[df_shots_show["result"]=="Goal"]
                            on_target   = df_shots_show[df_shots_show["result"].isin(["Goal","SavedShot"])]
                            total_xg    = df_shots_show["xG"].sum()
                            avg_xg_shot = df_shots_show["xG"].mean()
                            conv_rate   = len(goals_df)/total_shots*100 if total_shots>0 else 0
                            acc_rate    = len(on_target)/total_shots*100 if total_shots>0 else 0

                            s1,s2,s3,s4,s5 = st.columns(5)
                            for col,lbl,val in [
                                (s1,"Shots",     str(total_shots)),
                                (s2,"Goals",     str(len(goals_df))),
                                (s3,"Total xG",  f"{total_xg:.2f}"),
                                (s4,"Conv %",    f"{conv_rate:.1f}%"),
                                (s5,"Accuracy %",f"{acc_rate:.1f}%"),
                            ]:
                                col.markdown(f'<div class="metric-card"><h4>{lbl}</h4><p>{val}</p></div>',unsafe_allow_html=True)

                            # ── Filters ───────────────────────────────────────
                            fc1, fc2 = st.columns(2)
                            situations = ["All"] + sorted(df_shots_show["situation"].dropna().unique().tolist()) if "situation" in df_shots_show.columns else ["All"]
                            shot_types = ["All"] + sorted(df_shots_show["shotType"].dropna().unique().tolist())  if "shotType"  in df_shots_show.columns else ["All"]
                            with fc1:
                                sel_sit  = st.selectbox("Situation", situations, key="shot_sit")
                            with fc2:
                                sel_type = st.selectbox("Shot type",  shot_types,  key="shot_type")

                            df_filtered = df_shots_show.copy()
                            if sel_sit  != "All" and "situation" in df_filtered.columns:
                                df_filtered = df_filtered[df_filtered["situation"] == sel_sit]
                            if sel_type != "All" and "shotType"  in df_filtered.columns:
                                df_filtered = df_filtered[df_filtered["shotType"]  == sel_type]

                            if df_filtered.empty:
                                st.info("No shots match the selected filters.")
                            else:
                                goals_f  = df_filtered[df_filtered["result"]=="Goal"]
                                shots_f  = df_filtered[df_filtered["result"]!="Goal"]

                                # Build shot map
                                fig_sm = go.Figure()
                                fig_sm.update_layout(shapes=pitch_shapes_half())

                                # ── Non-goal shots ────────────────────────────
                                if not shots_f.empty:
                                    if "situation" in shots_f.columns:
                                        for sit, grp in shots_f.groupby("situation"):
                                            c = SITUATION_COLORS.get(sit, BLUE)
                                            fig_sm.add_trace(go.Scatter(
                                                x=grp["X"], y=grp["Y"],
                                                mode="markers",
                                                name=sit,
                                                marker=dict(
                                                    size=grp["xG"]*30+5,
                                                    color=c, opacity=0.55,
                                                    line=dict(width=1,color="white"),
                                                    symbol=[SHOT_TYPE_MARKERS.get(s,"circle")
                                                            for s in grp.get("shotType", ["circle"]*len(grp))],
                                                ),
                                                hovertemplate=(
                                                    f"<b>{sit}</b><br>"
                                                    "xG: %{customdata[0]:.2f}<br>"
                                                    "Min: %{customdata[1]}<extra>Miss/Save</extra>"
                                                ),
                                                customdata=list(zip(
                                                    grp["xG"],
                                                    grp.get("minute", [""]*len(grp)),
                                                )),
                                            ))
                                    else:
                                        fig_sm.add_trace(go.Scatter(
                                            x=shots_f["X"], y=shots_f["Y"], mode="markers",
                                            name="Shot",
                                            marker=dict(size=shots_f["xG"]*30+5,color=BLUE,opacity=0.55,line=dict(width=1,color="white")),
                                            hovertemplate="xG: %{customdata:.2f}<extra>Miss/Save</extra>",
                                            customdata=shots_f["xG"],
                                        ))

                                # ── Goal shots ────────────────────────────────
                                if not goals_f.empty:
                                    fig_sm.add_trace(go.Scatter(
                                        x=goals_f["X"], y=goals_f["Y"],
                                        mode="markers",
                                        name="⭐ Goal",
                                        marker=dict(
                                            size=goals_f["xG"]*40+10,
                                            color=GOLD, opacity=1.0,
                                            symbol="star",
                                            line=dict(width=1.5,color=DARK_BG),
                                        ),
                                        hovertemplate=(
                                            "⭐ <b>GOAL</b><br>"
                                            "xG: %{customdata[0]:.2f}<br>"
                                            "Min: %{customdata[1]}<extra></extra>"
                                        ),
                                        customdata=list(zip(
                                            goals_f["xG"],
                                            goals_f.get("minute", [""]*len(goals_f)),
                                        )),
                                    ))

                                lay = pitch_layout_half()
                                lay.update(dict(
                                    height=500,
                                    title=dict(
                                        text=f"{chosen[name_col]}  ·  {season_label}  ·  {len(df_filtered)} shots  ·  {len(goals_f)} goals",
                                        font=dict(color=TEXT, size=13), x=0.5,
                                    ),
                                    legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0.5)",
                                                orientation="h", yanchor="bottom", y=1.02, x=0),
                                ))
                                fig_sm.update_layout(lay)
                                st.plotly_chart(fig_sm, use_container_width=True)

                                # Situation breakdown
                                if "situation" in df_filtered.columns:
                                    st.markdown('<p class="section-label">Breakdown by Situation</p>', unsafe_allow_html=True)
                                    sit_stats = (df_filtered.groupby("situation")
                                                 .agg(Shots=("xG","count"), xG=("xG","sum"),
                                                      Goals=("result", lambda x: (x=="Goal").sum()))
                                                 .assign(Conv=lambda d: (d["Goals"]/d["Shots"]*100).round(1),
                                                         AvgxG=lambda d: (d["xG"]/d["Shots"]).round(2))
                                                 .round(2).sort_values("Shots", ascending=False))
                                    st.dataframe(
                                        sit_stats.style.format({"xG":"{:.2f}","AvgxG":"{:.2f}","Conv":"{:.1f}"})
                                                 .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                                        use_container_width=True,
                                    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PRESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab_press:
    df_hist = load_team_history(league, season)
    ppda_df = None

    if df_hist is not None and "ppda_coef" in df_hist.columns:
        tcol = next((c for c in ["team","title"] if c in df_hist.columns), None)
        if tcol:
            ppda_df = (df_hist.groupby(tcol)["ppda_coef"].mean()
                       .reset_index()
                       .rename(columns={tcol:"team","ppda_coef":"avg_ppda"})
                       .sort_values("avg_ppda"))

    if ppda_df is None and "PPDA" in df_main.columns:
        ppda_df = (df_main[["team","PPDA"]].rename(columns={"PPDA":"avg_ppda"})
                   .dropna(subset=["avg_ppda"]).sort_values("avg_ppda"))

    if ppda_df is not None and not ppda_df.empty:
        ppda_df = ppda_df.dropna(subset=["avg_ppda"])
        best, worst = ppda_df.iloc[0], ppda_df.iloc[-1]
        st.markdown('<p class="section-label">PPDA Rankings</p>', unsafe_allow_html=True)
        st.caption("Passes Allowed Per Defensive Action in opponent's half. Lower = more pressing.")
        k1,k2,k3 = st.columns(3)
        k1.markdown(f'<div class="metric-card"><h4>Most Pressing</h4><p>{best["team"]}</p><small>PPDA {fmt(best["avg_ppda"])}</small></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="metric-card"><h4>Least Pressing</h4><p>{worst["team"]}</p><small>PPDA {fmt(worst["avg_ppda"])}</small></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="metric-card"><h4>League Average</h4><p>{fmt(ppda_df["avg_ppda"].mean())}</p><small>avg PPDA</small></div>', unsafe_allow_html=True)
        st.divider()

        med = ppda_df["avg_ppda"].median()
        fig_ppda = go.Figure(go.Bar(
            x=ppda_df["avg_ppda"], y=ppda_df["team"], orientation="h",
            marker_color=[ACCENT if v<=med else RED for v in ppda_df["avg_ppda"]],
            text=[f"{v:.2f}" for v in ppda_df["avg_ppda"]], textposition="outside",
            textfont=dict(size=10,color=TEXT),
            hovertemplate="<b>%{y}</b><br>PPDA: %{x:.2f}<extra></extra>",
        ))
        fig_ppda.add_vline(x=ppda_df["avg_ppda"].mean(),
                           line=dict(color=GOLD,dash="dot",width=1.5),
                           annotation_text="League avg", annotation_font_color=GOLD)
        fig_ppda.update_layout(**plotly_dark(height=540),
            xaxis=dict(title="Avg PPDA",gridcolor="#2a2d3a"),
            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_ppda, use_container_width=True)

        if df_hist is not None and "ppda_coef" in df_hist.columns:
            tcol = next((c for c in ["team","title"] if c in df_hist.columns), None)
            if tcol:
                st.divider()
                st.markdown('<p class="section-label">PPDA Over Time — Rolling Average</p>', unsafe_allow_html=True)
                all_teams = sorted(df_hist[tcol].dropna().unique().tolist())
                selected  = st.multiselect("Teams",all_teams,default=all_teams[:5] if len(all_teams)>=5 else all_teams,key="ppda_teams")
                roll_n    = st.slider("Rolling window",3,10,5,key="ppda_roll")
                date_col  = next((c for c in ["date","datetime"] if c in df_hist.columns),None)
                fig_roll  = go.Figure()
                for team in selected:
                    grp = df_hist[df_hist[tcol]==team].copy()
                    if date_col: grp = grp.sort_values(date_col)
                    grp["roll"] = grp["ppda_coef"].rolling(roll_n,min_periods=1).mean()
                    fig_roll.add_trace(go.Scatter(
                        x=grp[date_col] if date_col else list(range(len(grp))),
                        y=grp["roll"].round(2), mode="lines", name=team,
                        line=dict(width=1.8),
                        hovertemplate=f"<b>{team}</b><br>PPDA: %{{y:.2f}}<extra></extra>",
                    ))
                fig_roll.update_layout(**plotly_dark(height=380),
                    xaxis=dict(title="Match",gridcolor="#2a2d3a"),
                    yaxis=dict(title=f"PPDA (rolling {roll_n})",gridcolor="#2a2d3a"),
                    legend=dict(bgcolor="rgba(0,0,0,0.4)",font=dict(color=TEXT)))
                st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("No PPDA data found for this season.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SEASON TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    if not compare_seasons:
        st.info("Select seasons in the sidebar to compare.")
        st.stop()

    all_frames = []
    for s in compare_seasons:
        df_s = load_league_table(league, s)
        if df_s is not None:
            df_s = df_s.copy()
            df_s["season"] = s
            all_frames.append(df_s)
    if not all_frames:
        st.warning("No data found.")
        st.stop()

    df_multi = pd.concat(all_frames, ignore_index=True)
    all_teams_m = sorted(df_multi["team"].dropna().unique().tolist())
    sel_teams   = st.multiselect("Teams to track", all_teams_m,
                                 default=all_teams_m[:6] if len(all_teams_m)>=6 else all_teams_m,
                                 key="trend_teams")
    metric_opt  = st.selectbox("Metric",
                               [c for c in ["xG","xGA","xGD","xPTS","actual_pts","luck","npxG","PPDA"]
                                if c in df_multi.columns])

    if sel_teams and metric_opt:
        fig_t = go.Figure()
        for team in sel_teams:
            grp = df_multi[df_multi["team"]==team].sort_values("season")
            fig_t.add_trace(go.Scatter(
                x=grp["season"].apply(lambda s: f"{s}/{str(int(s)+1)[2:]}"),
                y=grp[metric_opt].round(2), mode="lines+markers", name=team,
                line=dict(width=2), marker=dict(size=7),
                hovertemplate=f"<b>{team}</b><br>{metric_opt}: %{{y:.2f}}<extra></extra>",
            ))
        fig_t.update_layout(**plotly_dark(height=460),
            title=dict(text=f"{metric_opt} — {LEAGUE_LABELS.get(league,league)}",font=dict(color=TEXT)),
            xaxis=dict(title="Season",gridcolor="#2a2d3a"),
            yaxis=dict(title=metric_opt,gridcolor="#2a2d3a"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)",font=dict(color=TEXT)))
        st.plotly_chart(fig_t, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-label">Average across selected seasons</p>', unsafe_allow_html=True)
    agg_cols = [c for c in ["xG","xGA","xGD","xPTS","actual_pts","luck","PPDA"] if c in df_multi.columns]
    df_avg = df_multi.groupby("team")[agg_cols].mean().round(2).sort_values("xGD",ascending=False)
    df_avg.insert(0,"Seasons",df_multi.groupby("team")["season"].count())
    st.dataframe(
        df_avg.style.format({c:"{:.2f}" for c in agg_cols})
              .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
        use_container_width=True, height=420,
    )

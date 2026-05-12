"""
Football Scouting & Analytics
================================
General football analytics powered by Understat data.
No FPL dependency — works for any football fan or scout.
"""

import ast
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("understat_data")

AVAILABLE_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
LEAGUE_LABELS = {
    "EPL":        "Premier League",
    "La_liga":    "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "Serie A",
    "Ligue_1":    "Ligue 1",
    "RFPL":       "Russian Premier League",
}

DARK_BG = "#0e1117"
CARD_BG = "#1a1d27"
ACCENT  = "#5fffb0"
TEXT    = "#f0f0f0"
SUBTLE  = "#8899aa"
RED     = "#e63946"
BLUE    = "#457b9d"
GOLD    = "#f4d03f"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Football Scout | CoachFPL", page_icon="⚽", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; }}
  .metric-card {{
    background: {CARD_BG}; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 8px; border-left: 3px solid {ACCENT};
  }}
  .metric-card h4    {{ color:{ACCENT}; margin:0 0 4px; font-size:11px; letter-spacing:1px; text-transform:uppercase; }}
  .metric-card p     {{ color:{TEXT};   margin:0; font-size:22px; font-weight:700; }}
  .metric-card small {{ color:{SUBTLE}; font-size:11px; }}
  .section-label {{ font-size:11px; font-weight:800; color:{ACCENT}; letter-spacing:1px; margin-bottom:4px; }}
  .scout-header {{
    background: linear-gradient(135deg, #1a1d27 0%, #0d1b2a 100%);
    border: 1px solid #2a2d3a; border-radius:12px; padding:20px 24px; margin-bottom:16px;
  }}
  .scout-header h2 {{ color:{TEXT}; margin:0 0 6px; font-size:26px; letter-spacing:1px; }}
  .scout-tag {{
    display:inline-block; background:#2a2d3a; color:{ACCENT};
    border-radius:4px; padding:3px 8px; font-size:11px; font-weight:700;
    margin-right:6px; letter-spacing:0.5px;
  }}
  .scout-stat {{
    display:inline-block; color:{SUBTLE}; font-size:12px; margin-right:16px;
  }}
  .scout-stat b {{ color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_league_table(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_table.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.rename(columns={
        "Team": "team", "W": "wins", "D": "draws", "L": "loses",
        "PTS": "actual_pts", "NPxG": "npxG", "NPxGA": "npxGA",
        "NPxGD": "npxGD", "G": "scored", "GA": "missed",
    })
    for c in ["xG","xGA","npxG","npxGA","npxGD","xPTS","actual_pts",
              "wins","draws","loses","PPDA","OPPDA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["xGD"]  = (df["xG"]  - df["xGA"]).round(2)
    df["luck"]  = (df["actual_pts"] - df["xPTS"]).round(2)
    return df

@st.cache_data(ttl=600)
def load_league_players(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_players.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["xG","xA","npxG","npxA","goals","assists","time","shots","key_passes","npg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=600)
def load_team_history(league: str, season: str) -> pd.DataFrame | None:
    p = DATA_DIR / league / season / "league_team_history.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "ppda" in df.columns:
        def parse_ppda(val):
            try:
                d = ast.literal_eval(str(val))
                att, defn = float(d.get("att", 0)), float(d.get("def", 0))
                return round(att / defn, 2) if defn > 0 else None
            except Exception:
                return None
        df["ppda_coef"] = df["ppda"].apply(parse_ppda)
    return df

@st.cache_data(ttl=600)
def load_player_shots(player_id) -> pd.DataFrame | None:
    p = DATA_DIR / "players" / str(player_id) / "player_shots.csv"
    return pd.read_csv(p) if p.exists() else None

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
        if v > threshold: return f"color: {ACCENT}"
        if v < threshold: return f"color: {RED}"
    except Exception:
        pass
    return ""

def calc_percentile(value, series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return 50.0
    return float((s < value).sum() / len(s) * 100)

def pct_color(p: float) -> str:
    if p >= 80: return ACCENT
    if p >= 60: return "#2ecc71"
    if p >= 40: return GOLD
    if p >= 20: return "#e67e22"
    return RED

def enrich_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90 and derived columns to a player dataframe."""
    df = df.copy()
    mins = df["time"].replace(0, np.nan)
    df["npxG90"]       = (df["npxG"]        / (mins / 90)).round(2)
    df["xA90"]         = (df["xA"]           / (mins / 90)).round(2)
    df["goals90"]      = (df["goals"]         / (mins / 90)).round(2)
    df["assists90"]    = (df["assists"]       / (mins / 90)).round(2)
    df["goal_diff"]    = (df["goals"] - df["xG"]).round(2)
    df["assist_diff"]  = (df["assists"] - df["xA"]).round(2)
    if "shots" in df.columns:
        df["shots90"]      = (df["shots"]     / (mins / 90)).round(2)
        df["npxg_per_shot"]= (df["npxG"] / df["shots"].replace(0, np.nan)).round(2)
    if "key_passes" in df.columns:
        df["kp90"]         = (df["key_passes"] / (mins / 90)).round(2)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚽ Football Scout</p>', unsafe_allow_html=True)
    st.markdown("General football analytics. No FPL required.")
    st.divider()

    leagues = available_leagues()
    if not leagues:
        st.error("No data found. Copy `understat_data/` to your app folder.")
        st.stop()

    league = st.selectbox("League", leagues, format_func=lambda x: LEAGUE_LABELS.get(x, x))
    seasons = available_seasons(league)
    if not seasons:
        st.error(f"No seasons found for {league}.")
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

c1, c2 = st.columns([3, 1])
with c1:
    st.title("⚽ Football Scout")
    st.caption(f"{LEAGUE_LABELS.get(league, league)}  ·  {season}/{str(int(season)+1)[2:]}")
with c2:
    if df_main is not None:
        st.metric("Teams", len(df_main))

if df_main is None:
    st.error("No data for this season. Make sure `understat_data/` is in the app root.")
    st.stop()

tab_teams, tab_players, tab_press, tab_trends = st.tabs([
    "🏟️ Team xG", "👤 Player Scouting", "⚡ Pressing", "📈 Season Trends",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TEAM xG
# ══════════════════════════════════════════════════════════════════════════════
with tab_teams:
    df = df_main.copy().sort_values("xGD", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub in [
        (k1, "Best Attack (xG)",   df.nlargest(1,"xG").iloc[0]["team"],     f'{fmt(df.nlargest(1,"xG").iloc[0]["xG"])} xG'),
        (k2, "Best Defence (xGA)", df.nsmallest(1,"xGA").iloc[0]["team"],   f'{fmt(df.nsmallest(1,"xGA").iloc[0]["xGA"])} xGA'),
        (k3, "Luckiest Team",      df.nlargest(1,"luck").iloc[0]["team"],   f'+{fmt(df.nlargest(1,"luck").iloc[0]["luck"])} pts'),
        (k4, "Unluckiest Team",    df.nsmallest(1,"luck").iloc[0]["team"],  f'{fmt(df.nsmallest(1,"luck").iloc[0]["luck"])} pts'),
    ]:
        col.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{val}</p><small>{sub}</small></div>',
                     unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<p class="section-label">xG Difference — ranked</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=df["xGD"], y=df["team"], orientation="h",
            marker_color=[ACCENT if v >= 0 else RED for v in df["xGD"]],
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
            marker_color=[RED if v < 0 else ACCENT for v in dl["luck"]],
            text=[f"{v:.2f}" for v in dl["luck"]], textposition="outside",
            textfont=dict(size=10, color=TEXT),
        ))
        fig2.update_layout(**plotly_dark(height=520),
            xaxis=dict(title="Pts − xPTS", gridcolor="#2a2d3a", zeroline=True, zerolinecolor=SUBTLE),
            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-label">Full xG Table</p>', unsafe_allow_html=True)
    show_cols = [c for c in ["team","wins","draws","loses","xG","xGA","xGD",
                              "npxG","npxGA","xPTS","actual_pts","luck"] if c in df.columns]
    df_show = df[show_cols].round(2).reset_index(drop=True)
    df_show.insert(0, "#", range(1, len(df_show)+1))
    num_cols = [c for c in df_show.columns if c not in ["#", "team"]]
    styled = (
        df_show.style
        .map(lambda v: colour_val(v),             subset=["xGD"]  if "xGD"  in df_show.columns else [])
        .map(lambda v: colour_val(v, threshold=3), subset=["luck"] if "luck" in df_show.columns else [])
        .format({c: "{:.2f}" for c in num_cols})
        .set_properties(**{"background-color": CARD_BG, "color": TEXT, "border-color": "#2a2d3a"})
    )
    st.dataframe(styled, use_container_width=True, height=420)

    st.divider()
    st.markdown('<p class="section-label">xG vs xGA — Team Quadrant</p>', unsafe_allow_html=True)
    mx, my = df["xG"].median(), df["xGA"].median()
    fig3 = go.Figure()
    fig3.add_shape(type="line", x0=mx, x1=mx, y0=df["xGA"].min()-2, y1=df["xGA"].max()+2,
                   line=dict(color=SUBTLE, dash="dot", width=1))
    fig3.add_shape(type="line", x0=df["xG"].min()-2, x1=df["xG"].max()+2, y0=my, y1=my,
                   line=dict(color=SUBTLE, dash="dot", width=1))
    fig3.add_trace(go.Scatter(
        x=df["xG"], y=df["xGA"], mode="markers+text",
        text=df["team"], textposition="top center", textfont=dict(size=9, color=TEXT),
        marker=dict(size=14, color=df["xGD"],
                    colorscale=[[0,RED],[0.5,SUBTLE],[1,ACCENT]],
                    showscale=True, colorbar=dict(title="xGD", tickfont=dict(color=TEXT)),
                    line=dict(width=1, color=DARK_BG)),
        hovertemplate="<b>%{text}</b><br>xG: %{x:.2f}<br>xGA: %{y:.2f}<extra></extra>",
    ))
    for txt, x, y in [
        ("Good attack\nGood defence", df["xG"].max()-1, df["xGA"].min()+1),
        ("Good attack\nBad defence",  df["xG"].max()-1, df["xGA"].max()-1),
        ("Bad attack\nGood defence",  df["xG"].min()+1, df["xGA"].min()+1),
        ("Bad attack\nBad defence",   df["xG"].min()+1, df["xGA"].max()-1),
    ]:
        fig3.add_annotation(x=x, y=y, text=txt, showarrow=False,
                            font=dict(size=9, color=SUBTLE), align="center")
    fig3.update_layout(**plotly_dark(height=480),
        xaxis=dict(title="xG (attack)", gridcolor="#2a2d3a"),
        yaxis=dict(title="xGA (defence — lower is better)", gridcolor="#2a2d3a"))
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

    name_col = next((c for c in ["player_name","name"]     if c in df_p.columns), None)
    team_col = next((c for c in ["team_title","team"]      if c in df_p.columns), None)
    pos_col  = next((c for c in ["position","pos"]         if c in df_p.columns), None)

    # Position filter
    if pos_col:
        all_pos   = sorted(df_p[pos_col].dropna().unique().tolist())
        positions = st.multiselect("Filter by position", all_pos, default=all_pos, key="pos_filter")
        if positions:
            df_p = df_p[df_p[pos_col].isin(positions)]

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">npxG / 90  vs  xA / 90</p>', unsafe_allow_html=True)
    if name_col and not df_p.empty:
        custom = list(zip(
            df_p[name_col],
            df_p[team_col] if team_col else [""]*len(df_p),
        ))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_p["npxG90"], y=df_p["xA90"], mode="markers",
            marker=dict(
                size=df_p["time"]/35,
                color=df_p["npxG90"],
                colorscale=[[0,"#1a1d27"],[0.5,BLUE],[1,ACCENT]],
                showscale=True, colorbar=dict(title="npxG/90", tickfont=dict(color=TEXT)),
                opacity=0.8, line=dict(width=0.5, color=DARK_BG),
            ),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>npxG/90: %{x:.2f}<br>xA/90: %{y:.2f}<extra></extra>",
        ))
        for _, row in df_p.nlargest(12, "npxG90").iterrows():
            fig_sc.add_annotation(x=row["npxG90"], y=row["xA90"],
                text=row[name_col].split()[-1], showarrow=False,
                font=dict(size=8, color=TEXT), yshift=10)
        fig_sc.add_hline(y=df_p["xA90"].median(),   line=dict(color=SUBTLE, dash="dot", width=1))
        fig_sc.add_vline(x=df_p["npxG90"].median(), line=dict(color=SUBTLE, dash="dot", width=1))
        fig_sc.update_layout(**plotly_dark(height=500),
            xaxis=dict(title="npxG per 90", gridcolor="#2a2d3a"),
            yaxis=dict(title="xA per 90",   gridcolor="#2a2d3a"))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Top tables ────────────────────────────────────────────────────────────
    st.divider()
    cl, cr = st.columns(2)
    if name_col and team_col:
        with cl:
            st.markdown('<p class="section-label">Top 15 — npxG / 90</p>', unsafe_allow_html=True)
            t = df_p.nlargest(15,"npxG90")[[name_col,team_col,"npxG90","xA90","time"]].round(2)
            t.columns = ["Player","Team","npxG/90","xA/90","Mins"]
            t.insert(0,"#",range(1,len(t)+1))
            st.dataframe(t.style.format({"npxG/90":"{:.2f}","xA/90":"{:.2f}","Mins":"{:.0f}"})
                          .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                         use_container_width=True, height=420, hide_index=True)
        with cr:
            st.markdown('<p class="section-label">Top 15 — xA / 90 (Creators)</p>', unsafe_allow_html=True)
            t2 = df_p.nlargest(15,"xA90")[[name_col,team_col,"xA90","npxG90","time"]].round(2)
            t2.columns = ["Player","Team","xA/90","npxG/90","Mins"]
            t2.insert(0,"#",range(1,len(t2)+1))
            st.dataframe(t2.style.format({"xA/90":"{:.2f}","npxG/90":"{:.2f}","Mins":"{:.0f}"})
                          .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                         use_container_width=True, height=420, hide_index=True)

    # ── Over/underperformance ─────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Over / Underperformance — Goals vs xG</p>', unsafe_allow_html=True)
    df_perf  = df_p.dropna(subset=["goal_diff"]).copy()
    df_chart = pd.concat([df_perf.nlargest(10,"goal_diff"),
                          df_perf.nsmallest(10,"goal_diff")]).drop_duplicates()
    df_chart = df_chart.sort_values("goal_diff")
    fig_perf = go.Figure(go.Bar(
        x=df_chart["goal_diff"],
        y=df_chart[name_col] if name_col else df_chart.index,
        orientation="h",
        marker_color=[ACCENT if v >= 0 else RED for v in df_chart["goal_diff"]],
        text=[f"{v:.2f}" for v in df_chart["goal_diff"]], textposition="outside",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{y}</b><br>Goals − xG: %{x:.2f}<extra></extra>",
    ))
    fig_perf.update_layout(**plotly_dark(height=460),
        xaxis=dict(title="Goals − xG  (positive = overperformer)",
                   gridcolor="#2a2d3a", zeroline=True, zerolinecolor=SUBTLE),
        yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig_perf, use_container_width=True)
    st.caption("🟢 Overperformers may regress  ·  🔴 Underperformers may improve")

    # ══════════════════════════════════════════════════════════════════════════
    #  PLAYER SCOUTING CARD
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<p class="section-label">🎴 Player Scouting Card</p>', unsafe_allow_html=True)
    st.caption("Percentile bars compare the player vs all players in this league/season with enough minutes.")

    if not name_col:
        st.info("Player name column not found in data.")
    else:
        all_player_names = sorted(df_p[name_col].dropna().unique().tolist())
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_player = st.selectbox(
                "Select player",
                ["— select a player —"] + all_player_names,
                key="scout_player",
            )
        with col_sel2:
            compare_mode = st.radio(
                "Compare vs",
                ["All outfield", "Same position"],
                horizontal=True,
                key="scout_compare",
            )

        if selected_player and selected_player != "— select a player —":
            player_rows = df_p[df_p[name_col] == selected_player]
            if player_rows.empty:
                st.info("Player not found with enough minutes in this season.")
            else:
                p = player_rows.iloc[0]

                # Build comparison pool
                if compare_mode == "Same position" and pos_col and pd.notna(p.get(pos_col, None)):
                    pool = df_p[df_p[pos_col] == p[pos_col]]
                else:
                    pool = df_p

                # ── Player header ─────────────────────────────────────────────
                p_name   = p[name_col]
                p_team   = p[team_col] if team_col and pd.notna(p.get(team_col)) else ""
                p_pos    = p[pos_col]  if pos_col  and pd.notna(p.get(pos_col))  else ""
                p_mins   = int(p["time"]) if pd.notna(p["time"]) else 0
                p_games  = int(p["games"]) if "games" in p.index and pd.notna(p.get("games")) else "—"
                p_goals  = int(p["goals"]) if pd.notna(p.get("goals", np.nan)) else "—"
                p_assists= int(p["assists"]) if pd.notna(p.get("assists", np.nan)) else "—"
                season_label = f"{season}/{str(int(season)+1)[2:]}"

                st.markdown(f"""
                <div class="scout-header">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                      <span class="scout-tag">{p_pos}</span>
                      <span class="scout-tag" style="color:{SUBTLE}">{season_label}</span>
                      <h2>{p_name}</h2>
                      <span class="scout-stat">🏟️ <b>{p_team}</b></span>
                      <span class="scout-stat">⏱ <b>{p_mins:,}</b> mins</span>
                      <span class="scout-stat">🎮 <b>{p_games}</b> games</span>
                      <span class="scout-stat">⚽ <b>{p_goals}</b> goals</span>
                      <span class="scout-stat">🎯 <b>{p_assists}</b> assists</span>
                    </div>
                    <div style="color:{SUBTLE}; font-size:11px; text-align:right;">
                      Percentile vs<br><b style="color:{TEXT}">{compare_mode}</b><br>
                      {LEAGUE_LABELS.get(league, league)}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metric definitions ────────────────────────────────────────
                # Each entry: (display_name, column, higher_is_better)
                CATEGORIES = {
                    "⚡ SHOOTING": [
                        ("npxG / 90",        "npxG90",        True),
                        ("Goals / 90",       "goals90",       True),
                        ("Shots / 90",       "shots90",       True),
                        ("npxG per Shot",    "npxg_per_shot", True),
                    ],
                    "🎯 FINISHING": [
                        ("Goals vs xG",      "goal_diff",     True),
                        ("Assists vs xA",    "assist_diff",   True),
                    ],
                    "🔑 CREATING": [
                        ("xA / 90",          "xA90",          True),
                        ("Assists / 90",     "assists90",     True),
                        ("Key Passes / 90",  "kp90",          True),
                    ],
                    "📊 VOLUME": [
                        ("npxG + xA / 90",   None,            True),  # computed below
                        ("Games Played",     "games",         True),
                    ],
                }

                # Compute npxG+xA/90 on-the-fly
                pool     = pool.copy()
                df_p_vol = df_p.copy()
                for _df in [pool, df_p_vol]:
                    _df["npxg_xa90"] = (_df["npxG90"] + _df["xA90"]).round(2)
                p_vol = p["npxG90"] + p["xA90"]

                # Patch the "None" column
                CATEGORIES["📊 VOLUME"][0] = ("npxG + xA / 90", "npxg_xa90", True)
                pool["npxg_xa90"] = pool["npxG90"] + pool["xA90"]

                def build_chart(category_name, metric_defs):
                    rows = []
                    for label, col, higher in metric_defs:
                        if col not in pool.columns or col not in p.index:
                            continue
                        val = p[col]
                        if pd.isna(val):
                            continue
                        series = pool[col]
                        raw_pct = calc_percentile(val, series)
                        pct = raw_pct if higher else (100 - raw_pct)
                        rows.append((label, val, pct))

                    if not rows:
                        return None

                    n      = len(rows)
                    names  = [r[0] for r in rows]
                    vals   = [r[1] for r in rows]
                    pcts   = [r[2] for r in rows]
                    height = max(120, n * 48 + 70)

                    fig = go.Figure()

                    # Background track
                    fig.add_trace(go.Bar(
                        y=names, x=[100]*n, orientation="h",
                        marker_color="rgba(255,255,255,0.06)",
                        showlegend=False, hoverinfo="skip",
                        width=0.55,
                    ))
                    # Percentile bars
                    fig.add_trace(go.Bar(
                        y=names, x=pcts, orientation="h",
                        marker_color=[pct_color(p_) for p_ in pcts],
                        showlegend=False,
                        width=0.55,
                        hovertemplate=[
                            f"<b>{nm}</b><br>Value: {v:.2f}<br>Percentile: {int(round(pc))}<extra></extra>"
                            for nm, v, pc in zip(names, vals, pcts)
                        ],
                    ))

                    # Value labels on left, percentile on right
                    for nm, v, pc in zip(names, vals, pcts):
                        fig.add_annotation(y=nm, x=-1, text=f"{v:.2f}",
                                           xanchor="right", showarrow=False,
                                           font=dict(color=TEXT, size=10))
                        fig.add_annotation(y=nm, x=101, text=str(int(round(pc))),
                                           xanchor="left", showarrow=False,
                                           font=dict(color=pct_color(pc), size=11))

                    fig.update_layout(
                        barmode="overlay",
                        title=dict(text=category_name, font=dict(color=ACCENT, size=12), x=0),
                        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                        font=dict(color=TEXT),
                        height=height,
                        margin=dict(l=160, r=55, t=40, b=10),
                        xaxis=dict(range=[-20, 115], visible=False),
                        yaxis=dict(tickfont=dict(size=10), tickcolor=TEXT, autorange="reversed"),
                    )
                    return fig

                # ── Render card in 2-column grid ──────────────────────────────
                cat_items = list(CATEGORIES.items())
                col_a, col_b = st.columns(2)
                for i, (cat_name, metrics) in enumerate(cat_items):
                    target_col = col_a if i % 2 == 0 else col_b
                    fig_cat = build_chart(cat_name, metrics)
                    if fig_cat:
                        target_col.plotly_chart(fig_cat, use_container_width=True)

                # ── Quick summary line ────────────────────────────────────────
                summary_metrics = []
                for cat, mets in CATEGORIES.items():
                    for label, col, higher in mets:
                        if col and col in pool.columns and col in p.index and not pd.isna(p.get(col)):
                            v   = p[col]
                            pct = calc_percentile(v, pool[col])
                            pct = pct if higher else (100 - pct)
                            summary_metrics.append(pct)
                if summary_metrics:
                    overall = np.mean(summary_metrics)
                    color   = pct_color(overall)
                    st.markdown(
                        f'<p style="color:{SUBTLE}; font-size:12px; margin-top:4px;">'
                        f'Overall percentile average: <b style="color:{color}">{overall:.0f}</b> / 100 '
                        f'— compared vs <b style="color:{TEXT}">{len(pool)}</b> players '
                        f'({compare_mode.lower()})</p>',
                        unsafe_allow_html=True,
                    )

    # ── Player search + shot map ──────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Shot Map</p>', unsafe_allow_html=True)
    search = st.text_input("Search player for shot map", placeholder="e.g. Salah, Haaland…")
    if search and name_col:
        results = df_p[df_p[name_col].str.contains(search, case=False, na=False)]
        if not results.empty:
            show = [c for c in [name_col, team_col, "time","goals","xG","goal_diff",
                                 "assists","xA","assist_diff","npxG90","xA90"] if c in results.columns]
            st.dataframe(
                results[show].round(2).style
                .format({c:"{:.2f}" for c in results[show].round(2).select_dtypes("number").columns})
                .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
                use_container_width=True, hide_index=True,
            )
            if "id" in results.columns:
                pid = results.iloc[0]["id"]
                df_shots = load_player_shots(pid)
                if df_shots is not None and not df_shots.empty:
                    for c in ["X","Y","xG"]:
                        df_shots[c] = pd.to_numeric(df_shots[c], errors="coerce")
                    df_shots = df_shots.dropna(subset=["X","Y","xG"])
                    if "season" in df_shots.columns:
                        df_shots = df_shots[df_shots["season"].astype(str) == str(season)]

                    goals  = df_shots[df_shots["result"] == "Goal"]
                    misses = df_shots[df_shots["result"] != "Goal"]

                    st.markdown(f'<p class="section-label">Shot Map — {results.iloc[0][name_col]} ({season}/{str(int(season)+1)[2:]})</p>',
                                unsafe_allow_html=True)
                    fig_s = go.Figure()
                    fig_s.update_layout(shapes=[
                        dict(type="rect",   x0=0,    y0=0,    x1=1,    y1=1,    line=dict(color="white",width=2),   fillcolor="rgba(45,106,79,0.85)"),
                        dict(type="rect",   x0=0,    y0=0.21, x1=0.17, y1=0.79, line=dict(color="white",width=1.5), fillcolor="rgba(0,0,0,0)"),
                        dict(type="rect",   x0=0.83, y0=0.21, x1=1,    y1=0.79, line=dict(color="white",width=1.5), fillcolor="rgba(0,0,0,0)"),
                        dict(type="rect",   x0=0,    y0=0.36, x1=0.06, y1=0.64, line=dict(color="white",width=1),   fillcolor="rgba(0,0,0,0)"),
                        dict(type="rect",   x0=0.94, y0=0.36, x1=1,    y1=0.64, line=dict(color="white",width=1),   fillcolor="rgba(0,0,0,0)"),
                        dict(type="line",   x0=0.5,  y0=0,    x1=0.5,  y1=1,    line=dict(color="white",width=1.5)),
                        dict(type="circle", x0=0.4,  y0=0.4,  x1=0.6,  y1=0.6,  line=dict(color="white",width=1.5)),
                    ])
                    if not misses.empty:
                        fig_s.add_trace(go.Scatter(
                            x=misses["X"], y=misses["Y"], mode="markers",
                            marker=dict(size=misses["xG"]*25+4, color=BLUE, opacity=0.5,
                                        line=dict(width=0.5, color="white")),
                            name="Shot",
                            hovertemplate="xG: %{customdata:.2f}<extra>Miss/Save</extra>",
                            customdata=misses["xG"],
                        ))
                    if not goals.empty:
                        fig_s.add_trace(go.Scatter(
                            x=goals["X"], y=goals["Y"], mode="markers",
                            marker=dict(size=goals["xG"]*30+8, color=GOLD, opacity=0.95,
                                        symbol="star", line=dict(width=1, color="white")),
                            name="Goal ⭐",
                            hovertemplate="xG: %{customdata:.2f}<extra>Goal</extra>",
                            customdata=goals["xG"],
                        ))
                    fig_s.update_layout(
                        paper_bgcolor="#1d3557", plot_bgcolor="#2d6a4f",
                        font=dict(color=TEXT), margin=dict(l=10,r=10,t=30,b=10),
                        xaxis=dict(range=[0,1], visible=False),
                        yaxis=dict(range=[0,1], visible=False, scaleanchor="x"),
                        legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0.4)"),
                        height=420,
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.caption(f"⭐ Goals: {len(goals)}  ·  🔵 Shots: {len(misses)}")
        else:
            st.info(f"No players found matching '{search}'.")


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
        st.markdown('<p class="section-label">PPDA Rankings — lower = more aggressive pressing</p>', unsafe_allow_html=True)
        st.caption("PPDA = Passes Allowed Per Defensive Action in opponent's half.")
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="metric-card"><h4>Most Pressing</h4><p>{best["team"]}</p><small>PPDA {fmt(best["avg_ppda"])}</small></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="metric-card"><h4>Least Pressing</h4><p>{worst["team"]}</p><small>PPDA {fmt(worst["avg_ppda"])}</small></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="metric-card"><h4>League Average</h4><p>{fmt(ppda_df["avg_ppda"].mean())}</p><small>avg PPDA</small></div>', unsafe_allow_html=True)
        st.divider()

        med = ppda_df["avg_ppda"].median()
        fig_ppda = go.Figure(go.Bar(
            x=ppda_df["avg_ppda"], y=ppda_df["team"], orientation="h",
            marker_color=[ACCENT if v <= med else RED for v in ppda_df["avg_ppda"]],
            text=[f"{v:.2f}" for v in ppda_df["avg_ppda"]], textposition="outside",
            textfont=dict(size=10, color=TEXT),
            hovertemplate="<b>%{y}</b><br>PPDA: %{x:.2f}<extra></extra>",
        ))
        fig_ppda.add_vline(x=ppda_df["avg_ppda"].mean(),
                           line=dict(color=GOLD, dash="dot", width=1.5),
                           annotation_text="League avg", annotation_font_color=GOLD)
        fig_ppda.update_layout(**plotly_dark(height=540),
            xaxis=dict(title="Avg PPDA (lower = more pressing)", gridcolor="#2a2d3a"),
            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_ppda, use_container_width=True)

        if df_hist is not None and "ppda_coef" in df_hist.columns:
            tcol = next((c for c in ["team","title"] if c in df_hist.columns), None)
            if tcol:
                st.divider()
                st.markdown('<p class="section-label">PPDA Over Time — Rolling Average</p>', unsafe_allow_html=True)
                all_teams = sorted(df_hist[tcol].dropna().unique().tolist())
                selected  = st.multiselect("Teams", all_teams,
                                           default=all_teams[:5] if len(all_teams)>=5 else all_teams,
                                           key="ppda_teams")
                roll_n   = st.slider("Rolling window (matches)", 3, 10, 5, key="ppda_roll")
                date_col = next((c for c in ["date","datetime"] if c in df_hist.columns), None)
                fig_roll = go.Figure()
                for team in selected:
                    grp = df_hist[df_hist[tcol]==team].copy()
                    if date_col:
                        grp = grp.sort_values(date_col)
                    grp["roll"] = grp["ppda_coef"].rolling(roll_n, min_periods=1).mean()
                    fig_roll.add_trace(go.Scatter(
                        x=grp[date_col] if date_col else list(range(len(grp))),
                        y=grp["roll"].round(2), mode="lines", name=team,
                        line=dict(width=1.8),
                        hovertemplate=f"<b>{team}</b><br>PPDA: %{{y:.2f}}<extra></extra>",
                    ))
                fig_roll.update_layout(**plotly_dark(height=380),
                    xaxis=dict(title="Match", gridcolor="#2a2d3a"),
                    yaxis=dict(title=f"PPDA (rolling {roll_n})", gridcolor="#2a2d3a"),
                    legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color=TEXT)))
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
        st.warning("No data found for selected seasons.")
        st.stop()

    df_multi = pd.concat(all_frames, ignore_index=True)
    all_teams_multi = sorted(df_multi["team"].dropna().unique().tolist())

    sel_teams  = st.multiselect("Teams to track", all_teams_multi,
                                default=all_teams_multi[:6] if len(all_teams_multi)>=6 else all_teams_multi,
                                key="trend_teams")
    metric_opt = st.selectbox("Metric",
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
            title=dict(text=f"{metric_opt} over seasons — {LEAGUE_LABELS.get(league,league)}",
                       font=dict(color=TEXT)),
            xaxis=dict(title="Season", gridcolor="#2a2d3a"),
            yaxis=dict(title=metric_opt, gridcolor="#2a2d3a"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color=TEXT)))
        st.plotly_chart(fig_t, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-label">Average across selected seasons</p>', unsafe_allow_html=True)
    agg_cols = [c for c in ["xG","xGA","xGD","xPTS","actual_pts","luck","PPDA"] if c in df_multi.columns]
    df_avg = df_multi.groupby("team")[agg_cols].mean().round(2).sort_values("xGD", ascending=False)
    df_avg.insert(0, "Seasons", df_multi.groupby("team")["season"].count())
    st.dataframe(
        df_avg.style
        .format({c:"{:.2f}" for c in agg_cols})
        .set_properties(**{"background-color":CARD_BG,"color":TEXT}),
        use_container_width=True, height=420,
    )

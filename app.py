import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import math
import requests
import plotly.graph_objects as go

# ── FPL API ───────────────────────────────────────────────────────────────────
BASE_URL      = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL  = f"{BASE_URL}/fixtures/"

@st.cache_data(ttl=1800, show_spinner="📡 Fetching live FPL data...")
def _fetch_bootstrap():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(BOOTSTRAP_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner="📡 Fetching fixtures...")
def _fetch_live_fixtures():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(FIXTURES_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def get_current_gw(bootstrap):
    for ev in bootstrap["events"]:
        if ev["is_current"]:
            return ev["id"]
    for ev in bootstrap["events"]:
        if ev["is_next"]:
            return ev["id"]
    return 29

def player_photo_url(code):
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"

# ── Club brand colours (primary bg, text) ────────────────────────────────────
CLUB_COLORS = {
    "Arsenal":        {"bg": "#EF0107", "text": "#FFFFFF", "accent": "#063672"},
    "Aston Villa":    {"bg": "#670E36", "text": "#FFFFFF", "accent": "#95BFE5"},
    "Bournemouth":    {"bg": "#DA291C", "text": "#FFFFFF", "accent": "#000000"},
    "Brentford":      {"bg": "#E30613", "text": "#FFFFFF", "accent": "#FBB800"},
    "Brighton":       {"bg": "#0057B8", "text": "#FFFFFF", "accent": "#FFFFFF"},
    "Burnley":        {"bg": "#6C1D45", "text": "#FFFFFF", "accent": "#99D6EA"},
    "Chelsea":        {"bg": "#034694", "text": "#FFFFFF", "accent": "#DBA111"},
    "Crystal Palace": {"bg": "#1B458F", "text": "#FFFFFF", "accent": "#C4122E"},
    "Everton":        {"bg": "#003399", "text": "#FFFFFF", "accent": "#FFFFFF"},
    "Fulham":         {"bg": "#CC0000", "text": "#FFFFFF", "accent": "#000000"},
    "Leeds":          {"bg": "#FFCD00", "text": "#1D428A", "accent": "#FFFFFF"},
    "Liverpool":      {"bg": "#C8102E", "text": "#FFFFFF", "accent": "#00B2A9"},
    "Man City":       {"bg": "#6CABDD", "text": "#1C2C5B", "accent": "#FFFFFF"},
    "Man Utd":        {"bg": "#DA291C", "text": "#FFFFFF", "accent": "#FBE122"},
    "Newcastle":      {"bg": "#241F20", "text": "#FFFFFF", "accent": "#41B6E6"},
    "Nottm Forest":   {"bg": "#DD0000", "text": "#FFFFFF", "accent": "#FFFFFF"},
    "Sunderland":     {"bg": "#EB172B", "text": "#FFFFFF", "accent": "#000000"},
    "Spurs":          {"bg": "#132257", "text": "#FFFFFF", "accent": "#FFFFFF"},
    "West Ham":       {"bg": "#7A263A", "text": "#FFFFFF", "accent": "#1BB1E7"},
    "Wolves":         {"bg": "#FDB913", "text": "#231F20", "accent": "#231F20"},
}

def club_style(team_name):
    """Return (bg, text) for a club. Falls back to neutral."""
    c = CLUB_COLORS.get(team_name, {"bg": "#2a2a2a", "text": "#ffffff"})
    return c["bg"], c["text"]

# ── Configuration ─────────────────────────────────────────────────────────────
RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"
PROJECTIONS_CSV_FILE   = "projections.csv"

AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25
BGW_PENALTY_FDR = 6.0
FDR_THRESHOLDS  = {5: 120.0, 4: 108.0, 3: 99.0, 2: 90.0, 1: 0}

PREMIER_LEAGUE_TEAMS = sorted([
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley",
    "Chelsea","Crystal Palace","Everton","Fulham","Leeds","Liverpool",
    "Man City","Man Utd","Newcastle","Nottm Forest","Sunderland",
    "Spurs","West Ham","Wolves"
])

TEAM_ABBREVIATIONS = {
    "Arsenal":"ARS","Aston Villa":"AVL","Bournemouth":"BOU","Brentford":"BRE",
    "Brighton":"BHA","Burnley":"BUR","Chelsea":"CHE","Crystal Palace":"CRY",
    "Everton":"EVE","Fulham":"FUL","Ipswich":"IPS","Leeds":"LEE",
    "Leicester":"LEI","Liverpool":"LIV","Man City":"MCI","Man Utd":"MUN",
    "Newcastle":"NEW","Nottm Forest":"NFO","Southampton":"SOU",
    "Sunderland":"SUN","Spurs":"TOT","West Ham":"WHU","Wolves":"WOL",
    "Tottenham Hotspur":"TOT","Manchester City":"MCI","Manchester United":"MUN",
}

TEAM_NAME_MAP = {
    "A.F.C. Bournemouth":"Bournemouth","Bournemouth":"Bournemouth",
    "Brighton & Hove Albion":"Brighton","Brighton":"Brighton",
    "Ipswich Town":"Ipswich","Ipswich":"Ipswich",
    "Leeds United":"Leeds","Leeds":"Leeds",
    "Leicester City":"Leicester","Leicester":"Leicester",
    "Manchester City":"Man City","Man City":"Man City",
    "Manchester United":"Man Utd","Man Utd":"Man Utd",
    "Newcastle United":"Newcastle","Newcastle":"Newcastle",
    "Nottingham Forest":"Nottm Forest","Nottm Forest":"Nottm Forest",
    "Nott'm Forest":"Nottm Forest",
    "Southampton FC":"Southampton","Southampton":"Southampton",
    "Tottenham Hotspur":"Spurs","Spurs":"Spurs","Tottenham":"Spurs",
    "West Ham United":"West Ham","West Ham":"West Ham",
    "Wolverhampton Wanderers":"Wolves","Wolves":"Wolves",
    "Sunderland AFC":"Sunderland","Sunderland":"Sunderland",
    "Burnley FC":"Burnley","Burnley":"Burnley",
    "Brentford FC":"Brentford","Brentford":"Brentford",
}

FDR_BG = {1:"#00ff85",2:"#50c369",3:"#D3D3D3",4:"#9d66a0",5:"#6f2a74",6:"#1E1E1E"}
FDR_FG = {1:"#31333F",2:"#31333F",3:"#31333F",4:"#FFFFFF",5:"#FFFFFF",6:"#FF4B4B"}

# ── Data helpers ──────────────────────────────────────────────────────────────

def get_fdr_score_from_rating(team_rating):
    if pd.isna(team_rating): return 3
    if team_rating >= FDR_THRESHOLDS[5]: return 5
    if team_rating >= FDR_THRESHOLDS[4]: return 4
    if team_rating >= FDR_THRESHOLDS[3]: return 3
    if team_rating >= FDR_THRESHOLDS[2]: return 2
    return 1

@st.cache_data
def load_csv_data():
    try:
        ratings_df  = pd.read_csv(RATINGS_CSV_FILE)
        fixtures_df = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error("CSV files not found. Ensure they are in the same folder as app.py.")
        return None, None
    ratings_df["Team"] = ratings_df["Team"].map(TEAM_NAME_MAP).fillna(ratings_df["Team"])
    fixtures_df["HomeTeam_std"] = fixtures_df["Home Team"].map(TEAM_NAME_MAP).fillna(fixtures_df["Home Team"])
    fixtures_df["AwayTeam_std"] = fixtures_df["Away Team"].map(TEAM_NAME_MAP).fillna(fixtures_df["Away Team"])
    return ratings_df, fixtures_df

@st.cache_data
def load_projections_csv():
    try:
        df = pd.read_csv(PROJECTIONS_CSV_FILE)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df["Team"] = df["Team"].map(lambda t: TEAM_NAME_MAP.get(t, t))
        return df
    except FileNotFoundError:
        return None

def build_live_fixtures_df(bootstrap, raw_fixtures):
    teams = pd.DataFrame(bootstrap["teams"])
    id_to_name  = dict(zip(teams["id"], teams["name"]))
    id_to_short = dict(zip(teams["id"], teams["short_name"]))
    rows = []
    for f in raw_fixtures:
        if f["event"] is None: continue
        h_name = TEAM_NAME_MAP.get(id_to_name.get(f["team_h"],""), id_to_name.get(f["team_h"],""))
        a_name = TEAM_NAME_MAP.get(id_to_name.get(f["team_a"],""), id_to_name.get(f["team_a"],""))
        rows.append({
            "GW":f["event"],"Home Team":h_name,"Away Team":a_name,
            "HomeTeam_std":h_name,"AwayTeam_std":a_name,
            "HomeTeamShort":id_to_short.get(f["team_h"],"???"),
            "AwayTeamShort":id_to_short.get(f["team_a"],"???"),
            "HomeTeamID":f["team_h"],"AwayTeamID":f["team_a"],
        })
    return pd.DataFrame(rows).sort_values(["GW","Home Team"]).reset_index(drop=True)

def build_dgw_bgw(raw_fixtures, bootstrap, start_gw, end_gw):
    teams_df   = pd.DataFrame(bootstrap["teams"])
    id_to_name = dict(zip(teams_df["id"], teams_df["name"]))
    gw_range   = range(start_gw, end_gw+1)
    counts = {}
    for f in raw_fixtures:
        if f["event"] is None or f["event"] not in gw_range: continue
        for tid in [f["team_h"], f["team_a"]]:
            counts[(tid, f["event"])] = counts.get((tid, f["event"]), 0) + 1
    dgw, bgw = {}, {}
    for tid in teams_df["id"]:
        name = TEAM_NAME_MAP.get(id_to_name.get(tid,""), id_to_name.get(tid,""))
        dg = [gw for gw in gw_range if counts.get((tid,gw),0) >= 2]
        bg = [gw for gw in gw_range if counts.get((tid,gw),0) == 0]
        if dg: dgw[name] = dg
        if bg: bgw[name] = bg
    return dgw, bgw

# ── xG / CS model (improved) ─────────────────────────────────────────────────
# Uses Dixon-Coles-style strength factors with home advantage multiplier.
# CS probability uses Poisson: P(0 goals conceded) = exp(-xG_against)
# xG adjusted for home/away advantage with a realistic 1.1x home boost.

HOME_ADVANTAGE = 1.10   # home teams score ~10% more than expected
AWAY_DISCOUNT  = 1.0 / HOME_ADVANTAGE

@st.cache_data
def create_all_data(fixtures_df_dict, start_gw, end_gw, ratings_df_dict, free_hit_gw=None):
    fixtures_df = pd.DataFrame(fixtures_df_dict)
    ratings_df  = pd.DataFrame(ratings_df_dict)
    ratings_dict = ratings_df.set_index("Team").to_dict("index")
    pl_ratings   = ratings_df[ratings_df["Team"].isin(PREMIER_LEAGUE_TEAMS)]

    # Normalise off/def so league average = 1.0
    avg_off = pl_ratings["Off Score"].mean()
    avg_def = pl_ratings["Def Score"].mean()

    # League average goals (true 25-season PL average)
    league_avg_goals = 1.36   # per team per game (neutral venue equivalent)

    gw_range        = range(start_gw, end_gw+1)
    projection_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df["GW"].isin(gw_range)].iterrows():
        home = row.get("HomeTeam_std") or row.get("Home Team","")
        away = row.get("AwayTeam_std") or row.get("Away Team","")
        gw_key = f"GW{row['GW']}"

        hs = ratings_dict.get(home)
        as_ = ratings_dict.get(away)
        if not (hs and as_ and "Off Score" in hs and "Def Score" in as_):
            continue

        # Attack strength relative to league average
        home_att = (hs["Off Score"]  / avg_off)
        home_def = (hs["Def Score"]  / avg_def)
        away_att = (as_["Off Score"] / avg_off)
        away_def = (as_["Def Score"] / avg_def)

        # xG: attacker strength × opponent defensive weakness × league avg × venue factor
        # Defensive weakness = 1 / def_strength  (lower def score = weaker = higher xG conceded)
        home_xg = home_att * (1 / away_def) * league_avg_goals * HOME_ADVANTAGE
        away_xg = away_att * (1 / home_def) * league_avg_goals * AWAY_DISCOUNT

        # CS probability = P(Poisson(xG) = 0)
        home_cs = math.exp(-away_xg)   # home team CS = away team scores 0
        away_cs = math.exp(-home_xg)

        home_fdr = get_fdr_score_from_rating(as_.get("Final Rating"))
        away_fdr = get_fdr_score_from_rating(hs.get("Final Rating"))

        def _store(team, gw_key, display, fdr, xg, cs):
            if gw_key in projection_data[team]:
                ex = projection_data[team][gw_key]
                projection_data[team][gw_key] = {
                    "display": ex["display"] + " + " + display,
                    "fdr": round((ex["fdr"] + fdr) / 2),
                    "xG": ex["xG"] + xg,
                    "CS": ex["CS"] + cs,  # additive probability for DGW
                }
            else:
                projection_data[team][gw_key] = {"display":display,"fdr":fdr,"xG":xg,"CS":cs}

        if home in PREMIER_LEAGUE_TEAMS:
            _store(home, gw_key, f"{TEAM_ABBREVIATIONS.get(away,'???')} (H)", home_fdr, home_xg, home_cs)
        if away in PREMIER_LEAGUE_TEAMS:
            _store(away, gw_key, f"{TEAM_ABBREVIATIONS.get(home,'???')} (A)", away_fdr, away_xg, away_cs)

    df = pd.DataFrame.from_dict(projection_data, orient="index").reindex(
        columns=[f"GW{i}" for i in gw_range])
    free_hit_col = f"GW{free_hit_gw}" if free_hit_gw else None

    total_difficulty, total_xg, total_cs = [], [], []
    for _, row in df.iterrows():
        fd = xg = cs = 0
        for gw_col, cell in row.items():
            if gw_col == free_hit_col: continue
            if isinstance(cell, dict):
                fd += cell.get("fdr", 0)
                xg += cell.get("xG", 0)
                cs += cell.get("CS", 0)
            else:
                fd += BGW_PENALTY_FDR
        total_difficulty.append(fd)
        total_xg.append(xg)
        total_cs.append(cs)

    df["Total Difficulty"] = total_difficulty
    df["Total xG"]         = total_xg
    df["xCS"]              = total_cs
    return df

@st.cache_data
def find_fixture_runs(fixtures_df_dict, rating_dict, start_gw):
    fixtures_df  = pd.DataFrame(fixtures_df_dict)
    all_fixtures = {t: [] for t in PREMIER_LEAGUE_TEAMS}
    for gw in range(1, 39):
        for _, row in fixtures_df[fixtures_df["GW"] == gw].iterrows():
            home = row.get("HomeTeam_std") or row.get("Home Team","")
            away = row.get("AwayTeam_std") or row.get("Away Team","")
            if home in PREMIER_LEAGUE_TEAMS:
                r = rating_dict.get(away,{}).get("Final Rating")
                all_fixtures[home].append({"gw":gw,"opp":away,"loc":"H","fdr":get_fdr_score_from_rating(r)})
            if away in PREMIER_LEAGUE_TEAMS:
                r = rating_dict.get(home,{}).get("Final Rating")
                all_fixtures[away].append({"gw":gw,"opp":home,"loc":"A","fdr":get_fdr_score_from_rating(r)})
    good_runs = {}
    for team, fixs in all_fixtures.items():
        cur = []
        for f in sorted(fixs, key=lambda x: x["gw"]):
            if f["gw"] < start_gw: continue
            if f["fdr"] is not None and f["fdr"] <= 3:
                cur.append(f)
            else:
                if len(cur) >= 3: good_runs.setdefault(team,[]).append(cur)
                cur = []
        if len(cur) >= 3: good_runs.setdefault(team,[]).append(cur)
    return good_runs

# ── Captain helpers ───────────────────────────────────────────────────────────

def get_fdr_for_team_gw(team_name, gw, master_df_full):
    """Return (fixture_str, fdr_int) from your custom ratings."""
    gw_col   = f"GW{gw}"
    std_name = TEAM_NAME_MAP.get(team_name, team_name)
    if std_name not in master_df_full.index or gw_col not in master_df_full.columns:
        return "?", 3
    cell = master_df_full.loc[std_name, gw_col]
    if isinstance(cell, dict):
        return cell.get("display","?"), cell.get("fdr", 3)
    return "BGW", 6

def get_captain_picks(proj_df, gw, master_df_full, bootstrap):
    """Top 2 EV + 1 differential. Returns list of dicts including photo_url."""
    pts_col  = f"{gw}_Pts"
    mins_col = f"{gw}_xMins"
    if pts_col not in proj_df.columns:
        return []

    df = proj_df.copy()
    df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
    df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")
    df["Elite%"] = pd.to_numeric(df["Elite%"],  errors="coerce")

    active = df[df[mins_col] > 45].copy()
    top2   = active.nlargest(2, pts_col).copy()
    top2["PickType"] = ["🏆 Top Pick","🥈 2nd Pick"]

    diff_pool = active[(active["Elite%"] < 0.10) & (~active["Name"].isin(top2["Name"]))]
    diff = diff_pool.nlargest(1, pts_col).copy()
    diff["PickType"] = ["🎯 Differential"]

    picks = pd.concat([top2, diff], ignore_index=True)

    # Build name→code lookup from bootstrap
    code_lookup = {}
    if bootstrap:
        for p in bootstrap["elements"]:
            code_lookup[p["web_name"]] = p["code"]
            # also store first+last name combos
            full = f"{p.get('first_name','')} {p.get('second_name','')}".strip()
            code_lookup[full] = p["code"]

    result = []
    for _, r in picks.iterrows():
        fix, fdr = get_fdr_for_team_gw(r["Team"], gw, master_df_full)
        # Try to find player photo code
        code = code_lookup.get(r["Name"])
        if code is None:
            # fuzzy: match by last name token
            for key, val in code_lookup.items():
                if r["Name"].split(".")[-1].strip().lower() in key.lower():
                    code = val
                    break
        photo = player_photo_url(code) if code else None
        result.append({
            "PickType": r["PickType"],
            "Name":     r["Name"],
            "Team":     r["Team"],
            "Pos":      r["Pos"],
            "EV":       round(float(r[pts_col]), 2),
            "Fixture":  fix,
            "FDR":      fdr,
            "photo":    photo,
        })
    return result

def get_captain_matrix(proj_df, gws, master_df_full, bootstrap):
    """All players within 0.5 EV of top pick per GW. Includes photo + club colour."""
    # Build name→code lookup
    code_lookup = {}
    if bootstrap:
        for p in bootstrap["elements"]:
            code_lookup[p["web_name"]] = p["code"]
            full = f"{p.get('first_name','')} {p.get('second_name','')}".strip()
            code_lookup[full] = p["code"]

    matrix = {}
    for gw in gws:
        pts_col  = f"{gw}_Pts"
        mins_col = f"{gw}_xMins"
        if pts_col not in proj_df.columns: continue

        df = proj_df.copy()
        df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
        df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")

        active = df[df[mins_col] > 45]
        top_ev = active[pts_col].max()
        if pd.isna(top_ev): continue

        within = active[active[pts_col] >= top_ev - 0.5].nlargest(10, pts_col)

        rows = []
        for _, r in within.iterrows():
            fix, fdr = get_fdr_for_team_gw(r["Team"], gw, master_df_full)
            code = code_lookup.get(r["Name"])
            if code is None:
                for key, val in code_lookup.items():
                    if r["Name"].split(".")[-1].strip().lower() in key.lower():
                        code = val
                        break
            photo = player_photo_url(code) if code else None
            bg, fg = club_style(r["Team"])
            rows.append({
                "Name":    r["Name"],
                "Team":    r["Team"],
                "EV":      round(float(r[pts_col]), 2),
                "Fixture": fix,
                "FDR":     fdr,
                "photo":   photo,
                "club_bg": bg,
                "club_fg": fg,
            })
        matrix[gw] = rows
    return matrix

# ── Live radar helpers ─────────────────────────────────────────────────────────

def get_price_changes(bootstrap):
    players = pd.DataFrame(bootstrap["elements"])
    df = players[[c for c in ["web_name","team","element_type","now_cost","cost_change_event"]
                  if c in players.columns]].copy()
    df = df[df["cost_change_event"] != 0].copy()
    df["Price (£m)"] = df["now_cost"] / 10.0
    df["Change"]     = df["cost_change_event"] / 10.0
    teams_df    = pd.DataFrame(bootstrap["teams"])
    id2short    = dict(zip(teams_df["id"], teams_df["short_name"]))
    POS         = {1:"GKP",2:"DEF",3:"MID",4:"FWD"}
    df["Team"]     = df["team"].map(id2short)
    df["Position"] = df["element_type"].map(POS)
    df.rename(columns={"web_name":"Player"}, inplace=True)
    return df[["Player","Team","Position","Price (£m)","Change"]].sort_values("Change", ascending=False)

def get_injury_status(bootstrap):
    players = pd.DataFrame(bootstrap["elements"])
    cols = [c for c in ["web_name","team","element_type","status","news",
                         "chance_of_playing_next_round"] if c in players.columns]
    df = players[cols].copy()
    df = df[df["status"].isin(["d","i","s","u"])].copy()
    teams_df = pd.DataFrame(bootstrap["teams"])
    id2short = dict(zip(teams_df["id"], teams_df["short_name"]))
    POS  = {1:"GKP",2:"DEF",3:"MID",4:"FWD"}
    SMAP = {"i":"🔴 Injured","d":"⚠️ Doubt","s":"🟡 Suspended","u":"❌ Unavailable"}
    SORD = {"🔴 Injured":0,"⚠️ Doubt":1,"🟡 Suspended":2,"❌ Unavailable":3}
    df["Team"]     = df["team"].map(id2short)
    df["Position"] = df["element_type"].map(POS)
    df["Status"]   = df["status"].map(SMAP)
    df["Play%"]    = df["chance_of_playing_next_round"].apply(
        lambda x: f"{int(x)}%" if pd.notna(x) else "?")
    df.rename(columns={"web_name":"Player","news":"News"}, inplace=True)
    df["_sort"]     = df["Status"].map(SORD).fillna(9)
    df["_play_num"] = df["chance_of_playing_next_round"].fillna(50)
    df = df.sort_values(["_sort","_play_num"]).reset_index(drop=True)
    return df[["Player","Team","Position","Status","Play%","News"]]

# =============================================================================
# MAIN APP
# =============================================================================

st.set_page_config(layout="wide")
st.title("🏆 CoachFPL Command Center")

with st.sidebar:
    st.header("⚙️ Settings")
    use_live = st.toggle("📡 Live FPL Data", value=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.success("✅ Cleared!"); st.rerun()
    with c2:
        if st.button("🔄 Rerun", use_container_width=True): st.rerun()

# ── Load live data ─────────────────────────────────────────────────────────────
bootstrap = raw_fixtures = None
live_ok   = False
current_gw = 29

if use_live:
    try:
        bootstrap    = _fetch_bootstrap()
        raw_fixtures = _fetch_live_fixtures()
        live_ok      = True
        current_gw   = get_current_gw(bootstrap)
        st.sidebar.success(f"✅ Live | GW{current_gw}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Live failed: {e}")

ratings_df, fixtures_df = load_csv_data()
proj_df = load_projections_csv()

if live_ok and bootstrap and raw_fixtures:
    fixtures_df = build_live_fixtures_df(bootstrap, raw_fixtures)

with st.expander("Glossary"):
    st.markdown(f"""
    - **FDR 1–5:** Fixture Difficulty (lower = easier). Based on your custom team ratings.
    - **xG:** Expected goals scored (Dixon-Coles model, home advantage adjusted).
    - **xCS:** Expected clean sheet probability (Poisson P(0 goals conceded)).
    - **EV:** Expected projected points from your data file.
    - **Differential:** High-value pick with low ownership among top managers.
    - **Captain Matrix:** All captains within 0.5 EV of the top pick per GW.
    - **Live:** {"🟢 Active" if live_ok else "🔴 Off — CSV mode"}
    """)

if ratings_df is None or fixtures_df is None:
    st.stop()

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Controls")
cs, ce = st.sidebar.columns(2)
default_gw = current_gw if live_ok else 29
with cs: start_gw = st.number_input("Start GW:", min_value=1, max_value=38, value=default_gw)
with ce: end_gw   = st.number_input("End GW:",   min_value=1, max_value=38, value=min(default_gw+4, 38))

selected_teams = st.sidebar.multiselect("Teams:", PREMIER_LEAGUE_TEAMS, default=PREMIER_LEAGUE_TEAMS)
fh_opts = [None] + list(range(start_gw, end_gw+1))
free_hit_gw = st.sidebar.selectbox("Free Hit GW:", fh_opts,
                                    format_func=lambda x: "None" if x is None else f"GW{x}")

# Build full master_df (GW29–38 for captain lookups)
if proj_df is not None:
    proj_gws  = sorted([int(c.split("_")[0]) for c in proj_df.columns
                           if c.endswith("_Pts") and c.split("_")[0].isdigit()])
    future_gws  = [g for g in proj_gws if g >= current_gw]
    all_start   = min(future_gws) if future_gws else start_gw
    all_end     = 38
    master_df_full = create_all_data(
        fixtures_df.to_dict("records"), all_start, all_end,
        ratings_df.to_dict("records"), None
    )
else:
    future_gws = []
    master_df_full = None

# Display master_df (selected GW range)
master_df = create_all_data(
    fixtures_df.to_dict("records"), start_gw, end_gw,
    ratings_df.to_dict("records"), free_hit_gw
)
if selected_teams:
    master_df = master_df.loc[[t for t in master_df.index if t in selected_teams]]

gw_columns = [f"GW{i}" for i in range(start_gw, end_gw+1)]
if free_hit_gw and f"GW{free_hit_gw}" in gw_columns:
    gw_columns.remove(f"GW{free_hit_gw}")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 FDR", "⚽ xG", "🧤 xCS",
    "🎯 Captain Picks", "🏅 Captain Matrix",
    "📡 Live Radar", "📈 Team Ratings",
])

# ── Tab 1: FDR ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Fixture Difficulty Rating")
    st.caption("🟢 1–2 Easy  |  ⬜ 3 Neutral  |  🟣 4–5 Hard  |  🔴 BGW = Blank (no fixture). Sorted easiest → hardest.")
    df_display = master_df.sort_values("Total Difficulty").reset_index().rename(columns={"index":"Team"})
    df_display = df_display[["Team","Total Difficulty"] + gw_columns]

    df_for_grid = df_display[["Team","Total Difficulty"]].copy()
    for col in gw_columns:
        df_for_grid[col] = df_display[col]
        df_for_grid[f"{col}_display"] = df_display[col].apply(
            lambda x: x["display"] if isinstance(x, dict) else "BGW")
        df_for_grid[f"{col}_fdr"] = df_display[col].apply(
            lambda x: x["fdr"] if isinstance(x, dict) else BGW_PENALTY_FDR)

    gb = GridOptionsBuilder.from_dataframe(df_for_grid)
    gb.configure_column("Team", pinned="left", flex=2, minWidth=150, sortable=True)
    gb.configure_column("Total Difficulty", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)
    for col in gw_columns:
        gb.configure_column(f"{col}_display", hide=True)
        gb.configure_column(f"{col}_fdr", hide=True)
    for col in gw_columns:
        gb.configure_column(col, headerName=col,
            valueGetter=JsCode(f"function(p){{return p.data['{col}_fdr'];}}"),
            valueFormatter=JsCode(f"function(p){{return p.data['{col}_display']||'';}}"),
            cellStyle=JsCode(f"""function(p){{
                var d=p.data['{col}_display'];
                if(d==='BGW')return{{'backgroundColor':'#1E1E1E','color':'#FF4B4B','fontWeight':'bold','textAlign':'center','border':'1px solid #FF4B4B'}};
                var v=p.data['{col}_fdr'];
                var c={{1:'#00ff85',2:'#50c369',3:'#D3D3D3',4:'#9d66a0',5:'#6f2a74'}};
                var bg=c[Math.round(v)]||'#444';
                var fg=(v<=3)?'#31333F':'#FFFFFF';
                return{{'backgroundColor':bg,'color':fg,'fontWeight':'bold','textAlign':'center'}};
            }}"""),
            flex=1, minWidth=90, sortable=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
    AgGrid(df_for_grid, gridOptions=gb.build(), allow_unsafe_jscode=True,
           theme="streamlit-dark", height=(len(df_for_grid)+1)*35,
           fit_columns_on_grid_load=True, key=f"fdr_{start_gw}_{end_gw}")

# ── Tab 2: xG ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Expected Goals (xG)")
    st.caption("Attack strength ÷ opponent defence × league average goals. Home advantage built in. Darker green = more goals expected. 🔴 = Blank gameweek.")
    df_display = master_df.sort_values("Total xG", ascending=False).reset_index().rename(columns={"index":"Team"})
    df_display = df_display[["Team","Total xG"] + gw_columns]

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
    gb.configure_column("Team", pinned="left", flex=2, minWidth=150)
    gb.configure_column("Total xG", valueFormatter="data['Total xG'].toFixed(2)", flex=1.5,
                         type=["numericColumn"], minWidth=140)

    # 5-tier green gradient for xG
    jscode = JsCode("""function(p){
        var c=p.data[p.colDef.field];
        if(c&&typeof c==='object'&&c.xG!==undefined){
            var x=c.xG;
            var bg=x>=2.2?'#1a7a4a':x>=1.8?'#2da65c':x>=1.4?'#63be7b':x>=1.0?'#a8d8a8':'#D3D3D3';
            var fg=x>=1.0?'#FFFFFF':'#31333F';
            return{'backgroundColor':bg,'color':fg,'fontWeight':'bold','textAlign':'center'};
        }
        return{'textAlign':'center','backgroundColor':'#1E1E1E','color':'#FF4B4B',
               'fontWeight':'bold','border':'1px solid #FF4B4B'};
    }""")
    for col in gw_columns:
        gb.configure_column(col, headerName=col,
            valueGetter=f"(data['{col}']&&typeof data['{col}']==='object')?data['{col}'].xG.toFixed(2):'BGW'",
            cellStyle=jscode, flex=1, minWidth=90)
    AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
           theme="streamlit-dark", height=(len(df_display)+1)*35,
           fit_columns_on_grid_load=True, key=f"xg_{start_gw}_{end_gw}")

# ── Tab 3: xCS ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Expected Clean Sheet % (xCS)")
    st.caption("Probability of keeping a clean sheet = e^(−xG conceded). 🟢 ≥50 %  →  🟣 <15 %. Higher = better for defenders & keepers.")
    df_display = master_df.sort_values("xCS", ascending=False).reset_index().rename(columns={"index":"Team"})
    df_display = df_display[["Team","xCS"] + gw_columns]

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_column("Team", pinned="left", flex=2, minWidth=150)
    gb.configure_column("xCS", pinned="left",
                         valueFormatter="(data['xCS']*100).toFixed(1)+'%'",
                         flex=1.5, type=["numericColumn"], minWidth=140)

    jscode = JsCode("""function(p){
        var c=p.data[p.colDef.field];
        if(c&&typeof c==='object'&&c.CS!==undefined){
            var s=c.CS;
            var bg=s>=0.50?'#00ff85':s>=0.40?'#50c369':s>=0.30?'#a8d8a8':s>=0.20?'#D3D3D3':s>=0.10?'#9d66a0':'#6f2a74';
            var fg=(s>=0.20&&s<0.30)?'#31333F':'#FFFFFF';
            if(s>=0.50||s<0.10) fg='#31333F';
            if(s<0.10) fg='#FFFFFF';
            if(s>=0.50) fg='#31333F';
            return{'backgroundColor':bg,'color':fg,'fontWeight':'bold','textAlign':'center'};
        }
        return{'textAlign':'center','backgroundColor':'#1E1E1E','color':'#FF4B4B',
               'fontWeight':'bold','border':'1px solid #FF4B4B'};
    }""")
    for col in gw_columns:
        gb.configure_column(col, headerName=col,
            valueGetter=f"(data['{col}']&&typeof data['{col}']==='object')?(data['{col}'].CS*100).toFixed(0)+'%':'BGW'",
            cellStyle=jscode, flex=1, minWidth=90)
    gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
    AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
           theme="streamlit-dark", height=(len(df_display)+1)*35, key=f"cs_{start_gw}_{end_gw}")

# ── Tab 4: Captain Picks ──────────────────────────────────────────────────────
with tab4:
    st.subheader("🎯 Captain Picks")
    if proj_df is None:
        st.error("❌ projections.csv not found. Place it in the app folder.")
    elif master_df_full is None:
        st.error("❌ Could not build fixture data.")
    else:
        gw_sel = st.selectbox("Select GW:", future_gws, key="cap_gw")
        picks  = get_captain_picks(proj_df, gw_sel, master_df_full, bootstrap)

        if not picks:
            st.warning(f"No data for GW{gw_sel}.")
        else:
            st.caption(f"GW{gw_sel} recommendations — Top 2 picks + 1 Differential. FDR from your custom ratings.")

            TYPE_STYLE = {
                "🏆 Top Pick":     ("background:#1a472a","color:#00ff85"),
                "🥈 2nd Pick":     ("background:#1a3a47","color:#50c369"),
                "🎯 Differential": ("background:#3d2b1f","color:#f4a261"),
            }

            # Render as card row
            cols = st.columns(len(picks))
            for col, p in zip(cols, picks):
                bg_label, fg_label = TYPE_STYLE.get(p["PickType"], ("background:#222","color:#fff"))
                fdr_bg = FDR_BG.get(p["FDR"], "#444")
                fdr_fg = FDR_FG.get(p["FDR"], "#fff")
                club_bg, club_fg = club_style(p["Team"])

                img_html = ""
                if p["photo"]:
                    img_html = (f'<img src="{p["photo"]}" '
                                f'style="width:80px;height:100px;object-fit:cover;'
                                f'border-radius:8px;margin-bottom:6px;'
                                f'border:2px solid {club_bg}" '
                                f'onerror="this.style.display=\'none\'">')

                card = f"""
                <div style="border-radius:10px;overflow:hidden;margin:4px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.4)">
                  <div style="{bg_label};{fg_label};padding:6px 10px;
                              font-size:12px;font-weight:bold;text-align:center">
                    {p['PickType']}
                  </div>
                  <div style="background:{club_bg};color:{club_fg};
                              padding:12px;text-align:center">
                    {img_html}
                    <div style="font-size:16px;font-weight:bold;margin-top:4px">{p['Name']}</div>
                    <div style="font-size:12px;opacity:0.85">{p['Team']} · {p['Pos']}</div>
                    <div style="font-size:22px;font-weight:bold;margin:6px 0">{p['EV']} pts</div>
                    <span style="background:{fdr_bg};color:{fdr_fg};
                                 border-radius:4px;padding:2px 8px;
                                 font-size:12px;font-weight:bold">
                      {p['Fixture']}
                    </span>
                  </div>
                </div>"""
                col.markdown(card, unsafe_allow_html=True)

# ── Tab 5: Captain Matrix ─────────────────────────────────────────────────────
with tab5:
    st.subheader("🏅 Captain Matrix — Within 0.5 EV of Top Pick")
    if proj_df is None:
        st.error("❌ projections.csv not found. Place it in the app folder.")
    elif master_df_full is None:
        st.error("❌ Fixture data unavailable.")
    else:
        # Default: current GW → GW38
        matrix_gws = st.multiselect(
            "Show GWs:", future_gws,
            default=future_gws,   # ALL future GWs by default
            key="matrix_gws"
        )
        if matrix_gws:
            matrix = get_captain_matrix(proj_df, matrix_gws, master_df_full, bootstrap)

            # Build pure-inline HTML table with photos + club colours
            col_w = max(130, min(190, 1100 // len(matrix_gws)))

            header = "".join(
                f'<th style="background:#1a1a1a;color:#aaa;padding:8px 6px;'
                f'border:1px solid #333;text-align:center;min-width:{col_w}px;'
                f'font-size:13px;font-family:sans-serif">GW{gw}</th>'
                for gw in matrix_gws
            )

            max_rows = max((len(matrix.get(gw,[])) for gw in matrix_gws), default=0)
            body = ""
            for ri in range(max_rows):
                row_html = ""
                for gw in matrix_gws:
                    rows = matrix.get(gw, [])
                    if ri < len(rows):
                        r = rows[ri]
                        bg    = r["club_bg"]
                        fg    = r["club_fg"]
                        fdr_b = FDR_BG.get(r["FDR"], "#444")
                        fdr_f = FDR_FG.get(r["FDR"], "#fff")
                        top_b = "border-top:2px solid rgba(255,255,255,0.3);" if ri == 0 else ""
                        name_w = "font-weight:bold" if ri == 0 else ""

                        img_tag = ""
                        if r["photo"]:
                            img_tag = (
                                f'<img src="{r["photo"]}" '
                                f'style="width:40px;height:50px;object-fit:cover;'
                                f'border-radius:4px;vertical-align:middle;'
                                f'margin-right:6px;border:1px solid rgba(255,255,255,0.3)" '
                                f'onerror="this.style.display=\'none\'">'
                            )

                        row_html += (
                            f'<td style="padding:6px 6px;border:1px solid #2a2a2a;'
                            f'{top_b}background:{bg};vertical-align:middle">'
                            f'<div style="display:flex;align-items:center">'
                            f'{img_tag}'
                            f'<div>'
                            f'<div style="color:{fg};{name_w};font-size:12px;'
                            f'font-family:sans-serif;white-space:nowrap">{r["Name"]}</div>'
                            f'<div style="display:flex;align-items:center;gap:4px;margin-top:2px">'
                            f'<span style="background:{fdr_b};color:{fdr_f};border-radius:3px;'
                            f'padding:1px 5px;font-size:11px;font-weight:bold">{r["Fixture"]}</span>'
                            f'<span style="color:{fg};font-weight:bold;font-size:13px">{r["EV"]}</span>'
                            f'</div></div></div></td>'
                        )
                    else:
                        row_html += '<td style="border:1px solid #2a2a2a;background:#111"></td>'
                body += f"<tr>{row_html}</tr>"

            html = (
                f'<div style="overflow-x:auto;border-radius:8px;'
                f'border:1px solid #333;margin-top:8px">'
                f'<table style="border-collapse:collapse;width:100%">'
                f'<thead><tr>{header}</tr></thead>'
                f'<tbody>{body}</tbody>'
                f'</table></div>'
            )
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Top row = highest EV. Colour = club colours. Badge = your custom FDR. EV = projected points.")

# ── Tab 6: Live Radar ─────────────────────────────────────────────────────────
with tab6:
    st.subheader("📡 Live Radar")
    if not live_ok:
        st.warning("⚠️ Enable Live FPL Data in the sidebar.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 📈 Ownership Movers")
            players_live = pd.DataFrame(bootstrap["elements"])
            movers = players_live[["web_name","selected_by_percent",
                                    "transfers_in_event","transfers_out_event"]].copy()
            for c in movers.columns[1:]:
                movers[c] = pd.to_numeric(movers[c], errors="coerce")

            st.markdown("**🔼 Most transferred IN**")
            tin = movers.nlargest(8,"transfers_in_event")[["web_name","selected_by_percent","transfers_in_event"]]
            tin.columns = ["Player","Sel%","In"]
            st.dataframe(tin, hide_index=True, use_container_width=True)

            st.markdown("**🔽 Most transferred OUT**")
            tout = movers.nlargest(8,"transfers_out_event")[["web_name","selected_by_percent","transfers_out_event"]]
            tout.columns = ["Player","Sel%","Out"]
            st.dataframe(tout, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("### 💰 Price Changes")
            try:
                pdf = get_price_changes(bootstrap)
                if pdf.empty:
                    st.info("No price changes this GW.")
                else:
                    rises = pdf[pdf["Change"] > 0]
                    falls = pdf[pdf["Change"] < 0]
                    if not rises.empty:
                        st.markdown("**🔼 Rising**")
                        st.dataframe(rises.reset_index(drop=True), use_container_width=True, hide_index=True)
                    if not falls.empty:
                        st.markdown("**🔽 Falling**")
                        st.dataframe(falls.reset_index(drop=True), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Price data unavailable: {e}")

        with col_b:
            st.markdown("### ⚠️ DGW / BGW Radar")
            radar_end = st.number_input("Show up to GW:", min_value=current_gw,
                                         max_value=38, value=min(current_gw+5, 38), key="radar_gw")
            dgw, bgw = build_dgw_bgw(raw_fixtures, bootstrap, current_gw, radar_end)
            if dgw:
                st.markdown("**🟢 Double Gameweeks**")
                for name, gws in sorted(dgw.items()):
                    st.write(f"- **{name}**: GW{', GW'.join(map(str,gws))}")
            else:
                st.info("No DGWs detected.")
            if bgw:
                st.markdown("**🔴 Blank Gameweeks**")
                for name, gws in sorted(bgw.items()):
                    st.write(f"- **{name}**: GW{', GW'.join(map(str,gws))}")
            else:
                st.info("No BGWs detected.")

            st.markdown("---")
            st.markdown("### 🚑 Injuries & Availability")
            try:
                inj = get_injury_status(bootstrap)
                if inj.empty:
                    st.success("No injuries or doubts reported.")
                else:
                    for label in ["🔴 Injured","⚠️ Doubt","🟡 Suspended","❌ Unavailable"]:
                        sub = inj[inj["Status"] == label]
                        if not sub.empty:
                            st.markdown(f"**{label}** ({len(sub)})")
                            st.dataframe(sub[["Player","Team","Position","Play%","News"]].reset_index(drop=True),
                                         use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Injury data unavailable: {e}")

# ── Tab 7: Team Ratings ───────────────────────────────────────────────────────
with tab7:
    st.subheader("📊 Team Strength Map")
    st.caption(
        "X axis → Better Defence (right = harder to score against).  "
        "Y axis ↑ Better Attack (top = more goals scored).  "
        "Top-right = elite. Dashed lines show last 6 GW trend (recent form)."
    )

    pl_ratings = ratings_df[ratings_df["Team"].isin(PREMIER_LEAGUE_TEAMS)].copy()

    if pl_ratings.empty:
        st.warning("No PL teams found in ratings file.")
    elif "Off Score" not in pl_ratings.columns or "Def Score" not in pl_ratings.columns:
        st.warning("Ratings file must have 'Off Score' and 'Def Score' columns.")
    else:
        plot_df = pl_ratings[["Team","Off Score","Def Score"]].copy()
        plot_df["Abbr"] = plot_df["Team"].map(TEAM_ABBREVIATIONS).fillna(
            plot_df["Team"].str[:3].str.upper())
        plot_df["club_bg"]   = plot_df["Team"].apply(
            lambda t: CLUB_COLORS.get(t, {"bg":"#444444"})["bg"])
        plot_df["club_text"] = plot_df["Team"].apply(
            lambda t: CLUB_COLORS.get(t, {"text":"#ffffff"}).get("text","#ffffff"))

        # Axis bounds with padding
        pad_x = (plot_df["Def Score"].max() - plot_df["Def Score"].min()) * 0.12
        pad_y = (plot_df["Off Score"].max() - plot_df["Off Score"].min()) * 0.18
        x_min = plot_df["Def Score"].min() - pad_x
        x_max = plot_df["Def Score"].max() + pad_x
        y_min = plot_df["Off Score"].min() - pad_y
        y_max = plot_df["Off Score"].max() + pad_y

        x_avg = plot_df["Def Score"].mean()
        y_avg = plot_df["Off Score"].mean()

        fig = go.Figure()

        # Subtle quadrant lines
        for shape_args in [
            dict(x0=x_avg, x1=x_avg, y0=y_min, y1=y_max),
            dict(x0=x_min, x1=x_max, y0=y_avg, y1=y_avg),
        ]:
            fig.add_shape(type="line", **shape_args,
                          line=dict(color="rgba(255,255,255,0.10)", width=1, dash="dot"))

        # ── Pin-style markers ──────────────────────────────────────────────────
        # We draw each team as: a filled rectangle annotation (the badge)
        # + a tiny SVG triangle below it using a custom marker
        # Plotly doesn't natively support "pin" symbols, but we can combine
        # a large square marker with text, plus a small downward triangle marker
        # at the same position to create the pin look.

        MARKER_SIZE  = 36   # badge square
        PIN_SIZE     = 10   # triangle point below
        PIN_OFFSET   = 0.0  # will be computed per-axis in data coords

        # Compute a fixed pixel offset in data units for the pin tip
        # (approximate: we nudge Y downward slightly)
        y_range = y_max - y_min
        y_nudge = y_range * 0.025   # tiny offset for the pin tip

        for _, row in plot_df.iterrows():
            bg   = row["club_bg"]
            fg   = row["club_text"]
            abbr = row["Abbr"]
            team = row["Team"]
            x    = row["Def Score"]
            y    = row["Off Score"]

            # 1. The badge (large rounded square, coloured by club)
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=MARKER_SIZE,
                    color=bg,
                    symbol="square",
                    line=dict(color="rgba(255,255,255,0.18)", width=1),
                ),
                text=[f"<b>{abbr}</b>"],
                textfont=dict(color=fg, size=10, family="Arial Black, sans-serif"),
                textposition="middle center",
                name=team,
                showlegend=False,
                hoverinfo="skip",
            ))

            # 2. The pin triangle (small triangle below badge)
            fig.add_trace(go.Scatter(
                x=[x], y=[y - y_nudge],
                mode="markers",
                marker=dict(
                    size=PIN_SIZE,
                    color=bg,
                    symbol="triangle-down",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{team}</b><br>"
                    f"Attack: {y:.2f}<br>"
                    f"Defence: {x:.2f}<extra></extra>"
                ),
            ))

        # Subtle quadrant corner labels
        ann_style = dict(showarrow=False, font=dict(size=9, color="rgba(255,255,255,0.18)"))
        fig.add_annotation(x=x_max, y=y_max, text="ELITE", xanchor="right", yanchor="top",    **ann_style)
        fig.add_annotation(x=x_min, y=y_max, text="ATTACK", xanchor="left",  yanchor="top",   **ann_style)
        fig.add_annotation(x=x_max, y=y_min, text="SOLID",  xanchor="right", yanchor="bottom", **ann_style)
        fig.add_annotation(x=x_min, y=y_min, text="WEAK",   xanchor="left",  yanchor="bottom", **ann_style)

        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#cccccc", family="sans-serif", size=12),
            xaxis=dict(
                title=dict(text="Better Defence →", font=dict(size=12, color="#888")),
                range=[x_min, x_max],
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=10, color="#555"),
            ),
            yaxis=dict(
                title=dict(text="Better Attack ↑", font=dict(size=12, color="#888")),
                range=[y_min, y_max],
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=10, color="#555"),
            ),
            margin=dict(l=55, r=25, t=15, b=55),
            height=580,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Ranking table ──────────────────────────────────────────────────────
        st.markdown("---")
        rank_df = plot_df[["Team","Abbr","Off Score","Def Score"]].copy()
        off_max = rank_df["Off Score"].max()
        def_max = rank_df["Def Score"].max()
        rank_df["Overall"] = (
            (rank_df["Off Score"] / off_max + rank_df["Def Score"] / def_max) / 2 * 100
        ).round(1)
        rank_df = rank_df.sort_values("Overall", ascending=False).reset_index(drop=True)

        rows_html = ""
        for i, row in rank_df.iterrows():
            bg, fg = club_style(row["Team"])
            medal  = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}"
            dot = (f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                   f'background:{bg};margin-right:8px;vertical-align:middle"></span>')
            rows_html += (
                f'<tr style="border-bottom:1px solid #1e1e1e">'
                f'<td style="width:4px;background:{bg};padding:0"></td>'
                f'<td style="padding:6px 10px;text-align:center;color:#777;font-size:13px">{medal}</td>'
                f'<td style="padding:6px 10px;font-weight:bold;color:#fff;font-size:13px">{dot}{row["Team"]}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:#63be7b;font-size:13px">{row["Off Score"]:.2f}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:#6CABDD;font-size:13px">{row["Def Score"]:.2f}</td>'
                f'<td style="padding:6px 10px;text-align:center;font-weight:bold;color:#f4a261;font-size:13px">{row["Overall"]:.1f}</td>'
                f'</tr>'
            )

        st.markdown(
            '<div style="overflow-x:auto;border-radius:6px;border:1px solid #222;margin-top:4px">'
            '<table style="border-collapse:collapse;width:100%;font-family:sans-serif;background:#0d1117">'
            '<thead style="background:#161b22"><tr>'
            '<th style="width:4px;padding:0"></th>'
            '<th style="padding:6px 10px;color:#555;font-size:11px;font-weight:normal;text-align:center">#</th>'
            '<th style="padding:6px 10px;color:#555;font-size:11px;font-weight:normal;text-align:left">Team</th>'
            '<th style="padding:6px 10px;color:#63be7b;font-size:11px;font-weight:normal;text-align:center">⚔️ Attack</th>'
            '<th style="padding:6px 10px;color:#6CABDD;font-size:11px;font-weight:normal;text-align:center">🛡️ Defence</th>'
            '<th style="padding:6px 10px;color:#f4a261;font-size:11px;font-weight:normal;text-align:center">Overall %</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True
        )

# ── Easy Run Finder ───────────────────────────────────────────────────────────
st.markdown("---")
st.sidebar.header("Easy Run Finder")
st.sidebar.info("Find 3+ consecutive easy fixtures (FDR 1–3).")
teams_to_check = st.sidebar.multiselect("Teams to check:", PREMIER_LEAGUE_TEAMS, default=[], key="run_teams")

st.header("✅ Easy Fixture Runs")
if teams_to_check:
    rating_dict = ratings_df.set_index("Team").to_dict("index")
    all_runs    = find_fixture_runs(fixtures_df.to_dict("records"), rating_dict, start_gw)
    found = False
    for team in teams_to_check:
        runs = all_runs.get(team)
        if runs:
            found = True
            with st.expander(f"**{team}** — {len(runs)} run(s)"):
                for i, run in enumerate(runs):
                    s, e = run[0]["gw"], run[-1]["gw"]
                    st.markdown(f"**Run {i+1}: GW{s} – GW{e}**")
                    for fx in run:
                        opp = TEAM_ABBREVIATIONS.get(fx["opp"],"???")
                        st.markdown(f"- **GW{fx['gw']}:** {opp} ({fx['loc']}) — FDR {fx['fdr']}")
    if not found:
        st.warning(f"No easy runs found from GW{start_gw}.")
else:
    st.info("Select teams from the sidebar.")

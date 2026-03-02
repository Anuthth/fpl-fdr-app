import pandas as pd
import streamlit as st
import numpy as np
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
PROJECTIONS_CSV_FILE = "projections.csv"
EO_CSV_FILE          = "EO_.csv"          # Solio expected ownership file

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
    """Load Solio projections.csv — columns: Pos,ID,Name,BV,SV,Team,{GW}_xMins,{GW}_Pts"""
    try:
        df = pd.read_csv(PROJECTIONS_CSV_FILE)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "Team" in df.columns:
            df["Team"] = df["Team"].map(lambda t: TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()))
        return df
    except FileNotFoundError:
        return None

def load_eo_csv():
    """Load Solio EO_.csv — columns: Pos,ID,Name,BV,SV,Team,{GW}_xMins,{GW}_eo"""
    try:
        df = pd.read_csv(EO_CSV_FILE)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "Team" in df.columns:
            df["Team"] = df["Team"].map(lambda t: TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()))
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

def _build_code_lookup(bootstrap):
    """Build name → FPL photo code dict."""
    lookup = {}
    if not bootstrap:
        return lookup
    for p in bootstrap["elements"]:
        lookup[p["web_name"]] = p["code"]
        full = f"{p.get('first_name','')} {p.get('second_name','')}".strip()
        lookup[full] = p["code"]
        lookup[p.get("second_name","").lower()] = p["code"]
    return lookup

def _fuzzy_code(name, lookup):
    if name in lookup:
        return lookup[name]
    token = name.split(".")[-1].strip().lower()
    for key, val in lookup.items():
        if token and len(token) > 2 and token in key.lower():
            return val
    return None

def get_captain_picks(proj_df, eo_df, gw, master_df_full, bootstrap):
    """
    Top 2 EV picks + 1 Differential using Solio EO file for ownership.
    Differential = highest EV player with EO < 10%.
    """
    pts_col  = f"{gw}_Pts"
    mins_col = f"{gw}_xMins"
    eo_col   = f"{gw}_eo"
    if pts_col not in proj_df.columns:
        return []

    df = proj_df.copy()
    df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
    df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")
    active = df[df[mins_col] > 45].copy()

    top2 = active.nlargest(2, pts_col).copy()
    top2["PickType"] = ["🏆 Top Pick", "🥈 2nd Pick"]

    diff_pool = active[~active["Name"].isin(top2["Name"])].copy()
    if eo_df is not None and eo_col in eo_df.columns:
        eo_map = pd.to_numeric(eo_df.set_index("Name")[eo_col], errors="coerce").to_dict()
        diff_pool["_eo"] = diff_pool["Name"].map(eo_map).fillna(1.0)
        diff_pool = diff_pool[diff_pool["_eo"] < 0.10]
    diff = diff_pool.nlargest(1, pts_col).copy()
    diff["PickType"] = ["🎯 Differential"]

    picks = pd.concat([top2, diff], ignore_index=True)
    code_lookup = _build_code_lookup(bootstrap)

    # Build EO map for display
    eo_map_display = {}
    if eo_df is not None and eo_col in eo_df.columns:
        eo_map_display = pd.to_numeric(eo_df.set_index("Name")[eo_col], errors="coerce").to_dict()

    result = []
    for _, r in picks.iterrows():
        fix, fdr = get_fdr_for_team_gw(r["Team"], gw, master_df_full)
        code  = _fuzzy_code(r["Name"], code_lookup)
        photo = player_photo_url(code) if code else None
        eo_val = eo_map_display.get(r["Name"])
        eo_pct = round(float(eo_val) * 100, 1) if eo_val is not None and not np.isnan(float(eo_val)) else None
        result.append({
            "PickType": r["PickType"],
            "Name":     r["Name"],
            "Team":     r["Team"],
            "Pos":      r["Pos"],
            "EV":       round(float(r[pts_col]), 2),
            "EO%":      eo_pct,
            "Fixture":  fix,
            "FDR":      fdr,
            "photo":    photo,
        })
    return result

def get_captain_matrix(proj_df, eo_df, gws, master_df_full, bootstrap):
    """All players within 0.5 EV of top pick per GW, with Solio EO%."""
    code_lookup = _build_code_lookup(bootstrap)
    matrix = {}

    for gw in gws:
        pts_col  = f"{gw}_Pts"
        mins_col = f"{gw}_xMins"
        eo_col   = f"{gw}_eo"
        if pts_col not in proj_df.columns:
            continue

        df = proj_df.copy()
        df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
        df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")

        active = df[df[mins_col] > 45]
        top_ev = active[pts_col].max()
        if pd.isna(top_ev):
            continue

        within = active[active[pts_col] >= top_ev - 0.5].nlargest(10, pts_col)

        eo_map = {}
        if eo_df is not None and eo_col in eo_df.columns:
            eo_map = pd.to_numeric(eo_df.set_index("Name")[eo_col], errors="coerce").to_dict()

        rows = []
        for _, r in within.iterrows():
            fix, fdr = get_fdr_for_team_gw(r["Team"], gw, master_df_full)
            code  = _fuzzy_code(r["Name"], code_lookup)
            photo = player_photo_url(code) if code else None
            bg, fg = club_style(r["Team"])
            eo_raw = eo_map.get(r["Name"], 0)
            eo_pct = round(float(eo_raw) * 100, 1) if eo_raw else 0.0
            rows.append({
                "Name":    r["Name"],
                "Team":    r["Team"],
                "EV":      round(float(r[pts_col]), 2),
                "EO%":     eo_pct,
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
eo_df   = load_eo_csv()

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
with ce: end_gw   = st.number_input("End GW:",   min_value=1, max_value=38, value=min(default_gw+9, 38))

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 FDR", "⚽ xG", "🧤 xCS", "📈 Team Ratings",
    "🎯 Captain Picks", "🏅 Captain Matrix", "📡 Live Radar",
    "🏟️ Team Stats", "👤 Player Stats",
])

# ── Shared helper: build clean HTML heatmap table ─────────────────────────────
def _heatmap_table(df_display, gw_cols, value_key, label_fn, color_fn, total_col, total_fmt):
    """
    Render a minimal dark heatmap HTML table.
    value_key: key inside cell dict ('fdr', 'xG', 'CS')
    label_fn: cell dict -> display string
    color_fn: numeric value -> (bg, fg) tuple
    """
    # Header
    th = lambda s, extra="": f'<th style="padding:8px 10px;text-align:center;color:#666;font-size:11px;font-weight:600;letter-spacing:.5px;border-bottom:1px solid #2a2a2a;{extra}">{s}</th>'
    header = (
        th("TEAM", "text-align:left;min-width:130px;color:#888") +
        th(total_col, "color:#aaa;min-width:80px")
    )
    for col in gw_cols:
        header += th(col.replace("GW","<span style='color:#555;font-size:9px'>GW</span>"), "min-width:80px")

    rows = ""
    for _, row in df_display.iterrows():
        team = row["Team"]
        bg_club, _ = club_style(team)
        # dot
        dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{bg_club};margin-right:7px;vertical-align:middle;flex-shrink:0"></span>'
        total_val = row[total_col.replace(" ","_").replace("%","CS").replace("xG","xG")]
        # handle column name
        try:
            total_num = float(total_val)
        except:
            total_num = 0
        total_display = total_fmt(total_num)

        td_team  = f'<td style="padding:8px 10px;font-weight:600;color:#e0e0e0;font-size:13px;border-bottom:1px solid #1e1e1e;white-space:nowrap">{dot}{team}</td>'
        td_total = f'<td style="padding:8px 10px;text-align:center;color:#aaa;font-size:12px;border-bottom:1px solid #1e1e1e;font-weight:600">{total_display}</td>'

        cells = td_team + td_total
        for col in gw_cols:
            cell = row[col]
            if isinstance(cell, dict):
                val = cell.get(value_key, 0)
                lbl = label_fn(cell)
                cbg, cfg = color_fn(val)
                cells += (f'<td style="padding:6px 8px;text-align:center;background:{cbg};color:{cfg};'
                          f'font-size:12px;font-weight:700;border-bottom:1px solid #161616;'
                          f'border-right:1px solid #1e1e1e">{lbl}</td>')
            else:
                cells += ('<td style="padding:6px 8px;text-align:center;background:#111;color:#c0392b;'
                          'font-size:11px;font-weight:700;border-bottom:1px solid #161616;'
                          'border-right:1px solid #1e1e1e;letter-spacing:.5px">BGW</td>')

        rows += f'<tr style="transition:background .15s" onmouseover="this.style.background=\'#1a1a1a\'" onmouseout="this.style.background=\'transparent\'">{cells}</tr>'

    return (
        '<div style="overflow-x:auto;margin-top:6px">'
        '<table style="border-collapse:collapse;width:100%;font-family:\'Inter\',sans-serif;background:#0d1117">'
        f'<thead><tr style="background:#161b22">{header}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table></div>'
    )

def _fdr_color(v):
    c = {1:"#00ff85",2:"#50c369",3:"#2e2e2e",4:"#9d66a0",5:"#6f2a74"}
    fg = {1:"#0d1117",2:"#0d1117",3:"#999999",4:"#ffffff",5:"#ffffff"}
    vr = round(v)
    return c.get(vr,"#333"), fg.get(vr,"#fff")

def _xg_color(v):
    if v >= 2.2: return "#1a7a4a","#ffffff"
    if v >= 1.8: return "#2da65c","#ffffff"
    if v >= 1.4: return "#50c369","#0d1117"
    if v >= 1.0: return "#2e4a38","#a8d8a8"
    return "#1c1c1c","#555555"

def _xcs_color(v):
    # v is 0-1 probability
    if v >= 0.50: return "#00ff85","#0d1117"
    if v >= 0.40: return "#50c369","#0d1117"
    if v >= 0.28: return "#2e4a38","#a8d8a8"
    if v >= 0.18: return "#2e2e2e","#888888"
    if v >= 0.10: return "#4a2060","#cc99ee"
    return "#6f2a74","#ffffff"

# ── Tab 1: FDR ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Fixture Difficulty</span>'
        '<span style="font-size:12px;color:#555">sorted easiest → hardest</span></div>',
        unsafe_allow_html=True
    )
    # Legend
    legend_items = [
        ("#00ff85","#0d1117","1 Easy"), ("#50c369","#0d1117","2"),
        ("#2e2e2e","#999","3 Neutral"),
        ("#9d66a0","#fff","4"), ("#6f2a74","#fff","5 Hard"),
        ("#111","#c0392b","BGW"),
    ]
    legend_html = '<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">'
    for bg, fg, lbl in legend_items:
        legend_html += (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;'
                        f'font-size:11px;font-weight:700">{lbl}</span>')
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    df_d = master_df.sort_values("Total Difficulty").reset_index().rename(columns={"index":"Team"})
    df_d = df_d[["Team","Total Difficulty"] + gw_columns].copy()
    df_d.rename(columns={"Total Difficulty":"Total_Difficulty"}, inplace=True)

    html = _heatmap_table(
        df_d, gw_columns,
        value_key="fdr",
        label_fn=lambda c: c.get("display","?"),
        color_fn=_fdr_color,
        total_col="Total_Difficulty",
        total_fmt=lambda v: f"{v:.0f}",
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Tab 2: xG ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Expected Goals (xG)</span>'
        '<span style="font-size:12px;color:#555">higher = better for attackers · sorted by total</span></div>',
        unsafe_allow_html=True
    )
    legend_items_xg = [
        ("#1a7a4a","#fff","≥ 2.2"), ("#2da65c","#fff","≥ 1.8"),
        ("#50c369","#0d1117","≥ 1.4"), ("#2e4a38","#a8d8a8","≥ 1.0"),
        ("#1c1c1c","#555","< 1.0"), ("#111","#c0392b","BGW"),
    ]
    lh = '<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">'
    for bg, fg, lbl in legend_items_xg:
        lh += f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700">{lbl}</span>'
    lh += '</div>'
    st.markdown(lh, unsafe_allow_html=True)

    df_d = master_df.sort_values("Total xG", ascending=False).reset_index().rename(columns={"index":"Team"})
    df_d = df_d[["Team","Total xG"] + gw_columns].copy()
    df_d.rename(columns={"Total xG":"Total_xG"}, inplace=True)

    html = _heatmap_table(
        df_d, gw_columns,
        value_key="xG",
        label_fn=lambda c: f"{c.get('xG',0):.2f}",
        color_fn=_xg_color,
        total_col="Total_xG",
        total_fmt=lambda v: f"{v:.2f}",
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Tab 3: xCS ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Clean Sheet % (xCS)</span>'
        '<span style="font-size:12px;color:#555">higher = better for defenders & keepers · sorted by total</span></div>',
        unsafe_allow_html=True
    )
    legend_items_cs = [
        ("#00ff85","#0d1117","≥ 50%"), ("#50c369","#0d1117","≥ 40%"),
        ("#2e4a38","#a8d8a8","≥ 28%"), ("#2e2e2e","#888","≥ 18%"),
        ("#4a2060","#cc99ee","≥ 10%"), ("#6f2a74","#fff","< 10%"),
        ("#111","#c0392b","BGW"),
    ]
    lh = '<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">'
    for bg, fg, lbl in legend_items_cs:
        lh += f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700">{lbl}</span>'
    lh += '</div>'
    st.markdown(lh, unsafe_allow_html=True)

    df_d = master_df.sort_values("xCS", ascending=False).reset_index().rename(columns={"index":"Team"})
    df_d = df_d[["Team","xCS"] + gw_columns].copy()
    df_d.rename(columns={"xCS":"Total_xCS"}, inplace=True)

    html = _heatmap_table(
        df_d, gw_columns,
        value_key="CS",
        label_fn=lambda c: f"{c.get('CS',0)*100:.0f}%",
        color_fn=_xcs_color,
        total_col="Total_xCS",
        total_fmt=lambda v: f"{v*100:.0f}%",
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Tab 4: Team Ratings ───────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Team Strength Map</span>'
        '<span style="font-size:12px;color:#555">right = better defence · up = better attack</span></div>',
        unsafe_allow_html=True
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

        pad_x = (plot_df["Def Score"].max() - plot_df["Def Score"].min()) * 0.14
        pad_y = (plot_df["Off Score"].max() - plot_df["Off Score"].min()) * 0.20
        x_min = plot_df["Def Score"].min() - pad_x
        x_max = plot_df["Def Score"].max() + pad_x
        y_min = plot_df["Off Score"].min() - pad_y
        y_max = plot_df["Off Score"].max() + pad_y
        x_avg = plot_df["Def Score"].mean()
        y_avg = plot_df["Off Score"].mean()
        y_range = y_max - y_min

        fig = go.Figure()

        # Quadrant lines
        for shape_args in [
            dict(x0=x_avg, x1=x_avg, y0=y_min, y1=y_max),
            dict(x0=x_min, x1=x_max, y0=y_avg, y1=y_avg),
        ]:
            fig.add_shape(type="line", **shape_args,
                          line=dict(color="rgba(255,255,255,0.08)", width=1, dash="dot"))

        # ── Circle badges ──────────────────────────────────────────────────────
        # Two layers: large circle (club colour) + abbreviation text centred
        for _, row in plot_df.iterrows():
            bg   = row["club_bg"]
            fg   = row["club_text"]
            abbr = row["Abbr"]
            team = row["Team"]
            x    = row["Def Score"]
            y    = row["Off Score"]

            # Circle marker with text
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=42,
                    color=bg,
                    symbol="circle",
                    line=dict(color="rgba(255,255,255,0.15)", width=1.5),
                ),
                text=[f"<b>{abbr}</b>"],
                textfont=dict(color=fg, size=10, family="'Arial Black', Arial, sans-serif"),
                textposition="middle center",
                name=team,
                showlegend=False,
                hovertemplate=(
                    f"<b>{team}</b><br>"
                    f"⚔️ Attack: {y:.2f}<br>"
                    f"🛡️ Defence: {x:.2f}<extra></extra>"
                ),
            ))

        # Corner watermarks
        ann = dict(showarrow=False, font=dict(size=9, color="rgba(255,255,255,0.12)"))
        fig.add_annotation(x=x_max, y=y_max, text="ELITE",           xanchor="right", yanchor="top",    **ann)
        fig.add_annotation(x=x_min, y=y_max, text="ATTACK",          xanchor="left",  yanchor="top",    **ann)
        fig.add_annotation(x=x_max, y=y_min, text="SOLID DEFENCE",   xanchor="right", yanchor="bottom", **ann)
        fig.add_annotation(x=x_min, y=y_min, text="STRUGGLING",      xanchor="left",  yanchor="bottom", **ann)

        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#cccccc", family="sans-serif", size=12),
            xaxis=dict(
                title=dict(text="Better Defence →", font=dict(size=12, color="#555")),
                range=[x_min, x_max], showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=10, color="#444"),
            ),
            yaxis=dict(
                title=dict(text="Better Attack ↑", font=dict(size=12, color="#555")),
                range=[y_min, y_max], showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=10, color="#444"),
            ),
            margin=dict(l=55, r=25, t=15, b=55),
            height=560,
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Ranking table ──────────────────────────────────────────────────────
        rank_df = plot_df[["Team","Abbr","Off Score","Def Score"]].copy()
        off_max = rank_df["Off Score"].max()
        def_max = rank_df["Def Score"].max()
        rank_df["Overall"] = ((rank_df["Off Score"]/off_max + rank_df["Def Score"]/def_max)/2*100).round(1)
        rank_df = rank_df.sort_values("Overall", ascending=False).reset_index(drop=True)

        rows_html = ""
        for i, row in rank_df.iterrows():
            bg, fg = club_style(row["Team"])
            medal  = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}"
            dot = (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                   f'background:{bg};margin-right:8px;vertical-align:middle"></span>')
            rows_html += (
                f'<tr onmouseover="this.style.background=\'#161b22\'" onmouseout="this.style.background=\'transparent\'">'
                f'<td style="width:4px;background:{bg};padding:0"></td>'
                f'<td style="padding:6px 10px;text-align:center;color:#555;font-size:13px">{medal}</td>'
                f'<td style="padding:6px 10px;font-weight:700;color:#e0e0e0;font-size:13px">{dot}{row["Team"]}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:#50c369;font-size:13px;font-weight:600">{row["Off Score"]:.2f}</td>'
                f'<td style="padding:6px 10px;text-align:center;color:#6CABDD;font-size:13px;font-weight:600">{row["Def Score"]:.2f}</td>'
                f'<td style="padding:6px 10px;text-align:center;font-weight:700;color:#f4a261;font-size:13px">{row["Overall"]:.1f}</td>'
                f'</tr>'
            )
        st.markdown(
            '<div style="overflow-x:auto;border-radius:6px;border:1px solid #1e1e1e;margin-top:12px">'
            '<table style="border-collapse:collapse;width:100%;font-family:\'Inter\',sans-serif;background:#0d1117">'
            '<thead style="background:#161b22"><tr>'
            '<th style="width:4px;padding:0"></th>'
            '<th style="padding:6px 10px;color:#444;font-size:11px;font-weight:600;text-align:center">#</th>'
            '<th style="padding:6px 10px;color:#666;font-size:11px;font-weight:600;text-align:left">TEAM</th>'
            '<th style="padding:6px 10px;color:#50c369;font-size:11px;font-weight:600;text-align:center">⚔️ ATTACK</th>'
            '<th style="padding:6px 10px;color:#6CABDD;font-size:11px;font-weight:600;text-align:center">🛡️ DEFENCE</th>'
            '<th style="padding:6px 10px;color:#f4a261;font-size:11px;font-weight:600;text-align:center">OVERALL %</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True
        )

# ── Solio branding helper ─────────────────────────────────────────────────────
def _solio_credit_bar():
    import base64
    logo_b64 = ""
    try:
        with open("Solio_Logo_Neg_RGB.png", "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        img_tag = (f'<img src="data:image/png;base64,{logo_b64}" '
                   f'style="height:22px;vertical-align:middle;margin-right:10px;opacity:0.9">')
    except:
        img_tag = '<span style="font-weight:800;color:#fff;font-size:13px;margin-right:10px;letter-spacing:1px">SOLIO</span>'
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;' 
        'background:#111;border:1px solid #222;border-radius:6px;' 
        'padding:8px 14px;margin-bottom:14px">' 
        '<div style="display:flex;align-items:center">' 
        f'{img_tag}'
        '<span style="color:#555;font-size:11px">Projections &amp; EO data powered by Solio Analytics</span>' 
        '</div>' 
        '<a href="https://solioanalytics.com/" target="_blank" ' 
        'style="background:#fff;color:#000;font-size:11px;font-weight:700;' 
        'padding:4px 12px;border-radius:4px;text-decoration:none;' 
        'letter-spacing:.5px;white-space:nowrap">SUBSCRIBE ↗</a>' 
        '</div>'
    )

SOLIO_NO_FILE_MSG = (
    "\ud83d\udcc2 **{filename} not found.**\n\n"
    "Download from [Solio Analytics](https://www.solioanalytics.com) "
    "and place it in your app folder alongside `app.py`."
)

# ── Tab 5: Captain Picks ──────────────────────────────────────────────────────
with tab5:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">' 
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">🎯 Captain Picks</span>' 
        '<span style="font-size:12px;color:#444">Top 2 EV + 1 Differential per gameweek</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(_solio_credit_bar(), unsafe_allow_html=True)

    if proj_df is None:
        st.info(SOLIO_NO_FILE_MSG.format(filename="projections.csv"))
    elif master_df_full is None:
        st.error("Could not build fixture data.")
    else:
        gw_sel = st.selectbox("Select GW:", future_gws, key="cap_gw")
        picks  = get_captain_picks(proj_df, eo_df, gw_sel, master_df_full, bootstrap)

        if not picks:
            st.warning(f"No projection data for GW{gw_sel}.")
        else:
            TYPE_STYLE = {
                "🏆 Top Pick":     ("#0a2818", "#5fffb0"),
                "🥈 2nd Pick":     ("#0a1a2b", "#5aabff"),
                "🎯 Differential": ("#2b1a00", "#ffaa33"),
            }
            card_cols = st.columns(len(picks))
            for col, p in zip(card_cols, picks):
                label_bg, label_fg = TYPE_STYLE.get(p["PickType"], ("#222", "#fff"))
                fdr_bg = FDR_BG.get(p["FDR"], "#444")
                fdr_fg = FDR_FG.get(p["FDR"], "#fff")
                club_bg, club_fg = club_style(p["Team"])
                eo_str = f"{p['EO%']}%" if p.get("EO%") is not None else "—"

                img_html = ""
                if p["photo"]:
                    img_html = (
                        f'<img src="{p["photo"]}" '
                        f'style="width:80px;height:100px;object-fit:cover;object-position:top;' 
                        f'border-radius:8px;margin-bottom:8px;' 
                        f'border:2px solid rgba(255,255,255,0.15)" '
                        f'onerror="this.style.display=\'none\'">' 
                    )

                card = (
                    f'<div style="border-radius:10px;overflow:hidden;margin:4px;' 
                    f'border:1px solid #222;box-shadow:0 2px 12px rgba(0,0,0,0.5)">' 
                    f'<div style="background:{label_bg};color:{label_fg};padding:7px 12px;' 
                    f'font-size:11px;font-weight:700;text-align:center;' 
                    f'letter-spacing:.6px;border-bottom:1px solid #222">{p["PickType"]}</div>' 
                    f'<div style="background:{club_bg};color:{club_fg};padding:16px 12px;text-align:center">' 
                    f'{img_html}' 
                    f'<div style="font-size:17px;font-weight:800;letter-spacing:.3px">{p["Name"]}</div>' 
                    f'<div style="font-size:11px;opacity:0.75;margin-top:2px">{p["Team"]} &middot; {p["Pos"]}</div>' 
                    f'<div style="font-size:26px;font-weight:800;margin:10px 0 6px">{p["EV"]} ' 
                    f'<span style="font-size:13px;font-weight:400;opacity:0.8">pts</span></div>' 
                    f'<div style="display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap">' 
                    f'<span style="background:{fdr_bg};color:{fdr_fg};border-radius:4px;' 
                    f'padding:2px 9px;font-size:12px;font-weight:700">{p["Fixture"]}</span>' 
                    f'<span style="background:rgba(0,0,0,0.25);color:{club_fg};border-radius:4px;' 
                    f'padding:2px 9px;font-size:12px;opacity:0.85">EO {eo_str}</span>' 
                    f'</div></div></div>'
                )
                col.markdown(card, unsafe_allow_html=True)

# ── Tab 6: Captain Matrix ─────────────────────────────────────────────────────
with tab6:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">' 
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">🏅 Captain Matrix</span>' 
        '<span style="font-size:12px;color:#444">within 0.5 EV of top pick per gameweek</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(_solio_credit_bar(), unsafe_allow_html=True)

    if proj_df is None:
        st.info(SOLIO_NO_FILE_MSG.format(filename="projections.csv"))
    elif master_df_full is None:
        st.error("Fixture data unavailable.")
    else:
        matrix_gws = st.multiselect(
            "Show GWs:", future_gws,
            default=future_gws,
            key="matrix_gws"
        )
        if matrix_gws:
            matrix = get_captain_matrix(proj_df, eo_df, matrix_gws, master_df_full, bootstrap)
            col_w  = max(140, min(200, 1100 // len(matrix_gws)))

            header = "".join(
                f'<th style="background:#0d1117;color:#555;padding:9px 6px;' 
                f'border-bottom:2px solid #222;text-align:center;min-width:{col_w}px;' 
                f'font-size:11px;font-weight:700;letter-spacing:.6px;' 
                f'font-family:sans-serif">GW{gw}</th>'
                for gw in matrix_gws
            )

            max_rows = max((len(matrix.get(gw, [])) for gw in matrix_gws), default=0)
            body = ""
            for ri in range(max_rows):
                row_html = ""
                for gw in matrix_gws:
                    rows_gw = matrix.get(gw, [])
                    if ri < len(rows_gw):
                        r      = rows_gw[ri]
                        bg     = r["club_bg"]
                        fg     = r["club_fg"]
                        fdr_b  = FDR_BG.get(r["FDR"], "#444")
                        fdr_f  = FDR_FG.get(r["FDR"], "#fff")
                        is_top = ri == 0
                        top_border = "border-top:2px solid rgba(255,255,255,0.2);" if is_top else ""
                        name_style = "font-weight:800;font-size:13px" if is_top else "font-weight:600;font-size:12px"
                        eo_str = f"{r['EO%']}%" if r.get("EO%") is not None else "—"

                        img_tag = ""
                        if r["photo"]:
                            img_tag = (
                                f'<img src="{r["photo"]}" '
                                f'style="width:36px;height:46px;object-fit:cover;object-position:top;' 
                                f'border-radius:4px;flex-shrink:0;' 
                                f'border:1px solid rgba(255,255,255,0.2)" '
                                f'onerror="this.style.display=\'none\'">' 
                            )

                        row_html += (
                            f'<td style="padding:7px 8px;border:1px solid #1a1a1a;' 
                            f'{top_border}background:{bg};vertical-align:middle">' 
                            f'<div style="display:flex;align-items:center;gap:8px">' 
                            f'{img_tag}' 
                            f'<div style="min-width:0">' 
                            f'<div style="color:{fg};{name_style};' 
                            f'font-family:sans-serif;white-space:nowrap;overflow:hidden;' 
                            f'text-overflow:ellipsis">{r["Name"]}</div>' 
                            f'<div style="display:flex;align-items:center;gap:4px;margin-top:3px;flex-wrap:wrap">' 
                            f'<span style="background:{fdr_b};color:{fdr_f};border-radius:3px;' 
                            f'padding:1px 6px;font-size:11px;font-weight:700;white-space:nowrap">{r["Fixture"]}</span>' 
                            f'<span style="color:{fg};font-weight:700;font-size:13px">{r["EV"]}</span>' 
                            f'<span style="color:{fg};font-size:11px;opacity:0.65">EO {eo_str}</span>' 
                            f'</div></div></div></td>'
                        )
                    else:
                        row_html += f'<td style="border:1px solid #1a1a1a;background:#0a0a0a;min-width:{col_w}px"></td>'
                body += f"<tr>{row_html}</tr>"

            html = (
                f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #1e1e1e;margin-top:4px">' 
                f'<table style="border-collapse:collapse;width:100%;background:#0d1117">' 
                f'<thead><tr style="background:#0d1117">{header}</tr></thead>' 
                f'<tbody>{body}</tbody>' 
                f'</table></div>'
            )
            st.markdown(html, unsafe_allow_html=True)
            st.caption("Top row = highest EV · Colour = club · FDR badge = custom ratings · EO = Solio expected ownership")

with tab7:
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




# =============================================================================
# HELPERS — Team Stats & Player Stats  (100% FPL API, no external files)
# =============================================================================

def build_team_stats_df(bootstrap, raw_fixtures):
    """
    Derive W/D/L/MP/GF/GA/Pts from completed fixtures.
    xG/xGC from elements aggregation (most accurate source).
    CS from main GKP per team.
    """
    teams_df = pd.DataFrame(bootstrap["teams"])
    elements = pd.DataFrame(bootstrap["elements"])
    id2name  = dict(zip(teams_df["id"], teams_df["name"]))

    # ── Derive table stats from completed fixtures ────────────────────────────
    records = {tid: {"MP":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0}
               for tid in teams_df["id"]}

    for f in raw_fixtures:
        if not f.get("finished", False): continue
        hs = f.get("team_h_score")
        as_ = f.get("team_a_score")
        if hs is None or as_ is None: continue
        hs, as_ = int(hs), int(as_)
        h, a = f["team_h"], f["team_a"]
        for tid in (h, a):
            if tid not in records: records[tid] = {"MP":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0}
        records[h]["MP"] += 1; records[a]["MP"] += 1
        records[h]["GF"] += hs; records[h]["GA"] += as_
        records[a]["GF"] += as_; records[a]["GA"] += hs
        if hs > as_:
            records[h]["W"] += 1; records[h]["Pts"] += 3
            records[a]["L"] += 1
        elif hs < as_:
            records[a]["W"] += 1; records[a]["Pts"] += 3
            records[h]["L"] += 1
        else:
            records[h]["D"] += 1; records[h]["Pts"] += 1
            records[a]["D"] += 1; records[a]["Pts"] += 1

    # ── xG for team = sum of player xG ───────────────────────────────────────
    for c in ["expected_goals","expected_goals_conceded","expected_goal_involvements"]:
        if c in elements.columns:
            elements[c] = pd.to_numeric(elements[c], errors="coerce").fillna(0)

    xg_by_team  = elements.groupby("team")["expected_goals"].sum() \
                  if "expected_goals" in elements.columns else {}

    # ── xGC: from top GKP per team (most accurate — represents team's defensive xGC) ──
    rows = []
    for _, t in teams_df.iterrows():
        name = TEAM_NAME_MAP.get(t["name"], t["name"])
        if name not in PREMIER_LEAGUE_TEAMS:
            continue
        tid = t["id"]
        rec = records.get(tid, {})
        gf  = rec.get("GF", 0)
        ga  = rec.get("GA", 0)

        # xG for team
        xg = round(float(xg_by_team.get(tid, 0)), 1) if hasattr(xg_by_team, "get") else 0.0

        # xGC: expected_goals_conceded from main GKP
        gkps = elements[(elements["team"] == tid) & (elements["element_type"] == 1)].copy()
        if not gkps.empty and "expected_goals_conceded" in gkps.columns:
            gkps["minutes"] = pd.to_numeric(gkps.get("minutes", 0), errors="coerce").fillna(0)
            gkps["expected_goals_conceded"] = pd.to_numeric(
                gkps["expected_goals_conceded"], errors="coerce").fillna(0)
            xgc = round(float(gkps.sort_values("minutes", ascending=False)
                               .iloc[0]["expected_goals_conceded"]), 1)
        else:
            xgc = None

        # CS: clean sheets from main GKP
        if not gkps.empty and "clean_sheets" in gkps.columns:
            gkps["clean_sheets"] = pd.to_numeric(gkps["clean_sheets"], errors="coerce").fillna(0)
            cs = int(gkps.sort_values("minutes", ascending=False).iloc[0]["clean_sheets"])
        else:
            cs = 0

        gd     = gf - ga
        xgdiff = round(xg - xgc, 1) if xgc is not None else None

        rows.append({
            "Team":   name,
            "MP":     rec.get("MP", 0),
            "W":      rec.get("W",  0),
            "D":      rec.get("D",  0),
            "L":      rec.get("L",  0),
            "GF":     gf,
            "GA":     ga,
            "GD":     gd,
            "CS":     cs,
            "xG":     xg,
            "xGC":    xgc if xgc is not None else 0.0,
            "xGDiff": xgdiff if xgdiff is not None else 0.0,
            "Pts":    rec.get("Pts", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Pts", ascending=False).reset_index(drop=True)
        df.insert(0, "Rk", range(1, len(df)+1))
    return df


def build_player_stats_df(bootstrap):
    """Full player analytics from FPL API elements — raw totals."""
    elements = pd.DataFrame(bootstrap["elements"])
    teams_df = pd.DataFrame(bootstrap["teams"])
    id2name  = dict(zip(teams_df["id"], teams_df["name"]))
    POS      = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}

    # Coerce all numeric strings
    num_cols = ["minutes","starts","expected_goals","expected_assists",
                "expected_goal_involvements","expected_goals_conceded",
                "penalties_scored","now_cost","selected_by_percent",
                "total_points","clearances_blocks_interceptions","tackles","recoveries"]
    for c in num_cols:
        if c in elements.columns:
            elements[c] = pd.to_numeric(elements[c], errors="coerce").fillna(0)

    rows = []
    for _, p in elements.iterrows():
        mins     = float(p.get("minutes", 0))
        starts   = float(p.get("starts", 0))
        nineties = mins / 90 if mins >= 45 else 0

        team = TEAM_NAME_MAP.get(id2name.get(p.get("team"), ""), "")
        pos  = POS.get(p.get("element_type"), "?")

        xg  = float(p.get("expected_goals", 0))
        xa  = float(p.get("expected_assists", 0))
        xgi = float(p.get("expected_goal_involvements", 0))
        xgc = float(p.get("expected_goals_conceded", 0))

        pens  = float(p.get("penalties_scored", 0))
        npxg  = max(0.0, round(xg - pens * 0.76, 2))
        npxgi = max(0.0, round(npxg + xa, 2))

        def_raw    = (float(p.get("clearances_blocks_interceptions", 0)) +
                      float(p.get("tackles", 0)) +
                      float(p.get("recoveries", 0)))
        defcon_90  = round(def_raw / nineties, 2) if nineties >= 0.5 else None

        price       = round(float(p.get("now_cost", 0)) / 10, 1)
        own         = float(p.get("selected_by_percent", 0))
        total_pts   = float(p.get("total_points", 0))
        ppm         = round(total_pts / price, 1) if price > 0 else None
        min_start   = round(mins / starts, 0) if starts > 0 else None

        rows.append({
            "Player":    p.get("web_name", ""),
            "Team":      team,
            "Pos":       pos,
            "Price":     price,
            "Starts":    int(starts),
            "Mins":      int(mins),
            "Min/Start": int(min_start) if min_start else None,
            # raw totals
            "xG":        round(xg, 2),
            "xA":        round(xa, 2),
            "xGI":       round(xgi, 2),
            "NpxG":      npxg,
            "NpxGI":     npxgi,
            "xGC":       round(xgc, 2),
            "Defcon":    round(def_raw, 1),   # raw total (for /90 toggle)
            "Defcon/90": defcon_90,
            "PPM":       ppm,
            "Own%":      own,
            # hidden 90s divisor for toggle
            "_90s":      round(nineties, 2),
        })

    df = pd.DataFrame(rows)
    df = df[df["Mins"] > 0].copy().reset_index(drop=True)
    return df


def _per90(df, cols_to_divide):
    """Return a copy of df with specified cols divided by _90s."""
    out = df.copy()
    for c in cols_to_divide:
        if c in out.columns and "_90s" in out.columns:
            out[c] = (out[c] / out["_90s"].replace(0, np.nan)).round(2)
    return out


# ── Colour helpers ────────────────────────────────────────────────────────────
# Simple 3-tier palette: low / mid / high — easy on the eyes, high contrast text

def _tier_color(val, low, high, col_type="positive"):
    """
    Returns (background, text_color) using a 3-tier system:
      positive: low=grey, mid=teal, high=green
      negative: low=grey, mid=orange, high=red   (used for GA, xGC)
      diff:     positive=green, zero=grey, negative=red
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#1a1a1a", "#444444"

    fv = float(val)
    # normalise 0→1
    span = high - low
    if span == 0:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (fv - low) / span))

    if col_type == "positive":
        # dark grey → teal → bright green
        if t < 0.33:
            return "#1e1e1e", "#666666"
        elif t < 0.66:
            return "#0d2b2b", "#4ecdc4"
        else:
            return "#0a2818", "#5fffb0"
    elif col_type == "negative":
        # green (low GA is good) → amber → red
        if t < 0.33:
            return "#0a2818", "#5fffb0"
        elif t < 0.66:
            return "#2b1a00", "#ffaa33"
        else:
            return "#2b0a0a", "#ff6060"
    elif col_type == "diff":
        if fv > 0.5:
            return "#0a2818", "#5fffb0"
        elif fv < -0.5:
            return "#2b0a0a", "#ff6060"
        else:
            return "#1e1e1e", "#888888"
    elif col_type == "blue":
        # for CS, Defcon — blue scale
        if t < 0.33:
            return "#1e1e1e", "#555555"
        elif t < 0.66:
            return "#0a1a2b", "#5aabff"
        else:
            return "#051022", "#99ccff"
    elif col_type == "gold":
        if t < 0.33:
            return "#1e1e1e", "#555555"
        elif t < 0.66:
            return "#221a00", "#ccaa33"
        else:
            return "#2b1f00", "#ffd966"
    return "#1e1e1e", "#888888"


# Pre-compute column ranges for consistent colouring across sort orders
_TEAM_RANGES = {
    "GF":     (15, 70, "positive"),
    "GA":     (15, 65, "negative"),
    "GD":     (-40, 50, "diff"),
    "CS":     (0,  18, "blue"),
    "xG":     (15, 65, "positive"),
    "xGC":    (15, 65, "negative"),
    "xGDiff": (-35, 35, "diff"),
    "Pts":    (0,  90, "blue"),
    "W":      (0,  30, "positive"),
    "L":      (0,  25, "negative"),
    "D":      (0,  15, "positive"),
}

_PLAYER_RANGES = {
    "xG":        (0, 20,  "positive"),
    "xA":        (0, 12,  "positive"),
    "xGI":       (0, 25,  "positive"),
    "NpxG":      (0, 18,  "positive"),
    "NpxGI":     (0, 22,  "positive"),
    "xGC":       (0, 50,  "negative"),
    "Defcon/90": (0, 10,  "blue"),
    "Defcon":    (0, 200, "blue"),
    "PPM":       (0, 20,  "gold"),
    "Own%":      (0, 70,  "gold"),
    "Price":     (3.5, 15,"gold"),
}


def _stat_cell(val, col, ranges):
    """Styled <td> with 3-tier colour, clean white text, minimal look."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ('<td style="padding:6px 10px;text-align:center;color:#333;'
                'border-bottom:1px solid #161616">—</td>')

    fv = float(val)

    if col in ranges:
        low, high, col_type = ranges[col]
        bg, tc = _tier_color(fv, low, high, col_type)
    else:
        bg, tc = "#1e1e1e", "#888888"

    # Format display value
    if col in ("xG","xA","xGI","NpxG","NpxGI","xGC","xGDiff","xG/90","xA/90",
               "xGI/90","NpxG/90","NpxGI/90","xGC/90","Defcon/90","PPM","Price","Own%"):
        disp = f"{fv:.2f}"
    elif col in ("Pts","GF","GA","GD","W","D","L","MP","CS","Starts","Mins","Min/Start"):
        disp = f"{int(round(fv))}"
    else:
        disp = f"{fv:.2f}"

    return (f'<td style="padding:6px 10px;text-align:center;background:{bg};color:{tc};'
            f'font-size:12px;font-weight:600;border-bottom:1px solid #161616">{disp}</td>')


def _build_stats_html(df, ranges):
    """Render minimal dark stats HTML table."""
    cols = [c for c in df.columns if not c.startswith("_")]

    header = ""
    for c in cols:
        align = "left" if c in ("Player","Team") else "center"
        header += (f'<th style="padding:7px 10px;text-align:{align};color:#555;font-size:10px;'
                   f'font-weight:700;letter-spacing:.7px;border-bottom:2px solid #222;'
                   f'white-space:nowrap;background:#0d1117">{c.upper()}</th>')

    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for c in cols:
            val = row[c]
            if c == "Rk":
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#3a3a3a;'
                          f'font-size:11px;border-bottom:1px solid #161616">{val}</td>')
            elif c == "Team":
                bg, fg = club_style(str(val))
                dot = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                       f'background:{bg};margin-right:7px;vertical-align:middle"></span>')
                cells += (f'<td style="padding:6px 10px;font-weight:700;color:#e0e0e0;font-size:12px;'
                          f'border-bottom:1px solid #161616;white-space:nowrap">{dot}{val}</td>')
            elif c == "Player":
                cells += (f'<td style="padding:6px 10px;font-weight:600;color:#d0d0d0;font-size:12px;'
                          f'border-bottom:1px solid #161616;white-space:nowrap">{val}</td>')
            elif c == "Pos":
                pos_c = {"GKP":"#f4a261","DEF":"#5aabff","MID":"#5fffb0","FWD":"#ff6060"}
                pc = pos_c.get(str(val), "#888")
                cells += (f'<td style="padding:4px 10px;text-align:center;border-bottom:1px solid #161616">'
                          f'<span style="background:{pc}18;color:{pc};border-radius:3px;'
                          f'padding:2px 7px;font-size:11px;font-weight:700">{val}</span></td>')
            elif c == "W":
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#5fffb0;font-size:12px;'
                          f'font-weight:700;border-bottom:1px solid #161616">{val}</td>')
            elif c == "D":
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#888;font-size:12px;'
                          f'font-weight:700;border-bottom:1px solid #161616">{val}</td>')
            elif c == "L":
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#ff6060;font-size:12px;'
                          f'font-weight:700;border-bottom:1px solid #161616">{val}</td>')
            elif c in ("MP","Starts","Mins","Min/Start"):
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#555;font-size:12px;'
                          f'border-bottom:1px solid #161616">{val if val is not None else "—"}</td>')
            elif c == "Price":
                cells += (f'<td style="padding:6px 10px;text-align:center;color:#f4a261;font-size:12px;'
                          f'font-weight:600;border-bottom:1px solid #161616">£{val:.1f}m</td>')
            else:
                try:
                    cells += _stat_cell(pd.to_numeric(val, errors="coerce"), c, ranges)
                except:
                    cells += (f'<td style="padding:6px 10px;text-align:center;color:#555;'
                              f'font-size:12px;border-bottom:1px solid #161616">{val}</td>')

        rows_html += (f'<tr onmouseover="this.style.background=\'#111\'" '
                      f'onmouseout="this.style.background=\'transparent\'">{cells}</tr>')

    return (
        '<div style="overflow-x:auto;border-radius:6px;border:1px solid #1e1e1e;margin-top:8px">'
        '<table style="border-collapse:collapse;width:100%;font-family:\'Inter\',sans-serif;background:#0d1117">'
        f'<thead><tr style="background:#0d1117">{header}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )


# =============================================================================
# TAB 8 — Team Stats
# =============================================================================
with tab8:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Team Stats</span>'
        '<span style="font-size:12px;color:#444">live from FPL API · derived from completed fixtures</span></div>',
        unsafe_allow_html=True
    )

    if not live_ok:
        st.warning("⚠️ Enable Live FPL Data in the sidebar.")
    else:
        ts_df = build_team_stats_df(bootstrap, raw_fixtures)

        if ts_df.empty:
            st.warning("Could not build team stats.")
        else:
            c1, _ = st.columns([1, 3])
            with c1:
                sort_opts_ts = [c for c in ["Pts","GF","xG","xGDiff","GD","W","CS","xGC","GA"] if c in ts_df.columns]
                sort_ts = st.selectbox("Sort by:", sort_opts_ts, key="ts_sort")

            asc_ts = sort_ts in ("GA","xGC","L")
            d_ts = ts_df.sort_values(sort_ts, ascending=asc_ts).reset_index(drop=True)
            d_ts["Rk"] = range(1, len(d_ts)+1)
            ordered_ts = ["Rk","Team","MP","W","D","L","GF","GA","GD","CS","xG","xGC","xGDiff","Pts"]
            d_ts = d_ts[[c for c in ordered_ts if c in d_ts.columns]]

            st.markdown(_build_stats_html(d_ts, _TEAM_RANGES), unsafe_allow_html=True)

            # xG vs GF scatter
            if "xG" in ts_df.columns and "GF" in ts_df.columns:
                st.markdown("---")
                st.markdown(
                    '<span style="font-size:14px;font-weight:600;color:#aaa">'
                    'xG vs Goals Scored — Over/Under Performers</span>',
                    unsafe_allow_html=True
                )
                st.caption("Above the line = scoring more than expected (clinical). Below = wasting chances.")
                valid = ts_df.dropna(subset=["xG","GF"])
                if not valid.empty:
                    mx = max(valid["xG"].max(), valid["GF"].max()) * 1.08
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=[0, mx], y=[0, mx], mode="lines", showlegend=False,
                        line=dict(color="rgba(255,255,255,0.08)", dash="dot", width=1),
                        hoverinfo="skip"
                    ))
                    for _, r in valid.iterrows():
                        bg, fg = club_style(r["Team"])
                        abbr   = TEAM_ABBREVIATIONS.get(r["Team"], r["Team"][:3].upper())
                        diff   = r["GF"] - r["xG"]
                        fig_ts.add_trace(go.Scatter(
                            x=[r["xG"]], y=[r["GF"]],
                            mode="markers+text",
                            marker=dict(size=38, color=bg, symbol="circle",
                                        line=dict(color="rgba(255,255,255,0.12)", width=1)),
                            text=[f"<b>{abbr}</b>"],
                            textfont=dict(color=fg, size=9, family="Arial Black, sans-serif"),
                            textposition="middle center",
                            showlegend=False,
                            hovertemplate=(f"<b>{r['Team']}</b><br>xG: {r['xG']:.1f}"
                                           f"<br>GF: {int(r['GF'])}"
                                           f"<br>Diff: {diff:+.1f}<extra></extra>")
                        ))
                    fig_ts.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        xaxis=dict(title="xG", showgrid=False, zeroline=False,
                                   tickfont=dict(color="#444"), title_font=dict(color="#555", size=11)),
                        yaxis=dict(title="Goals Scored", showgrid=False, zeroline=False,
                                   tickfont=dict(color="#444"), title_font=dict(color="#555", size=11)),
                        margin=dict(l=50, r=20, t=10, b=50), height=420, hovermode="closest"
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)


# =============================================================================
# TAB 9 — Player Stats
# =============================================================================
with tab9:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
        '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Player Stats</span>'
        '<span style="font-size:12px;color:#444">live from FPL API · auto-updated</span></div>',
        unsafe_allow_html=True
    )

    if not live_ok:
        st.warning("⚠️ Enable Live FPL Data in the sidebar.")
    else:
        ps_df = build_player_stats_df(bootstrap)

        # ── Controls row ───────────────────────────────────────────────────────
        cc1, cc2, cc3, cc4, cc5 = st.columns([1, 1, 1, 1, 1])
        with cc1:
            pos_f = st.selectbox("Position:", ["All","GKP","DEF","MID","FWD"], key="ps_pos")
        with cc2:
            team_opts = ["All"] + sorted(ps_df["Team"].dropna().unique().tolist())
            team_f    = st.selectbox("Team:", team_opts, key="ps_team")
        with cc3:
            min_mins  = st.number_input("Min mins:", 0, 3420, 450, 90, key="ps_mins")
        with cc4:
            sort_opts_ps = [c for c in ["xGI","xG","xA","NpxGI","NpxG","xGC","Defcon/90","PPM","Own%","Mins","Price"]
                            if c in ps_df.columns]
            sort_ps = st.selectbox("Sort by:", sort_opts_ps, key="ps_sort")
        with cc5:
            per90 = st.toggle("Per 90 mins", value=False, key="ps_per90")

        # ── Filter ─────────────────────────────────────────────────────────────
        filt = ps_df[ps_df["Mins"] >= min_mins].copy()
        if pos_f  != "All": filt = filt[filt["Pos"]  == pos_f]
        if team_f != "All": filt = filt[filt["Team"] == team_f]

        # ── Per-90 toggle ──────────────────────────────────────────────────────
        PER90_COLS = ["xG","xA","xGI","NpxG","NpxGI","xGC","Defcon"]
        display_ps = filt.copy()

        if per90:
            for c in PER90_COLS:
                if c in display_ps.columns and "_90s" in display_ps.columns:
                    display_ps[c] = (display_ps[c] / display_ps["_90s"].replace(0, np.nan)).round(2)
            # Rename Defcon column header
            if "Defcon" in display_ps.columns:
                display_ps.rename(columns={"Defcon":"Defcon/90"}, inplace=True)
            per90_label = " /90"
        else:
            if "Defcon" in display_ps.columns:
                display_ps.rename(columns={"Defcon":"Defcon"}, inplace=True)
            per90_label = ""

        # Determine actual sort column (may have been renamed)
        sort_col_actual = sort_ps
        if per90 and sort_ps == "Defcon/90" and "Defcon/90" in display_ps.columns:
            sort_col_actual = "Defcon/90"

        if sort_col_actual not in display_ps.columns:
            sort_col_actual = "xGI" if "xGI" in display_ps.columns else display_ps.columns[4]

        display_ps = display_ps.sort_values(sort_col_actual, ascending=False).reset_index(drop=True)
        display_ps.insert(0, "Rk", range(1, len(display_ps)+1))

        # Column display order
        base_cols   = ["Rk","Player","Team","Pos","Price","Starts","Mins","Min/Start"]
        stat_cols   = ["xG","xA","xGI","NpxG","NpxGI","xGC"]
        if per90:
            stat_cols = [c for c in stat_cols if c in display_ps.columns]
            def_col   = ["Defcon/90"] if "Defcon/90" in display_ps.columns else []
        else:
            stat_cols = [c for c in stat_cols if c in display_ps.columns]
            def_col   = ["Defcon"] if "Defcon" in display_ps.columns else []

        final_cols  = base_cols + stat_cols + def_col + ["PPM","Own%"]
        final_cols  = [c for c in final_cols if c in display_ps.columns]
        display_ps  = display_ps[final_cols]

        # Build dynamic ranges for /90 mode
        active_ranges = _PLAYER_RANGES.copy()
        if per90:
            for c in PER90_COLS:
                if c in _PLAYER_RANGES:
                    lo, hi, ct = _PLAYER_RANGES[c]
                    active_ranges[c] = (round(lo/30, 2), round(hi/30, 2), ct)
            active_ranges["Defcon/90"] = (0, 10, "blue")

        st.markdown(_build_stats_html(display_ps, active_ranges), unsafe_allow_html=True)

        mode_label = "per 90 mins" if per90 else "season totals"
        st.caption(
            f"{len(display_ps)} players · {mode_label} · min {min_mins} mins · "
            "Defcon = clearances+blocks+interceptions + tackles + recoveries · "
            "NpxG = xG minus penalties × 0.76 · PPM = points ÷ price"
        )

        # ── Value scatter ──────────────────────────────────────────────────────
        xgi_col = "xGI"
        if xgi_col in display_ps.columns and "Price" in display_ps.columns:
            st.markdown("---")
            st.markdown(
                '<span style="font-size:14px;font-weight:600;color:#aaa">'
                f'Value Map — xGI{"" if not per90 else "/90"} vs Price</span>',
                unsafe_allow_html=True
            )
            st.caption("Top-left = best value. Bubble size = ownership %. Colour = position.")

            sc = display_ps[display_ps[xgi_col].fillna(0) > 0].dropna(subset=[xgi_col,"Price"])
            if not sc.empty:
                POS_C = {"GKP":"#f4a261","DEF":"#5aabff","MID":"#5fffb0","FWD":"#ff6060"}
                fig_ps = go.Figure()
                for _, r in sc.iterrows():
                    pc  = POS_C.get(str(r.get("Pos","")), "#888")
                    own = float(r["Own%"]) if pd.notna(r.get("Own%")) else 3
                    fig_ps.add_trace(go.Scatter(
                        x=[r["Price"]], y=[r[xgi_col]],
                        mode="markers",
                        marker=dict(size=max(7, min(own*1.3, 28)), color=pc, opacity=0.72,
                                    line=dict(color="rgba(255,255,255,0.10)", width=0.5)),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{r['Player']}</b> · {r.get('Team','')}<br>"
                            f"xGI: {r[xgi_col]:.2f}  |  £{r['Price']:.1f}m  |  {own:.1f}%<extra></extra>"
                        )
                    ))
                for lbl, pc in POS_C.items():
                    fig_ps.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                                marker=dict(size=10, color=pc),
                                                name=lbl, showlegend=True))
                y_title = f"xGI{'/90' if per90 else ''}"
                fig_ps.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    xaxis=dict(title="Price (£m)", showgrid=False, zeroline=False,
                               tickfont=dict(color="#444"), title_font=dict(color="#555", size=11)),
                    yaxis=dict(title=y_title, showgrid=False, zeroline=False,
                               tickfont=dict(color="#444"), title_font=dict(color="#555", size=11)),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888", size=11),
                                orientation="h", y=1.06),
                    margin=dict(l=50, r=20, t=30, b=50), height=440, hovermode="closest"
                )
                st.plotly_chart(fig_ps, use_container_width=True)

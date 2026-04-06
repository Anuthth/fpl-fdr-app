import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import math
import requests
import plotly.graph_objects as go

# ── FPL API ───────────────────────────────────────────────────────────────────
BASE_URL      = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL  = f"{BASE_URL}/fixtures/"

@st.cache_data(ttl=300, show_spinner="📡 Fetching live FPL data...")
def _fetch_bootstrap():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(BOOTSTRAP_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner="📡 Fetching fixtures...")
def _fetch_live_fixtures():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(FIXTURES_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def get_current_gw(bootstrap):
    """
    Returns the active or next upcoming GW.
    Logic:
      1. If any GW has is_current=True AND is not finished → that's the live GW
      2. If current GW is finished → return the next GW (deadline passed / upcoming)
      3. If is_next=True → return that
      4. Fallback: highest finished GW + 1
    """
    events = bootstrap.get("events", [])

    # Check is_current first
    for ev in events:
        if ev.get("is_current"):
            # If current GW is fully finished, advance to next
            if ev.get("finished"):
                nxt = ev["id"] + 1
                return min(nxt, 38)
            return ev["id"]

    # Fall back to is_next
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]

    # Last resort: find highest finished GW + 1
    finished = [ev["id"] for ev in events if ev.get("finished")]
    if finished:
        return min(max(finished) + 1, 38)

    return 1


def player_photo_url(code):
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"

@st.cache_data(ttl=3600, show_spinner=False)
def _verified_photo_url(code):
    """Return photo URL if it exists on PL CDN, else None. Cached 1 hr.
    Uses Range-GET (1 byte) — lightweight and works even when HEAD is blocked."""
    if not code:
        return None
    url = player_photo_url(code)
    try:
        r = requests.get(url, timeout=4, stream=True,
                         headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
        r.close()
        return url if r.status_code in (200, 206) else None
    except Exception:
        return None


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
EO_CSV_FILE          = "EO%.csv"          # Solio expected ownership file

AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25
BGW_PENALTY_FDR = 6.0
DGW_BONUS_FDR   = 2.0   # Reduce difficulty contribution by this for DGW weeks
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
    # FPL API short names (3-letter codes from bootstrap teams)
    "ARS":"Arsenal","AVL":"Aston Villa","BOU":"Bournemouth","BRE":"Brentford",
    "BHA":"Brighton","BUR":"Burnley","CHE":"Chelsea","CRY":"Crystal Palace",
    "EVE":"Everton","FUL":"Fulham","IPS":"Ipswich","LEE":"Leeds",
    "LEI":"Leicester","LIV":"Liverpool","MCI":"Man City","MUN":"Man Utd",
    "NEW":"Newcastle","NFO":"Nottm Forest","SOU":"Southampton",
    "SUN":"Sunderland","TOT":"Spurs","WHU":"West Ham","WOL":"Wolves",
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_fpl_id_map():
    """Load ChrisMusson/FPL-ID-Map — maps player code → FBRef / Understat / Transfermarkt IDs."""
    try:
        url = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/refs/heads/main/Master.csv"
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

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
                    "CS": ex["CS"] + cs,
                    "count": ex.get("count", 1) + 1,
                    "xG_parts": ex.get("xG_parts", [ex["xG"]]) + [xg],
                    "CS_parts": ex.get("CS_parts", [ex["CS"]]) + [cs],
                }
            else:
                projection_data[team][gw_key] = {
                    "display": display, "fdr": fdr, "xG": xg, "CS": cs,
                    "count": 1, "xG_parts": [], "CS_parts": [],
                }

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
                count = cell.get("count", 1)
                fdr_val = cell.get("fdr", 0)
                # DGW bonus: having 2 fixtures is better, reduce difficulty contribution
                if count >= 2:
                    fd += max(0, fdr_val - DGW_BONUS_FDR)
                else:
                    fd += fdr_val
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
    if master_df_full is None:
        return "?", 3
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

def _normalize(s):
    """Strip accents and lowercase for fuzzy matching."""
    import unicodedata
    return unicodedata.normalize('NFD', str(s)).encode('ascii', 'ignore').decode().lower().strip()

def _fuzzy_code(name, lookup):
    """Match player name to FPL photo code — handles accents, prefixes (H.Name), partial names."""
    # 1. Exact match
    if name in lookup:
        return lookup[name]

    name_norm = _normalize(name)

    # 2. Accent-stripped exact match against normalised lookup keys
    for key, val in lookup.items():
        if _normalize(key) == name_norm:
            return val

    # 3. Last token match (handles "H.Ekitiké" → "ekitike", or "Ekitiké" → "ekitike")
    token = name_norm.split(".")[-1].strip()
    if len(token) > 3:
        for key, val in lookup.items():
            if token in _normalize(key):
                return val

    # 4. Surname-only match (last word after space)
    surname = name_norm.split()[-1] if " " in name_norm else name_norm
    if len(surname) > 3:
        for key, val in lookup.items():
            if surname in _normalize(key):
                return val

    return None

def get_captain_picks(proj_df, eo_df, gw, master_df_full, bootstrap):
    """
    Top 2 EV picks + 1 Differential using Solio EO file.
    Uses Solio ID column to look up FPL photo code directly — no fuzzy matching.
    EO values in Solio are already in % (0.3 = 0.3%, 1.74 = 1.74%).
    Differential threshold: EO < 10%.
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

    # EO values are already in % — differential = EO < 10%
    diff_pool = active[~active["Name"].isin(top2["Name"])].copy()
    if eo_df is not None and eo_col in eo_df.columns:
        eo_map = pd.to_numeric(eo_df.set_index("Name")[eo_col], errors="coerce").to_dict()
        diff_pool["_eo"] = diff_pool["Name"].map(eo_map).fillna(999.0)
        diff_pool = diff_pool[diff_pool["_eo"] < 0.10]   # < 10% EO (raw value, multiply by 100 for display)
    diff = diff_pool.nlargest(1, pts_col).copy()
    diff["PickType"] = ["🎯 Differential"]

    picks = pd.concat([top2, diff], ignore_index=True)

    # Build FPL id→code map for direct photo lookup using Solio ID column
    id_to_code = {}
    if bootstrap:
        for p in bootstrap["elements"]:
            id_to_code[int(p["id"])] = p["code"]

    # EO map for display
    eo_map_display = {}
    if eo_df is not None and eo_col in eo_df.columns:
        eo_map_display = pd.to_numeric(eo_df.set_index("Name")[eo_col], errors="coerce").to_dict()

    result = []
    for _, r in picks.iterrows():
        fix, fdr = get_fdr_for_team_gw(r["Team"], gw, master_df_full)

        # Use Solio ID column for direct photo lookup, fall back to fuzzy name match
        solio_id = int(r["ID"]) if "ID" in r.index and pd.notna(r["ID"]) else None
        code  = (id_to_code.get(solio_id) if solio_id else None) or _fuzzy_code(r["Name"], _build_code_lookup(bootstrap))
        photo = _verified_photo_url(code)

        eo_val = eo_map_display.get(r["Name"])
        # EO already in % — just round and display directly
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
    """All players within 0.5 EV of top pick per GW, with Solio EO%.
    Uses Solio ID column for direct FPL photo lookup."""
    id_to_code = {}
    if bootstrap:
        for p in bootstrap["elements"]:
            id_to_code[int(p["id"])] = p["code"]

    # Fallback fuzzy lookup
    code_lookup = _build_code_lookup(bootstrap)

    # Build Solio ID→Name map for direct ID lookup
    id_to_name_solio = {}
    if "ID" in proj_df.columns:
        for _, r in proj_df.iterrows():
            if pd.notna(r.get("ID")):
                id_to_name_solio[int(r["ID"])] = r["Name"]

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

            # Direct ID-based photo lookup, fall back to fuzzy name match
            solio_id = int(r["ID"]) if "ID" in r.index and pd.notna(r["ID"]) else None
            code  = (id_to_code.get(solio_id) if solio_id else None) or _fuzzy_code(r["Name"], code_lookup)
            photo = _verified_photo_url(code)

            bg, fg = club_style(r["Team"])
            eo_raw = eo_map.get(r["Name"], 0)
            # EO already in % — display directly
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
    # ── Section navigation (top of sidebar) ───────────────────────────────────
    _NAV_GROUPS = ["📊 Planning", "🎯 Captain & Picks", "👕 My FPL", "🏟️ Stats"]
    if "nav_cat" not in st.session_state:
        st.session_state["nav_cat"] = _NAV_GROUPS[0]
    st.markdown(
        '<p style="font-size:11px;font-weight:800;color:#5fffb0;letter-spacing:1px;margin-bottom:4px;margin-top:0">SECTION</p>',
        unsafe_allow_html=True,
    )
    nav_cat = st.radio(
        "Section",
        _NAV_GROUPS,
        index=_NAV_GROUPS.index(st.session_state.get("nav_cat", _NAV_GROUPS[0])),
        key="nav_cat",
        label_visibility="collapsed",
    )
    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
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
        events       = bootstrap.get("events", [])
        ev_info      = next((ev for ev in events if ev.get("is_current") or ev.get("is_next")), None)
        is_live_gw   = ev_info and ev_info.get("is_current") and not ev_info.get("finished")
        status_label = "Live" if is_live_gw else "Next GW"
        st.sidebar.success(f"GW{current_gw} | {status_label}")

        # -- Deadline countdown Bangkok (GMT+7) --------------------------------
        from datetime import datetime, timezone, timedelta
        BKK     = timezone(timedelta(hours=7))
        now_utc = datetime.now(timezone.utc)

        next_deadline    = None
        next_deadline_gw = None
        for ev in sorted(events, key=lambda e: e["id"]):
            dl_raw = ev.get("deadline_time", "")
            if not dl_raw:
                continue
            try:
                dl_utc = datetime.fromisoformat(dl_raw.replace("Z", "+00:00"))
                if dl_utc > now_utc:
                    next_deadline    = dl_utc
                    next_deadline_gw = ev["id"]
                    break
            except Exception:
                continue

        if next_deadline:
            dl_bkk    = next_deadline.astimezone(BKK)
            dl_fmt    = dl_bkk.strftime("%a %d %b  %H:%M")
            # Pass deadline as Unix timestamp (ms) to JS
            deadline_ms = int(next_deadline.timestamp() * 1000)
            gw_label    = f"GW{next_deadline_gw}"

            countdown_html = f"""
<style>
  #cdbox {{
    background: #0a1a0a;
    border: 1px solid #1e3a1e;
    border-radius: 6px;
    padding: 10px 12px;
    font-family: 'Inter', sans-serif;
  }}
  #cdlabel {{
    font-size: 10px;
    color: #555;
    font-weight: 700;
    letter-spacing: .7px;
    margin-bottom: 3px;
  }}
  #cddate {{
    font-size: 12px;
    color: #888;
    margin-bottom: 4px;
  }}
  #cdtime {{
    font-size: 20px;
    font-weight: 800;
    color: #5fffb0;
    font-variant-numeric: tabular-nums;
    letter-spacing: 1px;
  }}
</style>
<div id="cdbox">
  <div id="cdlabel">{gw_label} DEADLINE</div>
  <div id="cddate">{dl_fmt} (Bangkok)</div>
  <div id="cdtime">--d --h --m --s</div>
</div>
<script>
  const deadline = {deadline_ms};
  const el = document.getElementById('cdtime');
  const box = document.getElementById('cdbox');
  const lbl = document.getElementById('cdlabel');

  function update() {{
    const now  = Date.now();
    const diff = Math.max(0, deadline - now);
    const s    = Math.floor(diff / 1000);
    const days = Math.floor(s / 86400);
    const hrs  = Math.floor((s % 86400) / 3600);
    const mins = Math.floor((s % 3600) / 60);
    const secs = s % 60;

    const pad = n => String(n).padStart(2, '0');

    if (diff === 0) {{
      el.style.color = '#ff6060';
      el.textContent = 'DEADLINE PASSED';
      box.style.background = '#2b0a0a';
      box.style.borderColor = '#ff4444';
      return;
    }}

    if (days > 0) {{
      el.textContent = days + 'd ' + pad(hrs) + 'h ' + pad(mins) + 'm ' + pad(secs) + 's';
      el.style.color = '#5fffb0';
      box.style.background = '#0a1a0a';
      box.style.borderColor = '#1e3a1e';
    }} else if (hrs > 3) {{
      el.textContent = pad(hrs) + 'h ' + pad(mins) + 'm ' + pad(secs) + 's';
      el.style.color = '#5aabff';
      box.style.background = '#0a1525';
      box.style.borderColor = '#1a3a5a';
    }} else {{
      el.textContent = pad(hrs) + 'h ' + pad(mins) + 'm ' + pad(secs) + 's';
      el.style.color = '#ff6060';
      box.style.background = '#2b1a00';
      box.style.borderColor = '#ff8800';
      lbl.textContent = '{gw_label} DEADLINE SOON';
      lbl.style.color = '#ff8800';
    }}
  }}

  update();
  setInterval(update, 1000);
</script>
"""
            with st.sidebar:
                components.html(countdown_html, height=130)

    except Exception as e:
        st.sidebar.warning(f"Live failed: {e}")

ratings_df, fixtures_df = load_csv_data()
proj_df = load_projections_csv()
eo_df   = load_eo_csv()
id_map_df = load_fpl_id_map()

# NOTE: Always use CSV fixtures as source of truth for DGW/BGW scheduling.
# The FPL live API may not reflect manually-confirmed DGW/BGW rescheduling yet.
# raw_fixtures is still available for live stats (team stats, live radar, etc.)
# if live_ok and bootstrap and raw_fixtures:
#     fixtures_df = build_live_fixtures_df(bootstrap, raw_fixtures)

# ── Build FPL element_id → external profile links lookup ──────────────────────
_ext_links: dict = {}
if not id_map_df.empty and bootstrap is not None:
    _elements     = pd.DataFrame(bootstrap["elements"])
    _code_to_elid = dict(zip(_elements["code"], _elements["id"]))
    for _, _r in id_map_df.iterrows():
        _el_id = _code_to_elid.get(_r.get("code"))
        if _el_id is None:
            continue
        _us = _r.get("understat")
        _fb = _r.get("fbref")
        _tm = _r.get("transfermarkt")
        _ext_links[int(_el_id)] = {
            "understat":    int(_us)  if pd.notna(_us) else None,
            "fbref":        str(_fb)  if pd.notna(_fb) else None,
            "transfermarkt":int(_tm)  if pd.notna(_tm) else None,
        }

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
    # Still build master_df_full so fixture lookups work in My Team / GW Planner
    _fb_start = current_gw if live_ok else start_gw
    master_df_full = create_all_data(
        fixtures_df.to_dict("records"), _fb_start, 38,
        ratings_df.to_dict("records"), None
    )

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
# ── Conditional tab groups based on sidebar navigation ────────────────────────
if nav_cat == "📊 Planning":
    tab1, tab2, tab3, tab4 = st.tabs(["📊 FDR", "⚽ xG", "🧤 xCS", "📈 Team Ratings"])
elif nav_cat == "🎯 Captain & Picks":
    tab5, tab6, tab7, tab13 = st.tabs(["🎯 Captain Picks", "🏅 Captain Matrix", "📋 Cheatsheet", "🔍 Differentials"])
elif nav_cat == "👕 My FPL":
    tab8, tab9, tab10, tab14 = st.tabs(["📡 Live Radar", "👕 My Team", "📅 GW Planner", "🏆 Mini-League"])
else:  # 🏟️ Stats
    tab11, tab12 = st.tabs(["🏟️ Team Stats", "👤 Player Stats"])

# ── Shared helper: build clean HTML heatmap table ─────────────────────────────
def _heatmap_table(df_display, gw_cols, value_key, label_fn, color_fn, total_col, total_fmt, table_id="ht"):
    """
    Render a dark heatmap HTML table with clickable column sorting.
    Click any column header to sort asc/desc. Arrow indicates direction.
    """
    th_base = "padding:8px 10px;text-align:center;color:#666;font-size:11px;font-weight:600;letter-spacing:.5px;border-bottom:1px solid #2a2a2a;cursor:pointer;user-select:none;"

    header = (
        f'<th style="{th_base}text-align:left;min-width:130px;color:#888" '
        f'onclick="sortHT(\'{table_id}\',0,\'str\')">TEAM ↕</th>'
        f'<th style="{th_base}color:#aaa;min-width:80px" '
        f'onclick="sortHT(\'{table_id}\',1,\'num\')">{total_col} ↕</th>'
    )
    for ci, col in enumerate(gw_cols):
        gw_label = col.replace("GW","<span style='color:#555;font-size:9px'>GW</span>")
        header += (f'<th style="{th_base}min-width:70px" '
                   f'onclick="sortHT(\'{table_id}\',{ci+2},\'num\')">{gw_label} ↕</th>')

    rows = ""
    for _, row in df_display.iterrows():
        team = row["Team"]
        bg_club, _ = club_style(team)
        dot = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
               f'background:{bg_club};margin-right:7px;vertical-align:middle;flex-shrink:0"></span>')
        try:
            total_num = float(row[total_col])
        except:
            total_num = 0
        total_display = total_fmt(total_num)

        td_team  = (f'<td style="padding:8px 10px;font-weight:600;color:#e0e0e0;font-size:13px;'
                    f'border-bottom:1px solid #1e1e1e;white-space:nowrap" data-val="{team}">{dot}{team}</td>')
        td_total = (f'<td style="padding:8px 10px;text-align:center;color:#aaa;font-size:12px;'
                    f'border-bottom:1px solid #1e1e1e;font-weight:600" data-val="{total_num}">{total_display}</td>')

        cells = td_team + td_total
        for col in gw_cols:
            cell = row[col]
            if isinstance(cell, dict):
                val  = cell.get(value_key, 0)
                lbl  = label_fn(cell)
                cbg, cfg = color_fn(val)
                # Detect DGW via count field
                is_dgw = cell.get("count", 1) >= 2
                if is_dgw:
                    dgw_badge = (
                        '<span style="font-size:8px;background:#FFD700;color:#111;border-radius:2px;'
                        'padding:0 4px;font-weight:800;letter-spacing:.5px;display:block;margin-bottom:2px">DGW</span>'
                    )
                    # lbl may be multi-part (e.g. "ARS (H) + BUR (A)" or "1.85 + 1.23")
                    lbl_parts = lbl.split(" + ") if " + " in lbl else [lbl]
                    content_html = '<br>'.join(
                        f'<span style="font-size:11px;font-weight:700">{p}</span>' for p in lbl_parts
                    )
                    inner_lbl = dgw_badge + content_html
                    td_style = (f'padding:4px 6px;text-align:center;background:{cbg};color:{cfg};'
                                f'font-weight:700;border-bottom:1px solid #161616;'
                                f'border-right:1px solid #1e1e1e;vertical-align:middle;min-width:80px;line-height:1.5')
                else:
                    inner_lbl = lbl
                    td_style = (f'padding:6px 8px;text-align:center;background:{cbg};color:{cfg};'
                                f'font-size:12px;font-weight:700;border-bottom:1px solid #161616;'
                                f'border-right:1px solid #1e1e1e')
                cells += f'<td style="{td_style}" data-val="{val}">{inner_lbl}</td>'
            else:
                cells += ('<td style="padding:6px 8px;text-align:center;background:#1a0a0a;color:#e74c3c;'
                          'font-size:10px;font-weight:800;border-bottom:1px solid #161616;'
                          'border-right:1px solid #1e1e1e;letter-spacing:1px" data-val="-1">BGW</td>')

        rows += (f'<tr onmouseover="this.style.background=\'#1a1a1a\'" '
                 f'onmouseout="this.style.background=\'transparent\'">{cells}</tr>')

    sort_js = f"""
<script>
(function() {{
  var _sortState = {{}};
  window.sortHT = function(tid, col, dtype) {{
    var tbl = document.getElementById(tid);
    if (!tbl) return;
    var tbody = tbl.querySelector('tbody');
    var rows  = Array.from(tbody.querySelectorAll('tr'));
    var key   = tid + '_' + col;
    var asc   = _sortState[key] !== true;
    _sortState[key] = asc;

    rows.sort(function(a, b) {{
      var av = a.cells[col] ? a.cells[col].getAttribute('data-val') : '';
      var bv = b.cells[col] ? b.cells[col].getAttribute('data-val') : '';
      if (dtype === 'num') {{
        av = parseFloat(av) || -999;
        bv = parseFloat(bv) || -999;
        return asc ? av - bv : bv - av;
      }} else {{
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }}
    }});

    // Update arrow indicators on all headers
    var ths = tbl.querySelectorAll('thead th');
    ths.forEach(function(th, i) {{
      var txt = th.innerHTML.replace(/ [↑↓↕]/g, '');
      th.innerHTML = txt + (i === col ? (asc ? ' ↑' : ' ↓') : ' ↕');
    }});

    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }};
}})();
</script>
"""

    return (
        f'<!DOCTYPE html><html><head>'
        f'<style>body{{margin:0;padding:0;background:#0d1117;}}'
        f'table{{border-collapse:collapse;width:100%;font-family:\'Inter\',Arial,sans-serif;background:#0d1117;}}'
        f'th{{cursor:pointer;user-select:none;}}'
        f'th:hover{{color:#fff!important;}}'
        f'</style></head><body>'
        f'<div style="overflow-x:auto;margin:0">'
        f'<table id="{table_id}" style="border-collapse:collapse;width:100%;'
        f'font-family:\'Inter\',sans-serif;background:#0d1117">'
        f'<thead><tr style="background:#161b22">{header}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>'
        f'{sort_js}'
        f'</body></html>'
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
        '<a href="https://fpl.solioanalytics.com/" target="_blank" '
        'style="background:#fff;color:#000;font-size:11px;font-weight:700;'
        'padding:4px 12px;border-radius:4px;text-decoration:none;'
        'letter-spacing:.5px;white-space:nowrap">Try Solio ↗</a>'
        '</div>'
    )

def _export_html(title: str, body_html: str) -> str:
    """Wrap body_html in a standalone dark-themed HTML with embedded Solio logo."""
    import base64
    try:
        with open("Solio_Logo_Neg_RGB.png", "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        logo_tag = (
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="height:28px;opacity:0.7;vertical-align:middle">'
        )
    except Exception:
        logo_tag = '<span style="color:#5fffb0;font-weight:800;font-size:13px">SOLIO</span>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 20px; background: #0d1117; font-family: 'Inter', Arial, sans-serif; color: #e0e0e0; }}
  .header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid #1e1e1e; padding-bottom: 12px; }}
  .title {{ font-size: 20px; font-weight: 800; color: #e0e0e0; }}
  .footer {{ margin-top: 20px; border-top: 1px solid #1e1e1e; padding-top: 10px; display: flex; align-items: center; gap: 10px; font-size: 11px; color: #444; }}
</style>
</head>
<body>
  <div class="header">
    <span class="title">{title}</span>
    {logo_tag}
  </div>
  {body_html}
  <div class="footer">
    {logo_tag}
    <span>Generated by Solio FPL · fpl-fdr-app</span>
  </div>
</body>
</html>"""

SOLIO_NO_FILE_MSG = (
    "\ud83d\udcc2 **{filename} not found.**\n\n"
    "Download from [Solio Analytics](https://www.solioanalytics.com) "
    "and place it in your app folder alongside `app.py`."
)


# ────────────────────────────────────────────────────────────────────────────

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


# =============================================================================
# FOTMOB SCRAPING — Team Stats & Player Stats
# =============================================================================

FOTMOB_PL_ID = 47  # Premier League on Fotmob

_FOTMOB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fotmob.com/",
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fotmob_league():
    """Fetch Premier League overview from Fotmob (table + season info)."""
    r = requests.get(
        f"https://www.fotmob.com/api/leagues?id={FOTMOB_PL_ID}",
        headers=_FOTMOB_HEADERS, timeout=15,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fotmob_deep_stat(season_id: str, stat: str):
    """Fetch top-100 player list for a single stat from Fotmob."""
    url = (
        f"https://www.fotmob.com/api/leagueseasondeepstats"
        f"?id={FOTMOB_PL_ID}&season={season_id}&type=players&stat={stat}"
    )
    r = requests.get(url, headers=_FOTMOB_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _fotmob_pos(pos_str: str) -> str:
    p = (pos_str or "").lower()
    if "goal" in p:                              return "GKP"
    if "defend" in p:                            return "DEF"
    if "mid" in p:                               return "MID"
    if "attack" in p or "forward" in p or "wing" in p: return "FWD"
    return "?"


def _fotmob_season_id(league_data: dict):
    """Extract the active season ID string from Fotmob league JSON."""
    details = league_data.get("details", {})
    sid = details.get("selectedSeason")
    if sid:
        return str(sid)
    for s in league_data.get("seasons", []):
        if s.get("isSelected") or s.get("selected"):
            return str(s.get("id") or s.get("seasonId", ""))
    return None


def _parse_fotmob_items(data: dict) -> list:
    """Robustly extract the player list from a deep-stat API response."""
    ss = data.get("statsSection", {})
    # path 1: statsSubSections[0].items
    sub = ss.get("statsSubSections", [])
    if sub:
        items = sub[0].get("items", [])
        if items:
            return items
    # path 2: topLists
    top = ss.get("topLists", [])
    if top:
        return top
    # path 3: flat fallback
    return data.get("items", data.get("topLists", []))


def build_fotmob_team_stats_df(league_data: dict) -> pd.DataFrame:
    """Build team stats table from Fotmob league-table data."""
    try:
        tables = league_data.get("table", [])
        if not tables:
            return pd.DataFrame()

        table_all = (
            tables[0]
            .get("data", {})
            .get("table", {})
            .get("all", [])
        )
        if not table_all:
            return pd.DataFrame()

        rows = []
        for t in table_all:
            scores = t.get("scoresStr", "0-0")
            try:
                gf, ga = map(int, scores.split("-"))
            except (ValueError, AttributeError):
                gf, ga = 0, 0

            xg_raw  = t.get("xg")
            xgc_raw = t.get("xgc")

            # Normalise team name through existing map
            raw_name = t.get("name", "")
            name = TEAM_NAME_MAP.get(raw_name, raw_name)

            rows.append({
                "Team": name,
                "MP":   int(t.get("played", 0)),
                "W":    int(t.get("wins",   0)),
                "D":    int(t.get("draws",  0)),
                "L":    int(t.get("losses", 0)),
                "GF":   gf,
                "GA":   ga,
                "GD":   int(t.get("goalConDiff", gf - ga)),
                "Pts":  int(t.get("pts", 0)),
                "xG":   round(float(xg_raw),  1) if xg_raw  is not None else 0.0,
                "xGC":  round(float(xgc_raw), 1) if xgc_raw is not None else 0.0,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["xGDiff"] = (df["xG"] - df["xGC"]).round(1)
            df = df.sort_values("Pts", ascending=False).reset_index(drop=True)
            df.insert(0, "Rk", range(1, len(df) + 1))
        return df
    except Exception:
        return pd.DataFrame()


def build_fotmob_player_stats_df(league_data: dict) -> pd.DataFrame:
    """
    Build player stats by merging multiple Fotmob deep-stat pages.
    Stats fetched: expected_goals, expected_assists, goals, assists,
    minutes_played, interceptions, tackles_won, clearances_defensive,
    clean_sheets, expected_goals_conceded.
    """
    season_id = _fotmob_season_id(league_data)
    if not season_id:
        return pd.DataFrame()

    # stat_key → DataFrame column name
    STAT_MAP = {
        "expected_goals":          "xG",
        "expected_assists":        "xA",
        "goals":                   "Goals",
        "assists":                 "Assists",
        "minutes_played":          "_mins_raw",
        "interceptions":           "Interceptions",
        "tackles_won":             "Tackles",
        "clearances_defensive":    "Clearances",
        "clean_sheets":            "CS",
        "expected_goals_conceded": "xGC",
    }

    players: dict = {}  # pid → {field: value}

    for stat_key, field in STAT_MAP.items():
        try:
            items = _parse_fotmob_items(fetch_fotmob_deep_stat(season_id, stat_key))
        except Exception:
            continue

        for item in items:
            pid = str(item.get("id", ""))
            if not pid:
                continue
            if pid not in players:
                raw_team = item.get("teamName", "")
                players[pid] = {
                    "Player": item.get("name", ""),
                    "Team":   TEAM_NAME_MAP.get(raw_team, raw_team),
                    "Pos":    _fotmob_pos(item.get("position", "")),
                    "Mins":   0,
                }
            players[pid][field] = item.get("statValue", 0)
            # Keep max minutes seen across all stat calls
            mp = int(item.get("minutesPlayed", 0) or 0)
            if mp > players[pid].get("Mins", 0):
                players[pid]["Mins"] = mp

    if not players:
        return pd.DataFrame()

    rows = []
    for p in players.values():
        # Prefer the dedicated minutes_played stat value
        mins = float(p.get("_mins_raw") or p.get("Mins") or 0)
        p["Mins"] = int(mins)
        nineties = mins / 90.0 if mins > 0 else 0.0

        xg  = float(p.get("xG",  0) or 0)
        xa  = float(p.get("xA",  0) or 0)
        xgi = round(xg + xa, 2)
        xgc = float(p.get("xGC", 0) or 0)

        inter  = float(p.get("Interceptions", 0) or 0)
        tackle = float(p.get("Tackles",       0) or 0)
        clear  = float(p.get("Clearances",    0) or 0)
        defcon = round(inter + tackle + clear, 1)
        defcon90 = round(defcon / nineties, 2) if nineties >= 0.5 else None

        rows.append({
            "Player":    p.get("Player", ""),
            "Team":      p.get("Team",   ""),
            "Pos":       p.get("Pos",    "?"),
            "Mins":      p["Mins"],
            "Goals":     int(p.get("Goals",   0) or 0),
            "Assists":   int(p.get("Assists", 0) or 0),
            "xG":        round(xg,  2),
            "xA":        round(xa,  2),
            "xGI":       xgi,
            "xGC":       round(xgc, 2),
            "CS":        int(p.get("CS", 0) or 0),
            "Defcon":    defcon,
            "Defcon/90": defcon90,
            "_90s":      round(nineties, 2),
        })

    df = pd.DataFrame(rows)
    df = df[df["Mins"] > 0].copy().reset_index(drop=True)
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
    # Fotmob-specific
    "Goals":     (0, 25,  "positive"),
    "Assists":   (0, 15,  "positive"),
    "CS":        (0, 15,  "blue"),
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
# CSV DATA LOADERS — Team & Player Stats
# =============================================================================

@st.cache_data(show_spinner=False)
def load_csv_team_stats():
    df = pd.read_csv("pl_teams_stats_2025_2026.csv")
    df = df[["Team Name", "Expected goals", "xG conceded", "xG difference"]].copy()
    df.columns = ["Team", "xG", "xGC", "xGDiff"]
    for c in ["xG", "xGC", "xGDiff"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_csv_player_stats():
    df = pd.read_csv("pl_players_stats_2025_2026.csv")
    col_map = {
        "Player Name":                     "Player",
        "Team":                            "Team",
        "Country":                         "Country",
        "Matches Played":                  "MP",
        "Minutes Played":                  "Mins",
        "Top scorer":                      "Goals",
        "Assists":                         "Assists",
        "Goals + Assists":                 "G+A",
        "Goals per 90":                    "G/90",
        "Big chances missed":              "BCM",
        "Shots per 90":                    "Sh/90",
        "Shots on target per 90":          "SoT/90",
        "Big chances created":             "BCC",
        "Chances created":                 "CC",
        "Expected goals (xG)":             "xG",
        "Expected goals (xG) per 90":      "xG/90",
        "Expected goals on target (xGOT)": "xGOT",
        "Expected assist (xA)":            "xA",
        "Expected assist (xA) per 90":     "xA/90",
        "xG + xA per 90":                  "xG+xA/90",
    }
    df = df[[c for c in col_map if c in df.columns]].rename(columns=col_map)
    num_cols = ["MP", "Mins", "Goals", "Assists", "G+A", "G/90", "BCM", "Sh/90",
                "SoT/90", "BCC", "CC", "xG", "xG/90", "xGOT", "xA", "xA/90", "xG+xA/90"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_fpl_positions(player_df, bootstrap):
    """Match FPL position and element ID using exact + fuzzy name strategies."""
    import unicodedata
    from difflib import get_close_matches

    def _norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()

    elements = pd.DataFrame(bootstrap["elements"])
    POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    lookup = {}
    for _, r in elements.iterrows():
        data = {"pos": POS.get(r["element_type"], "?"), "id": int(r["id"])}
        # full name, web_name, second_name, last word of web_name
        for key in [
            _norm(f"{r['first_name']} {r['second_name']}"),
            _norm(r["web_name"]),
            _norm(r["second_name"]),
            _norm(r["web_name"].split(".")[-1].strip()),  # "B.Guimarães" → "Guimarães"
        ]:
            if key and key not in lookup:
                lookup[key] = data

    all_keys = list(lookup.keys())

    def _match(name):
        n = _norm(name)
        if n in lookup:
            return lookup[n]
        # try last word of CSV name (e.g. "Guimarães" from "Bruno Guimarães")
        last = n.split()[-1] if n.split() else ""
        if last and last in lookup:
            return lookup[last]
        # fuzzy fallback — catches nicknames and minor spelling differences
        hits = get_close_matches(n, all_keys, n=1, cutoff=0.82)
        return lookup[hits[0]] if hits else {}

    df = player_df.copy()
    df["Pos"]     = df["Player"].apply(lambda n: _match(n).get("pos", "?"))
    df["_fpl_id"] = df["Player"].apply(lambda n: _match(n).get("id"))
    return df


# =============================================================================
# TAB 8 — Team Stats
# =============================================================================

# ── FPL Team fetch helpers ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_fpl_entry(team_id: int):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fpl_history(team_id: int):
    """Return chips played from /history/ endpoint: [{"name":"wildcard","event":5,...},...]"""
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fpl_picks(team_id: int, gw: int):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_mini_league(league_id: int, page: int = 1):
    """Fetch classic mini-league standings from FPL API."""
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    return r.json()

def build_squad_from_picks(picks_data, bootstrap):
    el_map = {p["id"]: p for p in bootstrap["elements"]}
    teams  = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    POS    = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    squad  = []
    for pick in picks_data["picks"]:
        el = el_map.get(pick["element"], {})
        squad.append({
            "id":         pick["element"],
            "name":       el.get("web_name", "?"),
            "team":       teams.get(el.get("team"), "?"),
            "pos":        POS.get(el.get("element_type"), "?"),
            "price":      el.get("now_cost", 0) / 10,
            "sel%":       el.get("selected_by_percent", "0"),
            "position":   pick["position"],
            "multiplier": pick["multiplier"],
            "is_captain": pick["is_captain"],
            "is_vice":    pick["is_vice_captain"],
            "code":       el.get("code"),
        })
    return squad

def enrich_squad_solio(squad, proj_df, eo_df, gw):
    pts_col = f"{gw}_Pts"
    eo_col  = f"{gw}_eo"
    ev_map, eo_map = {}, {}
    if proj_df is not None and pts_col in proj_df.columns:
        ev_map = pd.to_numeric(proj_df.set_index("ID")[pts_col], errors="coerce").to_dict()
    if eo_df is not None and eo_col in eo_df.columns:
        eo_map = pd.to_numeric(eo_df.set_index("ID")[eo_col], errors="coerce").to_dict()
    for p in squad:
        pid = p["id"]
        ev  = ev_map.get(pid)
        eo  = eo_map.get(pid)
        p["ev"]  = round(float(ev), 2) if ev is not None and not (isinstance(ev, float) and np.isnan(ev)) else None
        p["eo%"] = round(float(eo) * 100, 1) if eo is not None and not (isinstance(eo, float) and np.isnan(eo)) else None
    return squad

def _squad_player_card(p, gw, master_df_full):
    club_bg, club_fg = club_style(p["team"])
    fix, fdr = get_fdr_for_team_gw(p["team"], gw, master_df_full)
    fdr_bg = FDR_BG.get(fdr, "#444")
    fdr_fg = FDR_FG.get(fdr, "#fff")
    badge = ""
    if p["is_captain"]:
        badge = '<span style="position:absolute;top:4px;right:4px;background:#ffcc00;color:#000;font-size:9px;font-weight:800;border-radius:3px;padding:1px 5px">C</span>'
    elif p["is_vice"]:
        badge = '<span style="position:absolute;top:4px;right:4px;background:#aaa;color:#000;font-size:9px;font-weight:800;border-radius:3px;padding:1px 5px">V</span>'
    initials = "".join(w[0].upper() for w in p["name"].replace(".", " ").split() if w)[:2] or "?"
    photo_url = _verified_photo_url(p.get("code"))
    if photo_url:
        img_html = f'<img src="{photo_url}" style="width:44px;height:56px;object-fit:cover;object-position:top;border-radius:4px;display:block">'
    else:
        img_html = (
            f'<div style="width:44px;height:56px;border-radius:4px;display:flex;'
            f'align-items:center;justify-content:center;font-size:16px;font-weight:800;'
            f'color:{club_fg};background:rgba(0,0,0,0.3)">{initials}</div>'
        )
    ev_str = f"{p['ev']} pts" if p.get("ev") else "—"
    eo_str = f"EO {p['eo%']}%" if p.get("eo%") else ""
    opacity = "0.5" if p["position"] > 11 else "1"
    return (
        f'<div style="position:relative;background:{club_bg};border-radius:8px;'
        f'padding:8px 10px;display:flex;align-items:center;gap:10px;opacity:{opacity};'
        f'border:1px solid rgba(255,255,255,0.06);margin-bottom:4px">'
        f'{badge}<div style="flex-shrink:0">{img_html}</div>'
        f'<div style="min-width:0;flex:1">'
        f'<div style="color:{club_fg};font-weight:700;font-size:13px;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis">{p["name"]}</div>'
        f'<div style="color:{club_fg};font-size:11px;opacity:0.7">{p["team"]} · {p["pos"]}</div>'
        f'<div style="display:flex;align-items:center;gap:5px;margin-top:4px;flex-wrap:wrap">'
        f'<span style="background:{fdr_bg};color:{fdr_fg};border-radius:3px;padding:1px 6px;font-size:11px;font-weight:700">{fix}</span>'
        f'<span style="color:{club_fg};font-size:12px;font-weight:700">{ev_str}</span>'
        f'<span style="color:{club_fg};font-size:11px;opacity:0.65">{eo_str}</span>'
        f'</div></div></div>'
    )


def _pitch_token(p, gw, master_df_full, is_bench=False):
    """FPL-style compact pitch token: photo/initials + club badge + fixture + EV."""
    club_bg, club_fg = club_style(p["team"])
    fix, fdr = get_fdr_for_team_gw(p["team"], gw, master_df_full)
    fdr_bg  = FDR_BG.get(fdr, "#444")
    fdr_fg  = FDR_FG.get(fdr, "#fff")
    initials = "".join(w[0].upper() for w in p["name"].replace(".", " ").split() if w)[:2] or "?"
    photo_url = _verified_photo_url(p.get("code"))

    # Photo or initials in a circle
    if photo_url:
        avatar = (
            f'<div style="width:54px;height:54px;border-radius:50%;overflow:hidden;'
            f'border:2.5px solid rgba(255,255,255,0.5);margin:0 auto 3px;background:{club_bg}">'
            f'<img src="{photo_url}" style="width:100%;height:100%;object-fit:cover;object-position:top">'
            f'</div>'
        )
    else:
        avatar = (
            f'<div style="width:54px;height:54px;border-radius:50%;display:flex;'
            f'align-items:center;justify-content:center;font-size:18px;font-weight:800;'
            f'color:{club_fg};background:{club_bg};border:2.5px solid rgba(255,255,255,0.5);'
            f'margin:0 auto 3px">{initials}</div>'
        )

    # Captain / vice badge
    cap_badge = ""
    if p.get("is_captain"):
        cap_badge = '<span style="position:absolute;top:0;right:0;background:#ffcc00;color:#000;font-size:8px;font-weight:800;border-radius:50%;width:16px;height:16px;display:flex;align-items:center;justify-content:center;line-height:1">C</span>'
    elif p.get("is_vice"):
        cap_badge = '<span style="position:absolute;top:0;right:0;background:#bbb;color:#000;font-size:8px;font-weight:800;border-radius:50%;width:16px;height:16px;display:flex;align-items:center;justify-content:center;line-height:1">V</span>'

    ev_str  = f"{p['ev']}" if p.get("ev") else "—"
    fix_str = fix if fix else "?"
    opacity = "0.55" if is_bench else "1"

    # Short name (up to 9 chars)
    short = p["name"] if len(p["name"]) <= 9 else p["name"][:8] + "…"

    return (
        f'<div style="display:inline-block;text-align:center;width:68px;position:relative;opacity:{opacity};vertical-align:top">'
        f'{cap_badge}'
        f'{avatar}'
        f'<div style="background:{club_bg};color:{club_fg};font-size:9.5px;font-weight:700;'
        f'padding:2px 4px;border-radius:3px;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;max-width:68px;margin-bottom:2px">{short}</div>'
        f'<div style="display:flex;gap:2px;justify-content:center;flex-wrap:nowrap">'
        f'<span style="background:{fdr_bg};color:{fdr_fg};font-size:8px;font-weight:700;'
        f'padding:1px 4px;border-radius:2px;white-space:nowrap">{fix_str}</span>'
        f'<span style="background:rgba(0,0,0,0.55);color:#5fffb0;font-size:8px;font-weight:800;'
        f'padding:1px 4px;border-radius:2px;white-space:nowrap">{ev_str}</span>'
        f'</div>'
        f'</div>'
    )


def _render_pitch(players_by_pos, gw, master_df_full, bench_players=None):
    """Render a full-pitch SVG background with player tokens in formation rows.
    players_by_pos: dict with keys GKP/DEF/MID/FWD, each a list of player dicts."""
    rows_html = ""
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        grp = players_by_pos.get(pos, [])
        if not grp:
            continue
        tokens = "".join(_pitch_token(p, gw, master_df_full) for p in grp)
        rows_html += (
            f'<div style="display:flex;justify-content:center;gap:8px;'
            f'margin-bottom:14px;flex-wrap:nowrap">{tokens}</div>'
        )

    bench_html = ""
    if bench_players:
        btokens = "".join(_pitch_token(p, gw, master_df_full, is_bench=True) for p in bench_players)
        bench_html = (
            '<div style="margin-top:10px;padding-top:10px;border-top:2px dashed rgba(255,255,255,0.2)">'
            '<div style="font-size:9px;color:rgba(255,255,255,0.4);font-weight:700;'
            'letter-spacing:.6px;text-align:center;margin-bottom:8px">BENCH</div>'
            f'<div style="display:flex;justify-content:center;gap:8px;flex-wrap:nowrap">{btokens}</div>'
            '</div>'
        )

    return (
        # Outer pitch container
        '<div style="position:relative;border-radius:12px;overflow:hidden;'
        'border:3px solid rgba(255,255,255,0.25);margin:6px 0">'
        # Green pitch stripes background
        '<div style="position:absolute;inset:0;'
        'background:repeating-linear-gradient(to bottom,#1b5e20 0px,#1b5e20 64px,#1e7024 64px,#1e7024 128px)">'
        '</div>'
        # Pitch markings overlay
        '<div style="position:absolute;inset:0;pointer-events:none">'
        # Top penalty area
        '<div style="position:absolute;top:0;left:50%;transform:translateX(-50%);'
        'width:55%;height:56px;border:1.5px solid rgba(255,255,255,0.18);border-top:none;border-radius:0 0 8px 8px"></div>'
        # Center line
        '<div style="position:absolute;top:50%;left:4%;right:4%;height:1.5px;background:rgba(255,255,255,0.18)"></div>'
        # Center circle
        '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        'width:72px;height:72px;border:1.5px solid rgba(255,255,255,0.18);border-radius:50%"></div>'
        # Center dot
        '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        'width:5px;height:5px;background:rgba(255,255,255,0.25);border-radius:50%"></div>'
        # Bottom penalty area
        '<div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);'
        'width:55%;height:56px;border:1.5px solid rgba(255,255,255,0.18);border-bottom:none;border-radius:8px 8px 0 0"></div>'
        '</div>'
        # Content (tokens on top of pitch)
        f'<div style="position:relative;z-index:1;padding:18px 12px 14px">'
        f'{rows_html}'
        f'{bench_html}'
        '</div>'
        '</div>'
    )

if nav_cat == "📊 Planning":
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
            ("#1a0a0a","#e74c3c","BGW"),
        ]
        legend_html = '<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;align-items:center">'
        for bg, fg, lbl in legend_items:
            legend_html += (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;'
                            f'font-size:11px;font-weight:700">{lbl}</span>')
        legend_html += (
            '<span style="background:#FFD700;color:#111;padding:2px 8px;border-radius:3px;'
            'font-size:11px;font-weight:800">DGW</span>'
            ' <span style="color:#555;font-size:11px">= Double Gameweek</span>'
        )
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
            table_id="tbl_fdr",
        )
        # DGW rows are taller (two fixture lines), use 55px per row
        _tbl_h = 60 + len(df_d) * 55
        components.html(html, height=_tbl_h, scrolling=False)

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

        def _xg_label(c):
            parts = c.get("xG_parts", [])
            if len(parts) >= 2:
                return " + ".join(f"{v:.2f}" for v in parts)
            return f"{c.get('xG',0):.2f}"

        html = _heatmap_table(
            df_d, gw_columns,
            value_key="xG",
            label_fn=_xg_label,
            color_fn=_xg_color,
            total_col="Total_xG",
            total_fmt=lambda v: f"{v:.2f}",
            table_id="tbl_xg",
        )
        _tbl_h = 60 + len(df_d) * 55
        components.html(html, height=_tbl_h, scrolling=False)

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

        def _xcs_label(c):
            parts = c.get("CS_parts", [])
            if len(parts) >= 2:
                return " + ".join(f"{v*100:.0f}%" for v in parts)
            return f"{c.get('CS',0)*100:.0f}%"

        html = _heatmap_table(
            df_d, gw_columns,
            value_key="CS",
            label_fn=_xcs_label,
            color_fn=_xcs_color,
            total_col="Total_xCS",
            total_fmt=lambda v: f"{v*100:.0f}%",
            table_id="tbl_xcs",
        )
        _tbl_h = 60 + len(df_d) * 55
        components.html(html, height=_tbl_h, scrolling=False)

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


# ────────────────────────────────────────────────────────────────────────────
elif nav_cat == "🎯 Captain & Picks":
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

                    initials = "".join(w[0].upper() for w in p["Name"].replace(".", " ").split() if w)[:2] or "?"
                    if p["photo"]:
                        img_html = (
                            f'<div style="width:80px;margin:0 auto 8px auto">'
                            f'<img src="{p["photo"]}" '
                            f'style="width:80px;height:100px;object-fit:cover;object-position:top;'
                            f'border-radius:8px;border:2px solid rgba(255,255,255,0.15);display:block" '
                            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
                            f'<div style="width:80px;height:100px;border-radius:8px;'
                            f'background:rgba(0,0,0,0.25);display:none;align-items:center;'
                            f'justify-content:center;font-size:28px;font-weight:800;'
                            f'color:{club_fg};border:2px solid rgba(255,255,255,0.15)">{initials}</div>'
                            f'</div>'
                        )
                    else:
                        img_html = (
                            f'<div style="width:80px;height:100px;border-radius:8px;margin:0 auto 8px auto;'
                            f'background:rgba(0,0,0,0.25);display:flex;align-items:center;'
                            f'justify-content:center;font-size:28px;font-weight:800;'
                            f'color:{club_fg};opacity:0.7;'
                            f'border:2px solid rgba(255,255,255,0.15)">{initials}</div>'
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

                # Compact no-scroll layout: fixed-width columns, small cells, no photos
                n_cols  = len(matrix_gws)
                col_pct = f"{100 / n_cols:.2f}%"

                header = "".join(
                    f'<th style="background:#0d1117;color:#aaa;padding:6px 4px;'
                    f'border-bottom:2px solid #222;text-align:center;width:{col_pct};'
                    f'font-size:10px;font-weight:700;letter-spacing:.5px;'
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
                            name_w = "font-weight:800" if is_top else "font-weight:600"
                            eo_str = f"EO {r['EO%']}%" if r.get("EO%") is not None else ""
                            # Shorten name: last word or max 9 chars
                            parts = r["Name"].replace(".", " ").split()
                            short_name = parts[-1] if parts else r["Name"]
                            if len(short_name) > 9:
                                short_name = short_name[:8] + "…"
                            # Compact photo (24x30) or initials fallback
                            initials_m = "".join(w[0].upper() for w in r["Name"].replace(".", " ").split() if w)[:2] or "?"
                            if r.get("photo"):
                                img_tag = (
                                    f'<img src="{r["photo"]}" '
                                    f'style="width:24px;height:30px;object-fit:cover;object-position:top;'
                                    f'border-radius:3px;flex-shrink:0;vertical-align:middle" '
                                    f'onerror="this.style.display=\'none\'">'
                                )
                            else:
                                img_tag = (
                                    f'<span style="display:inline-flex;width:24px;height:30px;flex-shrink:0;'
                                    f'border-radius:3px;background:rgba(0,0,0,0.3);align-items:center;'
                                    f'justify-content:center;font-size:9px;font-weight:800;color:{fg}">{initials_m}</span>'
                                )

                            row_html += (
                                f'<td style="padding:4px 5px;border:1px solid #1a1a1a;'
                                f'background:{bg};vertical-align:middle;width:{col_pct}">'
                                f'<div style="display:flex;align-items:center;gap:4px">'
                                f'{img_tag}'
                                f'<div style="min-width:0;flex:1;overflow:hidden">'
                                f'<div style="color:{fg};{name_w};font-size:10.5px;'
                                f'font-family:sans-serif;white-space:nowrap;overflow:hidden;'
                                f'text-overflow:ellipsis;line-height:1.2">{short_name}</div>'
                                f'<div style="display:flex;align-items:center;gap:2px;margin-top:2px">'
                                f'<span style="background:{fdr_b};color:{fdr_f};border-radius:2px;'
                                f'padding:0px 3px;font-size:8.5px;font-weight:700;white-space:nowrap">{r["Fixture"]}</span>'
                                f'<span style="color:{fg};font-weight:800;font-size:10px;white-space:nowrap">{r["EV"]}</span>'
                                f'</div>'
                                f'<div style="color:{fg};font-size:8px;opacity:0.6;white-space:nowrap">{eo_str}</div>'
                                f'</div></div>'
                                f'</td>'
                            )
                        else:
                            row_html += f'<td style="border:1px solid #1a1a1a;background:#0a0a0a;width:{col_pct}"></td>'
                    body += f"<tr>{row_html}</tr>"

                html = (
                    f'<div style="border-radius:8px;border:1px solid #1e1e1e;margin-top:4px">'
                    f'<table style="border-collapse:collapse;width:100%;table-layout:fixed;background:#0d1117">'
                    f'<thead><tr style="background:#0d1117">{header}</tr></thead>'
                    f'<tbody>{body}</tbody>'
                    f'</table></div>'
                )
                st.markdown(html, unsafe_allow_html=True)
                st.caption("Top row = highest EV · Colour = club · FDR badge = custom ratings · EO = Solio expected ownership")


    # ── Tab 7: Cheatsheet ─────────────────────────────────────────────────────────
    # ── Tab 7: Cheatsheet ─────────────────────────────────────────────────────────
    with tab7:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">📋 Cheatsheet</span>'
            '<span style="font-size:12px;color:#444">GW strategy notes</span></div>',
            unsafe_allow_html=True
        )

        # Load from cheatsheet.json in same folder as app.py
        import json, os

        CHEATSHEET_FILE = "cheatsheet.json"

        def load_cheatsheet():
            if os.path.exists(CHEATSHEET_FILE):
                try:
                    with open(CHEATSHEET_FILE, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return {}
            return {}

        cs_data = load_cheatsheet()

        view_gw = st.selectbox(
            "Select Gameweek",
            options=list(range(1, 39)),
            index=current_gw - 1,
            key="cs_view_gw",
            label_visibility="collapsed",
        )

        gw_key   = f"gw{view_gw}"
        gw_notes = cs_data.get(gw_key, {})

        if not gw_notes:
            st.markdown(
                f'<div style="background:#111;border:1px dashed #222;border-radius:8px;'
                f'padding:32px;text-align:center;color:#444;font-size:14px">'
                f'No cheatsheet found for GW{view_gw}.<br>'
                f'<span style="font-size:12px;color:#333">Add a <code style="color:#555">gw{view_gw}</code> '
                f'entry in <code style="color:#555">cheatsheet.json</code></span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # Render each section
            SECTION_ICONS = {
                "disclaimer":  ("⚠️", "#2b1a00", "#1px solid #ffaa33", "#ffcc80"),
                "general":     ("📋", None, None, None),
                "transfers":   ("🔄", None, None, None),
                "holds":       ("⛔", None, None, None),
                "chips":       ("💊", None, None, None),
                "captaincy":   ("🎯", None, None, None),
                "notes":       ("📌", None, None, None),
            }

            html_parts = []

            # If there's a top-level disclaimer string, show it first
            if "disclaimer" in gw_notes:
                disc = gw_notes["disclaimer"]
                if isinstance(disc, str):
                    html_parts.append(
                        f'<div style="background:#2b1a00;border-left:3px solid #ffaa33;'
                        f'border-radius:4px;padding:10px 14px;color:#ffcc80;'
                        f'font-size:12px;margin-bottom:14px;line-height:1.6">{disc}</div>'
                    )

            # Render sections (skip disclaimer - already handled)
            SECTION_ORDER = ["general","transfers","holds","chips","captaincy","notes"]
            SECTION_LABELS = {
                "general":   "📋 General",
                "transfers": "🔄 Transfers",
                "holds":     "⛔ Holds",
                "chips":     "💊 Chips",
                "captaincy": "🎯 Captaincy",
                "notes":     "📌 Notes",
            }

            for section in SECTION_ORDER:
                if section not in gw_notes:
                    continue
                items = gw_notes[section]
                label = SECTION_LABELS.get(section, section.title())

                html_parts.append(
                    f'<div style="color:#5aabff;font-size:13px;font-weight:700;'
                    f'margin:14px 0 6px 0;letter-spacing:.3px">{label}</div>'
                )

                if isinstance(items, str):
                    items = [items]

                for item in items:
                    # Bold text between **...**
                    import re
                    item_html = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#fff">\1</strong>', item)
                    html_parts.append(
                        f'<div style="display:flex;gap:10px;padding:5px 0;'
                        f'color:#ccc;font-size:13px;line-height:1.5;border-bottom:1px solid #111">'
                        f'<span style="color:#333;flex-shrink:0;margin-top:1px">•</span>'
                        f'<span>{item_html}</span></div>'
                    )

            # Any extra string keys not in standard order
            extra_keys = [k for k in gw_notes if k not in ["disclaimer"] + SECTION_ORDER]
            for section in extra_keys:
                items = gw_notes[section]
                label = section.replace("_", " ").title()
                html_parts.append(
                    f'<div style="color:#5aabff;font-size:13px;font-weight:700;'
                    f'margin:14px 0 6px 0">{label}</div>'
                )
                if isinstance(items, str):
                    items = [items]
                for item in items:
                    html_parts.append(
                        f'<div style="display:flex;gap:10px;padding:5px 0;'
                        f'color:#ccc;font-size:13px;line-height:1.5;border-bottom:1px solid #111">'
                        f'<span style="color:#333;flex-shrink:0">•</span>'
                        f'<span>{item}</span></div>'
                    )

            card_html = (
                f'<div style="background:#0d1117;border:1px solid #1e1e1e;'
                f'border-radius:10px;padding:20px 22px;max-width:800px">'
                f'<div style="font-size:10px;color:#333;font-weight:700;'
                f'letter-spacing:.8px;margin-bottom:12px">GW{view_gw} CHEATSHEET</div>'
                + "".join(html_parts) +
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────────────────────
        if gw_notes:
            st.divider()
            export_body = card_html
            export_html_str = _export_html(f"GW{view_gw} Cheatsheet", export_body)
            st.download_button(
                label="📥 Download Cheatsheet (HTML)",
                data=export_html_str,
                file_name=f"cheatsheet_gw{view_gw}.html",
                mime="text/html",
                key="dl_cheatsheet",
            )

        # Show how-to hint

    # ── Tab 13: Differential Finder ──────────────────────────────────────────
    with tab13:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">🔍 Differential Finder</span>'
            '<span style="font-size:12px;color:#444">Low-ownership · high-projection picks</span></div>',
            unsafe_allow_html=True
        )

        if proj_df is None or eo_df is None:
            st.info("📂 Requires projections.csv and EO%.csv in the app folder.")
        elif not future_gws:
            st.info("No upcoming GW projections found.")
        else:
            dc1, dc2, dc3, dc4 = st.columns([1, 1, 1, 1])
            with dc1:
                diff_gw = st.selectbox("Gameweek", future_gws, key="diff_gw")
            with dc2:
                diff_pos = st.selectbox("Position", ["All", "GKP", "DEF", "MID", "FWD"], key="diff_pos")
            with dc3:
                diff_max_eo = st.slider("Max EO%", 1.0, 30.0, 10.0, 0.5, key="diff_max_eo")
            with dc4:
                diff_min_pts = st.slider("Min Proj Pts", 0.0, 12.0, 3.0, 0.5, key="diff_min_pts")

            pts_col_d = f"{diff_gw}_Pts"
            eo_col_d  = f"{diff_gw}_eo"

            if pts_col_d not in proj_df.columns:
                st.warning(f"No projections available for GW{diff_gw}.")
            elif eo_col_d not in eo_df.columns:
                st.warning(f"No EO% data available for GW{diff_gw}.")
            else:
                df_diff = proj_df[["ID", "Name", "Pos", "Team", "SV", pts_col_d]].copy()
                df_diff["ID"] = pd.to_numeric(df_diff["ID"], errors="coerce")
                df_diff[pts_col_d] = pd.to_numeric(df_diff[pts_col_d], errors="coerce")
                eo_map_d = pd.to_numeric(eo_df.set_index("ID")[eo_col_d], errors="coerce").to_dict()
                df_diff["EO%"] = (df_diff["ID"].map(eo_map_d) * 100).round(1)
                df_diff["SV"]  = pd.to_numeric(df_diff["SV"], errors="coerce")

                # Apply filters
                df_diff = df_diff[df_diff["EO%"] <= diff_max_eo]
                df_diff = df_diff[df_diff[pts_col_d] >= diff_min_pts]
                if diff_pos != "All":
                    df_diff = df_diff[df_diff["Pos"] == diff_pos]
                df_diff = df_diff.dropna(subset=[pts_col_d, "EO%"])
                df_diff = df_diff.sort_values(pts_col_d, ascending=False).reset_index(drop=True)

                st.markdown(
                    f'<div style="font-size:12px;color:#555;margin-bottom:8px">'
                    f'{len(df_diff)} differentials found · EO% ≤ {diff_max_eo}% · Proj pts ≥ {diff_min_pts}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if df_diff.empty:
                    st.info("No players match the current filters.")
                else:
                    # Build sortable styled HTML table
                    _tid = "tbl_diff"
                    _th_s = ("padding:8px 10px;font-size:11px;font-weight:700;letter-spacing:.5px;"
                             "border-bottom:2px solid #1e1e1e;cursor:pointer;user-select:none;"
                             "color:#666;white-space:nowrap")
                    def _dth(label, col_i, dtype="num", align="center"):
                        ta = "left" if align == "left" else "center"
                        return (f'<th style="{_th_s};text-align:{ta}" '
                                f'onclick="sortDiff({col_i},\'{dtype}\')">{label} ↕</th>')

                    rows_d = ""
                    for i, row in df_diff.head(60).iterrows():
                        bg, fg = club_style(row["Team"])
                        fdr_val = get_fdr_for_team_gw(row["Team"], diff_gw, master_df_full)
                        fdr_num = fdr_val[1] if isinstance(fdr_val, tuple) else 3
                        fdr_bg  = FDR_BG.get(fdr_num, "#444")
                        fdr_fg  = FDR_FG.get(fdr_num, "#fff")
                        fix_lbl = fdr_val[0] if isinstance(fdr_val, tuple) else "?"
                        eo_color = "#5fffb0" if row["EO%"] <= 5 else "#aaaaaa"
                        sv_val   = row["SV"]
                        rows_d += (
                            f'<tr onmouseover="this.style.background=\'#1a1a1a\'" onmouseout="this.style.background=\'transparent\'">'
                            f'<td style="padding:8px 10px;font-size:13px;font-weight:700;color:#e0e0e0;border-bottom:1px solid #141414;white-space:nowrap" data-val="{row["Name"]}">'
                            f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:3px;font-size:12px;font-weight:700;margin-right:8px">{row["Team"][:3].upper()}</span>'
                            f'{row["Name"]}</td>'
                            f'<td style="padding:8px 10px;text-align:center;color:#888;font-size:12px;border-bottom:1px solid #141414" data-val="{row["Pos"]}">{row["Pos"]}</td>'
                            f'<td style="padding:8px 10px;text-align:center;font-weight:800;font-size:14px;color:#5aabff;border-bottom:1px solid #141414" data-val="{row[pts_col_d]:.2f}">{row[pts_col_d]:.1f}</td>'
                            f'<td style="padding:8px 10px;text-align:center;font-weight:700;font-size:13px;color:{eo_color};border-bottom:1px solid #141414" data-val="{row["EO%"]:.2f}">{row["EO%"]:.1f}%</td>'
                            f'<td style="padding:8px 10px;text-align:center;border-bottom:1px solid #141414" data-val="{fdr_num}">'
                            f'<span style="background:{fdr_bg};color:{fdr_fg};padding:2px 7px;border-radius:3px;font-size:11px;font-weight:700">{fix_lbl}</span></td>'
                            f'<td style="padding:8px 10px;text-align:center;color:#aaa;font-size:12px;border-bottom:1px solid #141414" data-val="{sv_val:.2f}">£{sv_val:.1f}m</td>'
                            f'</tr>'
                        )

                    sort_js = f"""
<script>
(function(){{
  var _ds = {{}};
  window.sortDiff = function(col, dtype) {{
    var tbl = document.getElementById('{_tid}');
    if (!tbl) return;
    var tbody = tbl.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var key = col;
    var asc = _ds[key] !== true;
    _ds[key] = asc;
    rows.sort(function(a,b){{
      var av = a.cells[col] ? a.cells[col].getAttribute('data-val') : '';
      var bv = b.cells[col] ? b.cells[col].getAttribute('data-val') : '';
      if (dtype==='num'){{ av=parseFloat(av)||0; bv=parseFloat(bv)||0; return asc?av-bv:bv-av; }}
      return asc?av.localeCompare(bv):bv.localeCompare(av);
    }});
    var ths = tbl.querySelectorAll('thead th');
    ths.forEach(function(th,i){{
      th.innerHTML = th.innerHTML.replace(/ [↑↓↕]/g,'') + (i===col?(asc?' ↑':' ↓'):' ↕');
    }});
    rows.forEach(function(r){{ tbody.appendChild(r); }});
  }};
}})();
</script>"""

                    diff_table = (
                        f'<!DOCTYPE html><html><head>'
                        f'<style>body{{margin:0;background:#0d1117;}}table{{border-collapse:collapse;width:100%;font-family:Inter,Arial,sans-serif;background:#0d1117;}}th:hover{{color:#fff!important;}}</style>'
                        f'</head><body>'
                        f'<div style="overflow-x:auto">'
                        f'<table id="{_tid}" style="border-collapse:collapse;width:100%;background:#0d1117">'
                        f'<thead><tr style="background:#161b22">'
                        + _dth("PLAYER", 0, "str", "left")
                        + _dth("POS", 1, "str")
                        + _dth("PROJ PTS", 2, "num")
                        + _dth("EO%", 3, "num")
                        + _dth("FIXTURE", 4, "num")
                        + _dth("PRICE", 5, "num") +
                        f'</tr></thead><tbody>{rows_d}</tbody></table></div>'
                        f'{sort_js}</body></html>'
                    )
                    _tbl_h = 55 + min(len(df_diff), 60) * 42
                    components.html(diff_table, height=_tbl_h, scrolling=False)
                    st.caption("Click any column header to sort · EO% = Solio expected ownership · Green EO% = ≤ 5%")

                    # Export (plain table, no JS wrapper)
                    _export_table = (
                        f'<div style="overflow-x:auto">'
                        f'<table style="border-collapse:collapse;width:100%;background:#0d1117;font-family:sans-serif">'
                        f'<thead><tr style="background:#161b22">'
                        f'<th style="padding:8px 10px;text-align:left;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">PLAYER</th>'
                        f'<th style="padding:8px 10px;text-align:center;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">POS</th>'
                        f'<th style="padding:8px 10px;text-align:center;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">PROJ PTS</th>'
                        f'<th style="padding:8px 10px;text-align:center;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">EO%</th>'
                        f'<th style="padding:8px 10px;text-align:center;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">FIXTURE</th>'
                        f'<th style="padding:8px 10px;text-align:center;color:#555;font-size:11px;font-weight:700;border-bottom:2px solid #1e1e1e">PRICE</th>'
                        f'</tr></thead><tbody>{rows_d}</tbody></table></div>'
                    )
                    st.divider()
                    export_diff_html = _export_html(
                        f"GW{diff_gw} Differentials (EO% ≤ {diff_max_eo}%)",
                        _export_table
                    )
                    st.download_button(
                        label="📥 Download Differentials (HTML)",
                        data=export_diff_html,
                        file_name=f"differentials_gw{diff_gw}.html",
                        mime="text/html",
                        key="dl_differentials",
                    )

# ────────────────────────────────────────────────────────────────────────────
elif nav_cat == "👕 My FPL":
    with tab8:
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



    # ── Tab 9: My Team ─────────────────────────────────────────────────────────────
    with tab9:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">👕 My Team</span>'
            '<span style="font-size:12px;color:#444">Live squad · fixtures · EV · captaincy</span></div>',
            unsafe_allow_html=True
        )

        if not live_ok:
            st.warning("⚠️ Enable Live FPL Data in the sidebar.")
        else:
            col_tid, col_tgw = st.columns([2, 1])
            with col_tid:
                team_id_input = st.number_input(
                    "FPL Team ID", min_value=1, max_value=99999999,
                    value=int(st.session_state.get("fpl_team_id", 1)),
                    step=1, key="fpl_team_id_input",
                    help="Find your ID at: fantasy.premierleague.com/entry/{ID}/event/..."
                )
            with col_tgw:
                avail_gws = future_gws if future_gws else list(range(current_gw, 39))
                team_gw   = st.selectbox("View GW fixtures", avail_gws, key="myteam_gw")

            if st.button("🔄 Load Team", key="load_team_btn"):
                st.session_state["fpl_team_id"] = team_id_input
                for k in ["squad_data", "entry_data", "picks_raw"]:
                    st.session_state.pop(k, None)

            team_id = st.session_state.get("fpl_team_id")

            # Manual squad fallback
            with st.expander("✏️ Pick squad manually (if Team ID unavailable)", expanded=False):
                all_el    = pd.DataFrame(bootstrap["elements"])
                teams_bdf = pd.DataFrame(bootstrap["teams"])
                id2sname  = dict(zip(teams_bdf["id"], teams_bdf["short_name"]))
                all_el["label"] = all_el["web_name"] + " (" + all_el["team"].map(id2sname) + ")"
                opts = all_el.sort_values("web_name")["label"].tolist()
                pid_map = dict(zip(all_el["label"], all_el["id"]))

                man_xi    = st.multiselect("Starting XI (11)", opts, max_selections=11, key="man_xi")
                man_bench = st.multiselect("Bench (4)", opts, max_selections=4, key="man_bench")

                if st.button("✅ Use Manual Squad", key="use_manual_sq"):
                    el_map_m = {p["id"]: p for p in bootstrap["elements"]}
                    POS_M    = {1:"GKP",2:"DEF",3:"MID",4:"FWD"}
                    t_short  = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
                    msq = []
                    for idx, lbl in enumerate(man_xi + man_bench, 1):
                        pid = pid_map.get(lbl)
                        if not pid: continue
                        el  = el_map_m.get(pid, {})
                        msq.append({
                            "id": pid, "name": el.get("web_name","?"),
                            "team": t_short.get(el.get("team"),"?"),
                            "pos": POS_M.get(el.get("element_type"),"?"),
                            "price": el.get("now_cost",0)/10,
                            "sel%": el.get("selected_by_percent","0"),
                            "position": idx, "multiplier": 1,
                            "is_captain": idx==1, "is_vice": idx==2,
                            "code": el.get("code"),
                        })
                    if msq:
                        st.session_state["squad_data"] = msq
                        st.success("Manual squad loaded!")

            # Auto-load from FPL API
            # Picks for GW N are only available AFTER GW N deadline.
            # Before the deadline → fetch from the last finished GW.
            if team_id and "squad_data" not in st.session_state:
                try:
                    with st.spinner(f"Loading FPL team {team_id}..."):
                        from datetime import datetime, timezone
                        entry_d = fetch_fpl_entry(int(team_id))
                        now_utc = datetime.now(timezone.utc)
                        events  = bootstrap.get("events", [])

                        picks_gw       = current_gw
                        picks_gw_label = f"GW{current_gw}"
                        for ev in events:
                            if ev["id"] == current_gw:
                                dl_raw = ev.get("deadline_time", "")
                                if dl_raw:
                                    try:
                                        dl_utc = datetime.fromisoformat(dl_raw.replace("Z", "+00:00"))
                                        if now_utc < dl_utc:
                                            finished_gws = sorted(
                                                [e["id"] for e in events if e.get("finished")],
                                                reverse=True
                                            )
                                            if finished_gws:
                                                picks_gw = finished_gws[0]
                                                picks_gw_label = f"GW{picks_gw} (last confirmed squad — GW{current_gw} deadline not yet passed)"
                                    except Exception:
                                        pass
                                break

                        picks_d  = fetch_fpl_picks(int(team_id), picks_gw)
                        history_d = fetch_fpl_history(int(team_id))
                        squad_d  = build_squad_from_picks(picks_d, bootstrap)
                        st.session_state["squad_data"]     = squad_d
                        st.session_state["entry_data"]     = entry_d
                        st.session_state["picks_raw"]      = picks_d
                        st.session_state["history_data"]   = history_d
                        st.session_state["picks_gw_label"] = picks_gw_label
                except Exception as exc:
                    st.error(f"Could not load team {team_id}: {exc}")

            if "squad_data" in st.session_state:
                squad = enrich_squad_solio(
                    list(st.session_state["squad_data"]), proj_df, eo_df, team_gw
                )

                # Manager banner
                picks_gw_label = st.session_state.get("picks_gw_label", f"GW{current_gw}")
                if f"(last confirmed" in picks_gw_label:
                    st.info(f"⚠️ Showing **{picks_gw_label}** — update your squad after the deadline passes.")

                if "entry_data" in st.session_state:
                    ed       = st.session_state["entry_data"]
                    pr       = st.session_state.get("picks_raw", {})
                    eh       = pr.get("entry_history", {})
                    manager  = ed.get("name", f"Team {team_id}")
                    rank     = ed.get("summary_overall_rank", "—")
                    pts_tot  = ed.get("summary_overall_points", "—")
                    bank_val = eh.get("bank", 0) / 10
                    tv_val   = eh.get("value", 0) / 10
                    # Chips played come from the history endpoint (entry endpoint doesn't reliably include them)
                    rank_fmt = f"{rank:,}" if isinstance(rank, int) else str(rank)
                    st.markdown(
                        f'<div style="background:#0d1117;border:1px solid #1e1e1e;border-radius:8px;'
                        f'padding:12px 16px;margin-bottom:12px;display:flex;gap:24px;flex-wrap:wrap;align-items:center">'
                        f'<div><div style="font-size:16px;font-weight:800;color:#e0e0e0">{manager}</div>'
                        f'<div style="font-size:11px;color:#555">Rank {rank_fmt} · {pts_tot} pts</div></div>'
                        f'<div><div style="font-size:11px;color:#555">Team Value</div>'
                        f'<div style="font-size:14px;font-weight:700;color:#5fffb0">£{tv_val:.1f}m</div></div>'
                        f'<div><div style="font-size:11px;color:#555">Bank</div>'
                        f'<div style="font-size:14px;font-weight:700;color:#ffaa33">£{bank_val:.1f}m</div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                xi    = sorted([p for p in squad if p["position"] <= 11], key=lambda p: p["position"])
                bench = sorted([p for p in squad if p["position"] >  11], key=lambda p: p["position"])

                # Captain recommendation
                xi_ev = [p for p in xi if p.get("ev")]
                if xi_ev:
                    top_cap  = max(xi_ev, key=lambda p: p["ev"])
                    diff_cap = next((p for p in sorted(xi_ev, key=lambda p: -p["ev"])
                                     if (p.get("eo%") or 999) < 10), None)
                    cap_rec = (
                        f'<div style="background:#0a1a0a;border:1px solid #1e3a1e;border-radius:8px;'
                        f'padding:10px 14px;margin-bottom:10px;display:flex;gap:24px;flex-wrap:wrap">'
                        f'<div><div style="font-size:10px;color:#555;font-weight:700;letter-spacing:.5px">CAPTAIN REC</div>'
                        f'<div style="font-size:15px;font-weight:800;color:#5fffb0;margin-top:2px">🏆 {top_cap["name"]}</div>'
                        f'<div style="font-size:11px;color:#888">{top_cap["ev"]} pts · EO {top_cap.get("eo%","—")}%</div></div>'
                    )
                    if diff_cap and diff_cap["id"] != top_cap["id"]:
                        cap_rec += (
                            f'<div><div style="font-size:10px;color:#555;font-weight:700;letter-spacing:.5px">DIFFERENTIAL</div>'
                            f'<div style="font-size:15px;font-weight:800;color:#ffaa33;margin-top:2px">🎯 {diff_cap["name"]}</div>'
                            f'<div style="font-size:11px;color:#888">{diff_cap["ev"]} pts · EO {diff_cap.get("eo%","—")}%</div></div>'
                        )
                    cap_rec += '</div>'
                    st.markdown(cap_rec, unsafe_allow_html=True)

                # Build position groups sorted by position number
                pos_order = {"GKP":0,"DEF":1,"MID":2,"FWD":3}
                xi_by_pos = {"GKP":[],"DEF":[],"MID":[],"FWD":[]}
                for p in sorted(xi, key=lambda p: (pos_order.get(p["pos"],4), p["position"])):
                    xi_by_pos[p["pos"]].append(p)

                pitch_html = _render_pitch(xi_by_pos, team_gw, master_df_full, bench_players=bench)
                st.markdown(pitch_html, unsafe_allow_html=True)


    # ── Tab 10: GW Planner ────────────────────────────────────────────────────────
    with tab10:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">📅 GW Planner</span>'
            '<span style="font-size:12px;color:#444">Projected XI · transfer suggestions</span></div>',
            unsafe_allow_html=True
        )

        if not live_ok:
            st.warning("⚠️ Enable Live FPL Data in the sidebar.")
        elif proj_df is None:
            st.info("📂 projections.csv not found — required for GW Planner.")
        elif "squad_data" not in st.session_state:
            st.info("👕 Load your team in the **My Team** tab first.")
        else:
            avail_gws_p = future_gws if future_gws else list(range(current_gw, 39))
            plan_gw     = st.selectbox("Plan for GW:", avail_gws_p, key="plan_gw")
            pts_col_p   = f"{plan_gw}_Pts"
            mins_col_p  = f"{plan_gw}_xMins"
            eo_col_p    = f"{plan_gw}_eo"

            squad_p = enrich_squad_solio(
                list(st.session_state["squad_data"]), proj_df, eo_df, plan_gw
            )
            xi_p    = sorted([p for p in squad_p if p["position"] <= 11], key=lambda p: p["position"])

            # ── Projected XI ──────────────────────────────────────────────────────
            st.markdown("#### 🔢 Projected XI")

            if pts_col_p not in proj_df.columns:
                st.warning(f"No Solio projections available for GW{plan_gw} yet.")
            else:
                # Auto-build best XI by EV respecting formation rules
                def _best_xi(players):
                    POS_MIN = {"GKP":1,"DEF":3,"MID":2,"FWD":1}
                    POS_MAX = {"GKP":1,"DEF":5,"MID":5,"FWD":3}
                    selected, counts = [], {"GKP":0,"DEF":0,"MID":0,"FWD":0}
                    for pos in ["GKP","DEF","MID","FWD"]:
                        pool = sorted([p for p in players if p["pos"]==pos], key=lambda p: p.get("ev") or 0, reverse=True)
                        for p in pool[:POS_MIN[pos]]:
                            selected.append(p); counts[pos] += 1
                    remaining = sorted([p for p in players if p not in selected], key=lambda p: p.get("ev") or 0, reverse=True)
                    for p in remaining:
                        if len(selected) >= 11: break
                        if counts[p["pos"]] < POS_MAX[p["pos"]]:
                            selected.append(p); counts[p["pos"]] += 1
                    return selected

                sug_xi    = _best_xi(squad_p)
                cap_p     = max(sug_xi, key=lambda p: p.get("ev") or 0) if sug_xi else None
                total_ev  = sum(p.get("ev") or 0 for p in sug_xi)
                cap_bonus = (cap_p["ev"] if cap_p and cap_p.get("ev") else 0)
                total_cap = total_ev + cap_bonus  # captain doubles

                # Summary card
                cap_rec_html = ""
                if cap_p:
                    cap_rec_html = (
                        f'<div><div style="font-size:10px;color:#555;font-weight:700">CAPTAIN REC</div>'
                        f'<div style="font-size:16px;font-weight:800;color:#fff;margin-top:2px">'
                        f'{cap_p["name"]} ({cap_p["ev"]} pts)</div></div>'
                    )
                st.markdown(
                    '<div style="background:#0a1a0a;border:1px solid #1e3a1e;border-radius:8px;'
                    'padding:12px 16px;margin-bottom:12px;display:flex;gap:24px;flex-wrap:wrap">'
                    '<div><div style="font-size:10px;color:#555;font-weight:700">PROJECTED PTS (XI)</div>'
                    f'<div style="font-size:22px;font-weight:800;color:#5fffb0">{total_ev:.1f}</div></div>'
                    '<div><div style="font-size:10px;color:#555;font-weight:700">WITH CAPTAIN</div>'
                    f'<div style="font-size:22px;font-weight:800;color:#ffaa33">{total_cap:.1f}</div></div>'
                    + cap_rec_html + '</div>',
                    unsafe_allow_html=True
                )

                # Build best XI position groups, mark captain
                pos_ord   = {"GKP":0,"DEF":1,"MID":2,"FWD":3}
                xi_by_pos_p = {"GKP":[],"DEF":[],"MID":[],"FWD":[]}
                for p in sorted(sug_xi, key=lambda p: (pos_ord.get(p["pos"],4), -(p.get("ev") or 0))):
                    p["is_captain"] = (p == cap_p)
                    p["is_vice"]    = False
                    xi_by_pos_p[p["pos"]].append(p)
                bench_sug = sorted([p for p in squad_p if p not in sug_xi], key=lambda p: -(p.get("ev") or 0))
                for p in bench_sug:
                    p["is_captain"] = False
                    p["is_vice"]    = False

                pitch_html_p = _render_pitch(xi_by_pos_p, plan_gw, master_df_full, bench_players=bench_sug)
                st.markdown(pitch_html_p, unsafe_allow_html=True)

            # ── Transfer Suggestions ───────────────────────────────────────────────
            st.markdown("#### 🔄 Transfer Suggestions")

            if pts_col_p not in proj_df.columns:
                st.warning(f"No Solio projections for GW{plan_gw}.")
            else:
                ft_opt    = st.radio("Free transfers:", ["1 FT", "2 FTs", "3 FTs", "4 FTs", "5 FTs"],
                                     index=0, horizontal=True, key="plan_ft")
                n_suggest = {"1 FT":1,"2 FTs":2,"3 FTs":3,"4 FTs":4,"5 FTs":5}[ft_opt]
                avail_ft  = n_suggest  # no automatic detection; user selects

                squad_ids = {p["id"] for p in squad_p}
                df_out    = proj_df.copy()
                df_out[pts_col_p]  = pd.to_numeric(df_out[pts_col_p],  errors="coerce")
                df_out[mins_col_p] = pd.to_numeric(df_out[mins_col_p], errors="coerce")
                df_out["SV"]       = pd.to_numeric(df_out["SV"],       errors="coerce")

                outside = df_out[(df_out[mins_col_p] > 45) & (~df_out["ID"].isin(squad_ids))].copy()
                if eo_df is not None and eo_col_p in eo_df.columns:
                    eo_map_t = pd.to_numeric(eo_df.set_index("ID")[eo_col_p], errors="coerce").to_dict()
                    outside["EO%"] = outside["ID"].map(eo_map_t).fillna(0) * 100
                else:
                    outside["EO%"] = 0

                pos_map = {"GKP":"G","DEF":"D","MID":"M","FWD":"F"}
                # For more FTs, consider more sell candidates (bench included if ≥3 FTs)
                pool_xi    = sorted([p for p in squad_p if p["position"] <= 11], key=lambda p: p.get("ev") or 0)
                pool_bench = sorted([p for p in squad_p if p["position"] > 11],  key=lambda p: p.get("ev") or 0)
                sell_pool  = pool_xi[:max(n_suggest + 2, 6)] if n_suggest < 3 else (pool_xi + pool_bench)[:max(n_suggest + 3, 8)]
                xi_worst   = sell_pool  # alias for backwards compat

                suggestions = []
                used_buy_ids = set()  # prevent the same player being suggested as buy twice
                for sell in xi_worst[:max(n_suggest * 2, 8)]:
                    if len(suggestions) >= n_suggest:
                        break
                    sell_ev    = sell.get("ev") or 0
                    sell_price = sell["price"]
                    sell_pos   = sell["pos"]
                    same_pos   = outside[
                        (outside["Pos"].str.upper().str[0] == pos_map.get(sell_pos, sell_pos[0])) &
                        (~outside["ID"].isin(used_buy_ids))
                    ]
                    in_budget  = same_pos[same_pos["SV"] <= sell_price + 0.1]
                    if in_budget.empty:
                        in_budget = same_pos
                    if in_budget.empty:
                        continue
                    best_buy = in_budget.nlargest(1, pts_col_p)
                    if best_buy.empty: continue
                    buy_row  = best_buy.iloc[0]
                    buy_ev   = round(float(buy_row[pts_col_p]), 2)
                    gain     = round(buy_ev - sell_ev, 2)
                    if gain <= 0: continue
                    used_buy_ids.add(buy_row["ID"])
                    suggestions.append({
                        "out_name": sell["name"], "out_team": sell["team"], "out_ev": sell_ev,
                        "in_name":  buy_row["Name"], "in_team": buy_row["Team"],
                        "in_ev": buy_ev, "gain": gain,
                        "in_eo": round(float(buy_row.get("EO%", 0)), 1),
                        "in_price": buy_row["SV"],
                    })

                if not suggestions:
                    st.success("✅ Your squad looks optimal — no clear upgrades found.")
                else:
                    suggestions = sorted(suggestions, key=lambda x: -x["gain"])[:n_suggest]
                    th_s = 'style="padding:7px 10px;text-align:center;color:#444;font-size:10px;font-weight:700;letter-spacing:.5px;border-bottom:2px solid #222"'
                    rows_sg = ""
                    for sg in suggestions:
                        ob, of = club_style(sg["out_team"])
                        ib, if_ = club_style(sg["in_team"])
                        gc = "#5fffb0" if sg["gain"] > 0 else "#ff6060"
                        rows_sg += (
                            f'<tr>'
                            f'<td style="padding:7px 10px;border-bottom:1px solid #1a1a1a">'
                            f'<span style="background:{ob};color:{of};padding:2px 7px;border-radius:3px;font-size:12px;font-weight:700">{sg["out_name"]}</span>'
                            f'<span style="color:#444;padding:0 6px">→</span>'
                            f'<span style="background:{ib};color:{if_};padding:2px 7px;border-radius:3px;font-size:12px;font-weight:700">{sg["in_name"]}</span>'
                            f'</td>'
                            f'<td style="padding:7px 10px;text-align:center;color:#888;font-size:12px;border-bottom:1px solid #1a1a1a">{sg["out_ev"]} → {sg["in_ev"]}</td>'
                            f'<td style="padding:7px 10px;text-align:center;font-weight:800;font-size:13px;color:{gc};border-bottom:1px solid #1a1a1a">+{sg["gain"]}</td>'
                            f'<td style="padding:7px 10px;text-align:center;color:#666;font-size:11px;border-bottom:1px solid #1a1a1a">{sg["in_eo"]}%</td>'
                            f'<td style="padding:7px 10px;text-align:center;color:#888;font-size:11px;border-bottom:1px solid #1a1a1a">£{sg["in_price"]:.1f}m</td>'
                            f'</tr>'
                        )
                    st.markdown(
                        f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;background:#0d1117;font-family:sans-serif">'
                        f'<thead><tr>'
                        f'<th {th_s} style="text-align:left">TRANSFER</th>'
                        f'<th {th_s}>EV OUT→IN</th><th {th_s}>EV GAIN</th>'
                        f'<th {th_s}>EO%</th><th {th_s}>PRICE</th>'
                        f'</tr></thead><tbody>{rows_sg}</tbody></table></div>',
                        unsafe_allow_html=True
                    )

    # ── Tab 14: Mini-League Tracker ──────────────────────────────────────────
    with tab14:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">🏆 Mini-League</span>'
            '<span style="font-size:12px;color:#444">Standings · rank movements</span></div>',
            unsafe_allow_html=True
        )

        if not live_ok:
            st.warning("⚠️ Enable Live FPL Data in the sidebar.")
        else:
            ml_col1, ml_col2 = st.columns([3, 1])
            with ml_col1:
                ml_id_input = st.number_input(
                    "Mini-League ID",
                    min_value=1, max_value=9999999,
                    value=int(st.session_state.get("ml_league_id", 314)),
                    step=1, key="ml_id_input",
                    help="Find the ID in the URL: fantasy.premierleague.com/leagues/{ID}/standings/c"
                )
            with ml_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                load_ml = st.button("🔍 Load", key="load_ml_btn", use_container_width=True)

            if load_ml:
                st.session_state["ml_league_id"] = ml_id_input
                fetch_mini_league.clear()

            ml_league_id = st.session_state.get("ml_league_id")

            if ml_league_id:
                try:
                    ml_data     = fetch_mini_league(int(ml_league_id))
                    league_info = ml_data.get("league", {})
                    league_name = league_info.get("name", f"League {ml_league_id}")
                    results     = ml_data.get("standings", {}).get("results", [])

                    st.markdown(
                        f'<div style="font-size:16px;font-weight:700;color:#5fffb0;margin-bottom:12px">'
                        f'{league_name}</div>',
                        unsafe_allow_html=True
                    )

                    if not results:
                        st.info("No standings data found for this league.")
                    else:
                        th_ml = 'style="padding:8px 10px;color:#555;font-size:11px;font-weight:700;letter-spacing:.5px;border-bottom:2px solid #1e1e1e;text-align:center"'
                        th_ml_l = 'style="padding:8px 10px;color:#555;font-size:11px;font-weight:700;letter-spacing:.5px;border-bottom:2px solid #1e1e1e;text-align:left"'
                        rows_ml = ""
                        for entry in results:
                            rank      = entry.get("rank", "-")
                            last_rank = entry.get("last_rank", rank)
                            try:
                                move = int(last_rank) - int(rank)
                            except Exception:
                                move = 0
                            if move > 0:
                                arrow = f'<span style="color:#5fffb0;font-size:11px">▲{move}</span>'
                            elif move < 0:
                                arrow = f'<span style="color:#ff6060;font-size:11px">▼{abs(move)}</span>'
                            else:
                                arrow = '<span style="color:#444;font-size:11px">–</span>'

                            manager   = entry.get("player_name", "?")
                            team_name = entry.get("entry_name", "?")
                            gw_pts    = entry.get("event_total", 0)
                            total_pts = entry.get("total", 0)

                            rank_color = "#ffd700" if rank == 1 else ("#c0c0c0" if rank == 2 else ("#cd7f32" if rank == 3 else "#e0e0e0"))

                            rows_ml += (
                                f'<tr style="border-bottom:1px solid #141414">'
                                f'<td style="padding:8px 10px;text-align:center;font-weight:800;font-size:15px;color:{rank_color}">{rank}</td>'
                                f'<td style="padding:8px 4px;text-align:center">{arrow}</td>'
                                f'<td style="padding:8px 10px;font-size:13px;font-weight:600;color:#e0e0e0">{team_name}<br>'
                                f'<span style="font-size:11px;color:#555;font-weight:400">{manager}</span></td>'
                                f'<td style="padding:8px 10px;text-align:center;font-weight:800;font-size:14px;color:#5aabff">{gw_pts}</td>'
                                f'<td style="padding:8px 10px;text-align:center;font-weight:700;font-size:13px;color:#aaa">{total_pts}</td>'
                                f'</tr>'
                            )

                        ml_table = (
                            f'<div style="overflow-x:auto">'
                            f'<table style="border-collapse:collapse;width:100%;background:#0d1117;font-family:sans-serif">'
                            f'<thead><tr style="background:#161b22">'
                            f'<th {th_ml}>RANK</th>'
                            f'<th {th_ml}></th>'
                            f'<th {th_ml_l}>TEAM / MANAGER</th>'
                            f'<th {th_ml}>GW PTS</th>'
                            f'<th {th_ml}>TOTAL</th>'
                            f'</tr></thead><tbody>{rows_ml}</tbody></table></div>'
                        )
                        st.markdown(ml_table, unsafe_allow_html=True)
                        has_next = ml_data.get("standings", {}).get("has_next", False)
                        if has_next:
                            st.caption("Showing first 50 entries · FPL API paginates at 50")

                        # Export
                        st.divider()
                        export_ml_html = _export_html(f"{league_name} Standings", ml_table)
                        st.download_button(
                            label="📥 Download Standings (HTML)",
                            data=export_ml_html,
                            file_name=f"mini_league_{ml_league_id}.html",
                            mime="text/html",
                            key="dl_mini_league",
                        )

                except Exception as e:
                    st.error(f"Could not load league {ml_league_id}: {e}")
                    st.caption("Check the league ID and ensure it's a Classic league (not H2H).")



# ────────────────────────────────────────────────────────────────────────────
else:  # 🏟️ Stats
    with tab11:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Team Stats</span>'
            '<span style="font-size:12px;color:#444">Premier League 2025/26</span></div>',
            unsafe_allow_html=True
        )

        ts_df = load_csv_team_stats()

        if ts_df.empty:
            st.info("No team stats available.")
        else:
            st.dataframe(
                ts_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Team":    st.column_config.TextColumn("Team"),
                    "xG":      st.column_config.NumberColumn("xG", format="%.1f"),
                    "xGC":     st.column_config.NumberColumn("xGC", format="%.1f"),
                    "xGDiff":  st.column_config.NumberColumn("xG Diff", format="%.1f"),
                },
            )
            st.caption("Click any column header to sort · xGC = xG conceded · xG Diff = xG − xGC")


    # =============================================================================
    # TAB 9 — Player Stats
    # =============================================================================
    with tab12:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px">'
            '<span style="font-size:18px;font-weight:700;color:#e0e0e0">Player Stats</span>'
            '<span style="font-size:12px;color:#444">Premier League 2025/26</span></div>',
            unsafe_allow_html=True
        )

        ps_df = load_csv_player_stats()
        ps_df = add_fpl_positions(ps_df, bootstrap)

        # ── Primary filters ────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            name_f = st.text_input("Search player:", placeholder="e.g. Salah", key="csv_ps_name")
        with fc2:
            club_opts = ["All"] + sorted(ps_df["Team"].dropna().unique().tolist())
            club_f = st.selectbox("Club:", club_opts, key="csv_ps_club")
        with fc3:
            pos_f = st.selectbox("Position:", ["All", "GKP", "DEF", "MID", "FWD"], key="csv_ps_pos")

        # ── Numeric column filters (expander) ──────────────────────────────────────
        num_cols = ["MP", "Mins", "Goals", "Assists", "G+A", "G/90",
                    "Sh/90", "SoT/90", "BCC", "CC", "BCM",
                    "xG", "xG/90", "xGOT", "xA", "xA/90", "xG+xA/90"]
        num_cols = [c for c in num_cols if c in ps_df.columns]

        slider_vals = {}
        with st.expander("Column filters", expanded=False):
            exp_cols = st.columns(3)
            for i, col in enumerate(num_cols):
                col_min = float(ps_df[col].min())
                col_max = float(ps_df[col].max())
                with exp_cols[i % 3]:
                    fmt = "%.0f" if col in ("MP", "Mins", "Goals", "Assists", "G+A", "BCC", "CC", "BCM") else "%.2f"
                    slider_vals[col] = st.slider(
                        col, col_min, col_max, (col_min, col_max),
                        format=fmt, key=f"csv_ps_sl_{col}"
                    )

        # ── Apply all filters ──────────────────────────────────────────────────────
        filt_ps = ps_df.copy()
        if name_f:
            filt_ps = filt_ps[filt_ps["Player"].str.contains(name_f, case=False, na=False)]
        if club_f != "All":
            filt_ps = filt_ps[filt_ps["Team"] == club_f]
        if pos_f != "All":
            filt_ps = filt_ps[filt_ps["Pos"] == pos_f]
        for col, (lo, hi) in slider_vals.items():
            filt_ps = filt_ps[(filt_ps[col] >= lo) & (filt_ps[col] <= hi)]
        filt_ps = filt_ps.reset_index(drop=True)

        # Build external links columns from id map
        def _safe_eid(eid):
            try:
                return int(eid) if eid is not None and eid == eid else None
            except (ValueError, TypeError):
                return None

        def _tm_url(eid, player_name):
            sid = _safe_eid(eid)
            if sid is not None and sid in _ext_links and _ext_links[sid].get("transfermarkt"):
                return f"https://www.transfermarkt.com/-/profil/spieler/{_ext_links[sid]['transfermarkt']}"
            # Fallback: TM search by name
            import urllib.parse
            return f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={urllib.parse.quote(str(player_name))}"

        filt_ps["Understat"] = filt_ps.apply(
            lambda row: (f"https://understat.com/player/{_ext_links[_safe_eid(row['_fpl_id'])]['understat']}"
                         if _safe_eid(row["_fpl_id"]) is not None
                         and _safe_eid(row["_fpl_id"]) in _ext_links
                         and _ext_links[_safe_eid(row["_fpl_id"])].get("understat")
                         else None), axis=1
        )
        filt_ps["FBRef"] = filt_ps.apply(
            lambda row: (f"https://fbref.com/en/players/{_ext_links[_safe_eid(row['_fpl_id'])]['fbref']}/"
                         if _safe_eid(row["_fpl_id"]) is not None
                         and _safe_eid(row["_fpl_id"]) in _ext_links
                         and _ext_links[_safe_eid(row["_fpl_id"])].get("fbref")
                         else None), axis=1
        )
        filt_ps["Transfermarkt"] = filt_ps.apply(
            lambda row: _tm_url(row["_fpl_id"], row["Player"]), axis=1
        )
        filt_ps["FPL"] = filt_ps["_fpl_id"].apply(
            lambda eid: (f"https://fantasy.premierleague.com/api/element-summary/{_safe_eid(eid)}/"
                         if _safe_eid(eid) is not None else None)
        )

        col_order = ["Player", "Team", "Country", "Pos", "MP", "Mins",
                     "Goals", "Assists", "G+A", "G/90",
                     "Sh/90", "SoT/90", "BCC", "CC", "BCM",
                     "xG", "xG/90", "xGOT", "xA", "xA/90", "xG+xA/90",
                     "Understat", "FBRef", "Transfermarkt", "FPL"]
        show_cols = [c for c in col_order if c in filt_ps.columns]

        st.dataframe(
            filt_ps[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Player":        st.column_config.TextColumn("Player"),
                "Team":          st.column_config.TextColumn("Team"),
                "Country":       st.column_config.TextColumn("Country"),
                "Pos":           st.column_config.TextColumn("Pos"),
                "MP":            st.column_config.NumberColumn("MP"),
                "Mins":          st.column_config.NumberColumn("Mins"),
                "Goals":         st.column_config.NumberColumn("Goals"),
                "Assists":       st.column_config.NumberColumn("Assists"),
                "G+A":           st.column_config.NumberColumn("G+A"),
                "G/90":          st.column_config.NumberColumn("G/90", format="%.2f"),
                "Sh/90":         st.column_config.NumberColumn("Sh/90", format="%.2f"),
                "SoT/90":        st.column_config.NumberColumn("SoT/90", format="%.2f"),
                "BCC":           st.column_config.NumberColumn("BCC"),
                "CC":            st.column_config.NumberColumn("CC"),
                "BCM":           st.column_config.NumberColumn("BCM"),
                "xG":            st.column_config.NumberColumn("xG", format="%.1f"),
                "xG/90":         st.column_config.NumberColumn("xG/90", format="%.2f"),
                "xGOT":          st.column_config.NumberColumn("xGOT", format="%.1f"),
                "xA":            st.column_config.NumberColumn("xA", format="%.1f"),
                "xA/90":         st.column_config.NumberColumn("xA/90", format="%.2f"),
                "xG+xA/90":      st.column_config.NumberColumn("xG+xA/90", format="%.2f"),
                "Understat":     st.column_config.LinkColumn("Understat", display_text="US ↗"),
                "FBRef":         st.column_config.LinkColumn("FBRef", display_text="FB ↗"),
                "Transfermarkt": st.column_config.LinkColumn("Transfermarkt", display_text="TM ↗"),
                "FPL":           st.column_config.LinkColumn("FPL", display_text="FPL ↗"),
            },
        )
        st.caption(
            f"{len(filt_ps)} players · Click any column header to sort · "
            "BCC = big chances created · BCM = big chances missed · xGOT = xG on target · "
            "US = Understat · FB = FBRef · TM = Transfermarkt · FPL = FPL API stats"
        )


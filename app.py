import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import math
import requests

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
    return 1

TEAM_NAME_MAP = {
    "A.F.C. Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton", "Brighton": "Brighton",
    "Ipswich Town": "Ipswich", "Ipswich": "Ipswich",
    "Leeds United": "Leeds", "Leeds": "Leeds",
    "Leicester City": "Leicester", "Leicester": "Leicester",
    "Manchester City": "Man City", "Man City": "Man City",
    "Manchester United": "Man Utd", "Man Utd": "Man Utd",
    "Newcastle United": "Newcastle", "Newcastle": "Newcastle",
    "Nottingham Forest": "Nottm Forest", "Nottm Forest": "Nottm Forest",
    "Southampton FC": "Southampton", "Southampton": "Southampton",
    "Tottenham Hotspur": "Spurs", "Spurs": "Spurs", "Tottenham": "Spurs",
    "West Ham United": "West Ham", "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Wolves": "Wolves",
    "Sunderland AFC": "Sunderland", "Sunderland": "Sunderland",
    "Burnley FC": "Burnley", "Burnley": "Burnley",
    "Brentford FC": "Brentford", "Brentford": "Brentford",
}

def build_live_fixtures_df(bootstrap, raw_fixtures):
    teams = pd.DataFrame(bootstrap["teams"])
    id_to_name  = dict(zip(teams["id"], teams["name"]))
    id_to_short = dict(zip(teams["id"], teams["short_name"]))
    rows = []
    for f in raw_fixtures:
        if f["event"] is None:
            continue
        h_raw  = id_to_name.get(f["team_h"], "Unknown")
        a_raw  = id_to_name.get(f["team_a"], "Unknown")
        h_name = TEAM_NAME_MAP.get(h_raw, h_raw)
        a_name = TEAM_NAME_MAP.get(a_raw, a_raw)
        rows.append({
            "GW": f["event"], "Home Team": h_name, "Away Team": a_name,
            "HomeTeam_std": h_name, "AwayTeam_std": a_name,
            "HomeTeamShort": id_to_short.get(f["team_h"], "???"),
            "AwayTeamShort": id_to_short.get(f["team_a"], "???"),
            "HomeTeamID": f["team_h"], "AwayTeamID": f["team_a"],
            "finished": f["finished"],
        })
    return pd.DataFrame(rows).sort_values(["GW", "Home Team"]).reset_index(drop=True)

def build_dgw_bgw(raw_fixtures, bootstrap, start_gw, end_gw):
    teams_df   = pd.DataFrame(bootstrap["teams"])
    id_to_name = dict(zip(teams_df["id"], teams_df["name"]))
    gw_range   = range(start_gw, end_gw + 1)
    counts = {}
    for f in raw_fixtures:
        if f["event"] is None or f["event"] not in gw_range:
            continue
        for tid in [f["team_h"], f["team_a"]]:
            counts[(tid, f["event"])] = counts.get((tid, f["event"]), 0) + 1
    dgw, bgw = {}, {}
    for tid in teams_df["id"]:
        raw_name = id_to_name.get(tid, "")
        name = TEAM_NAME_MAP.get(raw_name, raw_name)
        dgw_gws = [gw for gw in gw_range if counts.get((tid, gw), 0) >= 2]
        bgw_gws = [gw for gw in gw_range if counts.get((tid, gw), 0) == 0]
        if dgw_gws: dgw[name] = dgw_gws
        if bgw_gws: bgw[name] = bgw_gws
    return dgw, bgw

# ── Review CSV helpers ─────────────────────────────────────────────────────────

@st.cache_data
def load_review_csv(path="review.csv"):
    try:
        df = pd.read_csv(path)
        df.columns = [c.lstrip('\ufeff') for c in df.columns]
        return df
    except FileNotFoundError:
        return None

def _team_id_lookup(bootstrap):
    """Build a comprehensive team name → API id mapping."""
    teams_df = pd.DataFrame(bootstrap["teams"])
    lookup = {}
    review_overrides = {
        "Nott'm Forest": "Nottingham Forest",
        "Man Utd": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
    }
    for _, row in teams_df.iterrows():
        lookup[row["name"]] = row["id"]
        lookup[row["short_name"]] = row["id"]
        mapped = TEAM_NAME_MAP.get(row["name"], row["name"])
        lookup[mapped] = row["id"]
    for alias, full in review_overrides.items():
        for _, row in teams_df.iterrows():
            if row["name"] == full:
                lookup[alias] = row["id"]
    return lookup

def _get_fixture_for_team(team_name, gw_fixtures, bootstrap):
    """Return (fixture_str, fdr_int) for a team in a given GW's fixtures."""
    teams_df    = pd.DataFrame(bootstrap["teams"])
    id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))
    id_lookup   = _team_id_lookup(bootstrap)

    tid = id_lookup.get(team_name)
    if tid is None:
        return "?", 3

    for f in gw_fixtures:
        if f["team_h"] == tid:
            opp = id_to_short.get(f["team_a"], "?")
            return f"{opp} (H)", f["team_h_difficulty"]
        if f["team_a"] == tid:
            opp = id_to_short.get(f["team_h"], "?")
            return f"{opp} (A)", f["team_a_difficulty"]
    return "BGW", 6

def get_captain_picks_from_review(review_df, gw, raw_fixtures, bootstrap):
    pts_col  = f"{gw}_Pts"
    mins_col = f"{gw}_xMins"
    if pts_col not in review_df.columns:
        return pd.DataFrame()

    df = review_df.copy()
    df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
    df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")
    df["Elite%"] = pd.to_numeric(df["Elite%"],  errors="coerce")

    active = df[df[mins_col] > 45].copy()
    top2   = active.nlargest(2, pts_col).copy()
    top2["Type"] = ["🏆 Top EV", "🥈 2nd EV"]

    diff_pool = active[(active["Elite%"] < 0.10) & (~active["Name"].isin(top2["Name"]))]
    diff = diff_pool.nlargest(1, pts_col).copy()
    diff["Type"] = ["🎯 Differential"]

    picks = pd.concat([top2, diff], ignore_index=True)

    gw_fixtures = [f for f in raw_fixtures if f["event"] == gw]
    fix_info    = picks["Team"].apply(lambda t: _get_fixture_for_team(t, gw_fixtures, bootstrap))
    picks["Fixture"] = [f[0] for f in fix_info]
    picks["FDR"]     = [f[1] for f in fix_info]
    picks["EV"]      = picks[pts_col].round(2)
    picks["Elite%"]  = picks["Elite%"].apply(lambda x: f"{x*100:.1f}%")

    return picks[["Type", "Name", "Team", "Pos", "EV", "Fixture", "FDR", "Elite%"]]

def get_captain_matrix(review_df, gws, raw_fixtures, bootstrap):
    matrix = {}
    for gw in gws:
        pts_col  = f"{gw}_Pts"
        mins_col = f"{gw}_xMins"
        if pts_col not in review_df.columns:
            continue
        df = review_df.copy()
        df[pts_col]  = pd.to_numeric(df[pts_col],  errors="coerce")
        df[mins_col] = pd.to_numeric(df[mins_col], errors="coerce")
        df["Elite%"] = pd.to_numeric(df["Elite%"],  errors="coerce")

        active = df[df[mins_col] > 45]
        top_ev = active[pts_col].max()
        if pd.isna(top_ev):
            continue

        within      = active[active[pts_col] >= top_ev - 0.5].nlargest(10, pts_col)
        gw_fixtures = [f for f in raw_fixtures if f["event"] == gw]

        rows = []
        for _, r in within.iterrows():
            fix, fdr = _get_fixture_for_team(r["Team"], gw_fixtures, bootstrap)
            rows.append({
                "Name": r["Name"], "EV": round(r[pts_col], 2),
                "Elite%": round(r["Elite%"] * 100, 1),
                "Fixture": fix, "FDR": fdr,
            })
        matrix[gw] = rows
    return matrix

# ── Price changes & injury ────────────────────────────────────────────────────

def get_price_changes(bootstrap):
    players  = pd.DataFrame(bootstrap["elements"])
    cols = [c for c in ["web_name", "team", "element_type", "now_cost",
                         "cost_change_event", "cost_change_start"] if c in players.columns]
    df = players[cols].copy()
    df = df[df["cost_change_event"] != 0].copy()
    df["Price (£m)"] = df["now_cost"] / 10.0
    df["Change"]     = df["cost_change_event"] / 10.0

    teams_df    = pd.DataFrame(bootstrap["teams"])
    id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))
    POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["Team"]     = df["team"].map(id_to_short)
    df["Position"] = df["element_type"].map(POSITION_MAP)
    df.rename(columns={"web_name": "Player"}, inplace=True)
    return df[["Player", "Team", "Position", "Price (£m)", "Change"]].sort_values("Change", ascending=False)

def get_injury_status(bootstrap):
    players = pd.DataFrame(bootstrap["elements"])
    cols = [c for c in ["web_name", "team", "element_type", "status", "news",
                         "chance_of_playing_next_round"] if c in players.columns]
    df = players[cols].copy()
    df = df[df["status"].isin(["d", "i", "s", "u"])].copy()

    teams_df    = pd.DataFrame(bootstrap["teams"])
    id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))
    POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    STATUS_MAP   = {"d": "⚠️ Doubt", "i": "🔴 Injured", "s": "🟡 Suspended", "u": "❌ Unavailable"}

    df["Team"]     = df["team"].map(id_to_short)
    df["Position"] = df["element_type"].map(POSITION_MAP)
    df["Status"]   = df["status"].map(STATUS_MAP)
    df["Play%"]    = df["chance_of_playing_next_round"].apply(
        lambda x: f"{int(x)}%" if pd.notna(x) else "?"
    )
    df.rename(columns={"web_name": "Player", "news": "News"}, inplace=True)
    return df[["Player", "Team", "Position", "Status", "Play%", "News"]].sort_values("Status").reset_index(drop=True)

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"
REVIEW_CSV_FILE   = "review.csv"

AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25
BGW_PENALTY_FDR = 6.0
FDR_THRESHOLDS  = {5: 120.0, 4: 108.0, 3: 99.0, 2: 90.0, 1: 0}

PREMIER_LEAGUE_TEAMS = sorted([
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
    'Man City', 'Man Utd', 'Newcastle', 'Nottm Forest', 'Sunderland',
    'Spurs', 'West Ham', 'Wolves'
])

TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Ipswich': 'IPS', 'Leeds': 'LEE',
    'Leicester': 'LEI', 'Liverpool': 'LIV', 'Man City': 'MCI', 'Man Utd': 'MUN',
    'Newcastle': 'NEW', 'Nottm Forest': 'NFO', 'Southampton': 'SOU',
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL',
    'Tottenham Hotspur': 'TOT', 'Manchester City': 'MCI', 'Manchester United': 'MUN'
}

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
        st.error("CSV files not found.")
        return None, None
    ratings_df['Team'] = ratings_df['Team'].map(TEAM_NAME_MAP).fillna(ratings_df['Team'])
    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    return ratings_df, fixtures_df

@st.cache_data
def create_all_data(fixtures_df_dict, start_gw, end_gw, ratings_df_dict, free_hit_gw=None):
    fixtures_df = pd.DataFrame(fixtures_df_dict)
    ratings_df  = pd.DataFrame(ratings_df_dict)
    ratings_dict  = ratings_df.set_index('Team').to_dict('index')
    pl_ratings    = ratings_df[ratings_df['Team'].isin(PREMIER_LEAGUE_TEAMS)]
    avg_off_score = pl_ratings['Off Score'].mean()
    avg_def_score = pl_ratings['Def Score'].mean()

    gw_range        = range(start_gw, end_gw + 1)
    projection_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team = row.get('HomeTeam_std') or row.get('Home Team', '')
        away_team = row.get('AwayTeam_std') or row.get('Away Team', '')
        gw_key    = f"GW{row['GW']}"

        home_stats = ratings_dict.get(home_team)
        away_stats = ratings_dict.get(away_team)

        if home_stats and away_stats and 'Off Score' in home_stats and 'Def Score' in away_stats:
            home_xg  = (home_stats['Off Score'] / avg_off_score) * (avg_def_score / away_stats['Def Score']) * AVG_LEAGUE_HOME_GOALS
            away_xg  = (away_stats['Off Score'] / avg_off_score) * (avg_def_score / home_stats['Def Score']) * AVG_LEAGUE_AWAY_GOALS
            home_fdr = get_fdr_score_from_rating(away_stats.get('Final Rating'))
            away_fdr = get_fdr_score_from_rating(home_stats.get('Final Rating'))

            if home_team in PREMIER_LEAGUE_TEAMS:
                if gw_key in projection_data[home_team]:
                    ex = projection_data[home_team][gw_key]
                    projection_data[home_team][gw_key] = {
                        "display": ex['display'] + " + " + f"{TEAM_ABBREVIATIONS.get(away_team,'???')} (H)",
                        "fdr": round((ex['fdr'] + home_fdr) / 2),
                        "xG": ex['xG'] + home_xg, "CS": ex['CS'] + math.exp(-away_xg)
                    }
                else:
                    projection_data[home_team][gw_key] = {
                        "display": f"{TEAM_ABBREVIATIONS.get(away_team,'???')} (H)",
                        "fdr": home_fdr, "xG": home_xg, "CS": math.exp(-away_xg)
                    }
            if away_team in PREMIER_LEAGUE_TEAMS:
                if gw_key in projection_data[away_team]:
                    ex = projection_data[away_team][gw_key]
                    projection_data[away_team][gw_key] = {
                        "display": ex['display'] + " + " + f"{TEAM_ABBREVIATIONS.get(home_team,'???')} (A)",
                        "fdr": round((ex['fdr'] + away_fdr) / 2),
                        "xG": ex['xG'] + away_xg, "CS": ex['CS'] + math.exp(-home_xg)
                    }
                else:
                    projection_data[away_team][gw_key] = {
                        "display": f"{TEAM_ABBREVIATIONS.get(home_team,'???')} (A)",
                        "fdr": away_fdr, "xG": away_xg, "CS": math.exp(-home_xg)
                    }

    df = pd.DataFrame.from_dict(projection_data, orient='index').reindex(
        columns=[f'GW{i}' for i in gw_range])
    free_hit_col = f'GW{free_hit_gw}' if free_hit_gw else None

    total_difficulty, total_xg, total_cs = [], [], []
    for _, row in df.iterrows():
        fdr_sum = xg_sum = cs_sum = 0
        for gw_col, cell_data in row.items():
            if gw_col != free_hit_col:
                if isinstance(cell_data, dict):
                    fdr_sum += cell_data.get('fdr', 0)
                    xg_sum  += cell_data.get('xG', 0)
                    cs_sum  += cell_data.get('CS', 0)
                else:
                    fdr_sum += BGW_PENALTY_FDR
        total_difficulty.append(fdr_sum)
        total_xg.append(xg_sum)
        total_cs.append(cs_sum)

    df['Total Difficulty'] = total_difficulty
    df['Total xG']         = total_xg
    df['xCS']              = total_cs
    return df

@st.cache_data
def find_fixture_runs(fixtures_df_dict, rating_dict, start_gw):
    fixtures_df  = pd.DataFrame(fixtures_df_dict)
    all_fixtures = {team: [] for team in PREMIER_LEAGUE_TEAMS}
    for gw in range(1, 39):
        for _, row in fixtures_df[fixtures_df['GW'] == gw].iterrows():
            home_team = row.get('HomeTeam_std') or row.get('Home Team', '')
            away_team = row.get('AwayTeam_std') or row.get('Away Team', '')
            if home_team in PREMIER_LEAGUE_TEAMS:
                rating = rating_dict.get(away_team, {}).get('Final Rating')
                all_fixtures[home_team].append({"gw": gw, "opp": away_team, "loc": "H", "fdr": get_fdr_score_from_rating(rating)})
            if away_team in PREMIER_LEAGUE_TEAMS:
                rating = rating_dict.get(home_team, {}).get('Final Rating')
                all_fixtures[away_team].append({"gw": gw, "opp": home_team, "loc": "A", "fdr": get_fdr_score_from_rating(rating)})

    good_runs = {}
    for team, fixtures in all_fixtures.items():
        current_run = []
        for fixture in sorted(fixtures, key=lambda x: x['gw']):
            if fixture['gw'] < start_gw: continue
            if fixture['fdr'] is not None and fixture['fdr'] <= 3:
                current_run.append(fixture)
            else:
                if len(current_run) >= 3:
                    good_runs.setdefault(team, []).append(current_run)
                current_run = []
        if len(current_run) >= 3:
            good_runs.setdefault(team, []).append(current_run)
    return good_runs

# =============================================================================
# MAIN APP
# =============================================================================

st.set_page_config(layout="wide")
st.title("🏆 CoachFPL Command Center")

with st.sidebar:
    st.header("⚙️ Settings")
    use_live = st.toggle("📡 Live FPL Data", value=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.success("✅ Cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Rerun", use_container_width=True):
            st.rerun()

bootstrap    = None
raw_fixtures = None
live_ok      = False
current_gw   = 29

if use_live:
    try:
        bootstrap    = _fetch_bootstrap()
        raw_fixtures = _fetch_live_fixtures()
        live_ok      = True
        current_gw   = get_current_gw(bootstrap)
        st.sidebar.success(f"✅ Live data | GW{current_gw}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Live failed ({e}). Using CSV.")

ratings_df, fixtures_df = load_csv_data()
review_df = load_review_csv(REVIEW_CSV_FILE)

if live_ok and bootstrap and raw_fixtures:
    fixtures_df = build_live_fixtures_df(bootstrap, raw_fixtures)

with st.expander("Glossary & How It Works"):
    st.markdown(f"""
    - **FDR:** Fixture Difficulty Rating (1-5). Lower is better.
    - **xG:** Projected Goals. Higher = better for attackers.
    - **xCS:** Expected Clean Sheets. Higher = better for defenders.
    - **EV:** Expected Value (projected points from FPL Review CSV).
    - **Elite%:** % of elite FPL managers owning the player.
    - **Differential:** Good EV, low Elite% (under 10%).
    - **Captain Matrix:** All captains within 0.5 EV of the top pick per GW.
    - **Live Data:** {"🟢 Active" if live_ok else "🔴 Off — CSV mode"}
    """)

if ratings_df is not None and fixtures_df is not None:
    st.sidebar.header("Controls")
    col_start, col_end = st.sidebar.columns(2)
    default_gw = current_gw if live_ok else 29
    with col_start:
        start_gw = st.number_input("Start GW:", min_value=1, max_value=38, value=default_gw)
    with col_end:
        end_gw = st.number_input("End GW:", min_value=1, max_value=38, value=min(default_gw + 4, 38))

    selected_teams = st.sidebar.multiselect("Select teams:", PREMIER_LEAGUE_TEAMS, default=PREMIER_LEAGUE_TEAMS)
    fh_options  = [None] + list(range(start_gw, end_gw + 1))
    free_hit_gw = st.sidebar.selectbox("Free Hit GW:", options=fh_options,
                                        format_func=lambda x: "None" if x is None else f"GW{x}")

    master_df = create_all_data(
        fixtures_df.to_dict('records'), start_gw, end_gw,
        ratings_df.to_dict('records'), free_hit_gw
    )
    if selected_teams:
        master_df = master_df.loc[[t for t in master_df.index if t in selected_teams]]

    gw_columns = [f'GW{i}' for i in range(start_gw, end_gw + 1)]
    if free_hit_gw and f'GW{free_hit_gw}' in gw_columns:
        gw_columns.remove(f'GW{free_hit_gw}')

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Fixture Difficulty (FDR)", "⚽ Projected Goals (xG)",
        "🧤 Clean Sheets (xCS)", "🎯 Captain Picks",
        "🏅 Captain Matrix", "📡 Live Radar",
    ])

    # ── Tab 1: FDR ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Fixture Difficulty Rating (Lower = easier)")
        df_display = master_df.sort_values('Total Difficulty').reset_index().rename(columns={'index': 'Team'})
        df_display = df_display[['Team', 'Total Difficulty'] + gw_columns]

        df_for_grid = df_display[['Team', 'Total Difficulty']].copy()
        for col in gw_columns:
            df_for_grid[col] = df_display[col]
            df_for_grid[f'{col}_display'] = df_display[col].apply(
                lambda x: x['display'] if isinstance(x, dict) else 'BGW')
            df_for_grid[f'{col}_fdr'] = df_display[col].apply(
                lambda x: x['fdr'] if isinstance(x, dict) else BGW_PENALTY_FDR)

        gb = GridOptionsBuilder.from_dataframe(df_for_grid)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150, sortable=True)
        gb.configure_column("Total Difficulty", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)
        for col in gw_columns:
            gb.configure_column(f'{col}_display', hide=True)
            gb.configure_column(f'{col}_fdr', hide=True)
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
               theme='streamlit-dark', height=(len(df_for_grid)+1)*35,
               fit_columns_on_grid_load=True, key=f'fdr_{start_gw}_{end_gw}')

    # ── Tab 2: xG ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Projected Goals (Higher = better for attackers)")
        df_display = master_df.sort_values('Total xG', ascending=False).reset_index().rename(columns={'index': 'Team'})
        df_display = df_display[['Team', 'Total xG'] + gw_columns]
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150)
        gb.configure_column("Total xG", valueFormatter="data['Total xG'].toFixed(2)", flex=1.5, type=["numericColumn"], minWidth=140)
        jscode = JsCode("""function(p){var c=p.data[p.colDef.field];if(c&&typeof c==='object'&&c.xG!==undefined){var x=c.xG;var bg=x>=2.0?'#63be7b':x>=1.5?'#95d2a6':x>=1.0?'#bfe4cb':'#D3D3D3';return{'backgroundColor':bg,'color':'#31333F','fontWeight':'bold'};}return{'textAlign':'center','backgroundColor':'#1E1E1E','color':'#FF4B4B','fontWeight':'bold','border':'1px solid #FF4B4B'};}""")
        for col in gw_columns:
            gb.configure_column(col, headerName=col,
                valueGetter=f"(data['{col}']&&typeof data['{col}']==='object')?data['{col}'].xG.toFixed(2):'BGW'",
                cellStyle=jscode, flex=1, minWidth=90)
        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
               theme='streamlit-dark', height=(len(df_display)+1)*35,
               fit_columns_on_grid_load=True, key=f'xg_{start_gw}_{end_gw}')

    # ── Tab 3: xCS ────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Expected Clean Sheets (Higher = better for defenders)")
        df_display = master_df.sort_values('xCS', ascending=False).reset_index().rename(columns={'index': 'Team'})
        df_display = df_display[['Team', 'xCS'] + gw_columns]
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150)
        gb.configure_column("xCS", pinned='left', valueFormatter="data['xCS'].toFixed(2)", flex=1.5, type=["numericColumn"], minWidth=140)
        jscode = JsCode("""function(p){var c=p.data[p.colDef.field];if(c&&typeof c==='object'&&c.CS!==undefined){var s=c.CS;var bg=s>=0.5?'#00ff85':s>=0.35?'#50c369':s>=0.2?'#D3D3D3':s>=0.1?'#9d66a0':'#6f2a74';var fg=(s>=0.2&&s<0.35)?'#31333F':'#FFFFFF';return{'backgroundColor':bg,'color':fg,'fontWeight':'bold'};}return{'textAlign':'center','backgroundColor':'#1E1E1E','color':'#FF4B4B','fontWeight':'bold','border':'1px solid #FF4B4B'};}""")
        for col in gw_columns:
            gb.configure_column(col, headerName=col,
                valueGetter=f"(data['{col}']&&typeof data['{col}']==='object')?(data['{col}'].CS*100).toFixed(0)+'%':'BGW'",
                cellStyle=jscode, flex=1, minWidth=90)
        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
               theme='streamlit-dark', height=(len(df_display)+1)*35, key=f'cs_{start_gw}_{end_gw}')

    # ── Tab 4: Captain Picks ──────────────────────────────────────────────────
    with tab4:
        st.subheader("🎯 Captain Picks — from FPL Review EV")
        if review_df is None:
            st.error("❌ review.csv not found. Place it in the same folder as app.py.")
        elif not live_ok:
            st.warning("⚠️ Enable Live FPL Data for fixture info.")
        else:
            review_gws  = sorted([int(c.split("_")[0]) for c in review_df.columns
                                   if c.endswith("_Pts") and c.split("_")[0].isdigit()])
            future_gws  = [g for g in review_gws if g >= current_gw]
            if not future_gws:
                st.warning("No upcoming GWs in review.csv.")
            else:
                gw_sel = st.selectbox("Select GW:", future_gws, key="cap_picks_gw")
                picks  = get_captain_picks_from_review(review_df, gw_sel, raw_fixtures, bootstrap)

                if picks.empty:
                    st.warning(f"No data for GW{gw_sel}.")
                else:
                    st.markdown(f"**GW{gw_sel} Captain Recommendations**")
                    st.caption("Top 2 EV + 1 Differential (Elite% < 10%). EV = projected points from FPL Review.")

                    FDR_BG = {1:'#00ff85',2:'#50c369',3:'#D3D3D3',4:'#9d66a0',5:'#6f2a74',6:'#1E1E1E'}
                    FDR_FG = {1:'#31333F',2:'#31333F',3:'#31333F',4:'white',5:'white',6:'#FF4B4B'}

                    def colour_row(row):
                        type_styles = {
                            "🏆 Top EV":       "background-color:#1a472a;color:#00ff85;font-weight:bold",
                            "🥈 2nd EV":       "background-color:#1a3a47;color:#50c369;font-weight:bold",
                            "🎯 Differential": "background-color:#3d2b1f;color:#f4a261;font-weight:bold",
                        }
                        base = [""] * len(row)
                        idx  = list(row.index)
                        base[idx.index("Type")] = type_styles.get(row["Type"], "")
                        fdr_val = int(row.get("FDR", 3))
                        base[idx.index("FDR")] = (f"background-color:{FDR_BG.get(fdr_val,'#444')};"
                                                   f"color:{FDR_FG.get(fdr_val,'white')};font-weight:bold")
                        return base

                    st.dataframe(picks.style.apply(colour_row, axis=1),
                                 use_container_width=True, hide_index=True)

                    st.markdown("---")
                    chart_data = picks.set_index("Name")[["EV"]]
                    st.bar_chart(chart_data)

    # ── Tab 5: Captain Matrix ─────────────────────────────────────────────────
    with tab5:
        st.subheader("🏅 Captain Matrix — Within 0.5 EV of Top Pick")
        if review_df is None:
            st.error("❌ review.csv not found.")
        elif not live_ok:
            st.warning("⚠️ Enable Live FPL Data.")
        else:
            review_gws = sorted([int(c.split("_")[0]) for c in review_df.columns
                                  if c.endswith("_Pts") and c.split("_")[0].isdigit()])
            future_gws = [g for g in review_gws if g >= current_gw]
            matrix_gws = st.multiselect("Show GWs:", future_gws,
                                         default=future_gws[:6] if len(future_gws) >= 6 else future_gws,
                                         key="matrix_gws")
            if matrix_gws:
                matrix = get_captain_matrix(review_df, matrix_gws, raw_fixtures, bootstrap)

                FDR_BG = {1:'#00ff85',2:'#50c369',3:'#D3D3D3',4:'#9d66a0',5:'#6f2a74',6:'#1E1E1E'}
                FDR_FG = {1:'#31333F',2:'#31333F',3:'#31333F',4:'white',5:'white',6:'#FF4B4B'}

                html = """<style>
                .cm{border-collapse:collapse;width:100%;font-family:sans-serif;font-size:13px}
                .cm th{background:#2a2a2a;color:#bbb;padding:8px 10px;border:1px solid #444;text-align:center;min-width:80px}
                .cm td{padding:6px 8px;border:1px solid #333;vertical-align:top}
                .cm tr:first-child td{border-top:2px solid #555}
                .fdr-badge{border-radius:3px;padding:2px 5px;font-weight:bold;font-size:11px;display:inline-block}
                .ev-num{font-weight:bold;color:#00ff85;margin-left:4px}
                .elite-pct{color:#888;font-size:11px}
                .top-name{font-weight:bold;color:white}
                .top-row td{border-top:2px solid #00ff8544}
                </style><table class="cm"><thead><tr>"""

                for gw in matrix_gws:
                    html += f"<th>GW{gw}</th>"
                html += "</tr></thead><tbody>"

                max_rows = max((len(matrix.get(gw, [])) for gw in matrix_gws), default=0)

                for ri in range(max_rows):
                    row_class = " class='top-row'" if ri == 0 else ""
                    html += f"<tr{row_class}>"
                    for gw in matrix_gws:
                        rows = matrix.get(gw, [])
                        if ri < len(rows):
                            r   = rows[ri]
                            bg  = FDR_BG.get(r["FDR"], '#444')
                            fg  = FDR_FG.get(r["FDR"], 'white')
                            top = " class='top-name'" if ri == 0 else ""
                            html += f"""<td>
                                <span{top}>{r['Name']}</span>
                                <span class='ev-num'>{r['EV']}</span><br>
                                <span class='fdr-badge' style='background:{bg};color:{fg}'>{r['Fixture']}</span>
                                <span class='elite-pct'> {r['Elite%']}%</span>
                            </td>"""
                        else:
                            html += "<td></td>"
                    html += "</tr>"

                html += "</tbody></table>"
                st.markdown(html, unsafe_allow_html=True)
                st.caption("Top row = highest EV pick. All picks within 0.5 EV. Colour = live FDR. % = Elite manager ownership.")

    # ── Tab 6: Live Radar ─────────────────────────────────────────────────────
    with tab6:
        st.subheader("📡 Live FPL Radar")
        if not live_ok:
            st.warning("⚠️ Enable Live FPL Data.")
        else:
            col_a, col_b = st.columns(2)

            with col_a:
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
                st.markdown("### 💰 Price Changes (This GW)")
                try:
                    price_df = get_price_changes(bootstrap)
                    if price_df.empty:
                        st.info("No price changes this GW.")
                    else:
                        rises = price_df[price_df["Change"] > 0]
                        falls = price_df[price_df["Change"] < 0]
                        if not rises.empty:
                            st.markdown("**🔼 Price Rises**")
                            st.dataframe(rises.reset_index(drop=True), use_container_width=True, hide_index=True)
                        if not falls.empty:
                            st.markdown("**🔽 Price Falls**")
                            st.dataframe(falls.reset_index(drop=True), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Price data unavailable: {e}")

            with col_b:
                st.markdown("### 🚑 Injuries & Availability")
                try:
                    injury_df = get_injury_status(bootstrap)
                    if injury_df.empty:
                        st.success("No injuries or doubts reported.")
                    else:
                        for status_label in ["🔴 Injured", "⚠️ Doubt", "🟡 Suspended", "❌ Unavailable"]:
                            sub = injury_df[injury_df["Status"] == status_label]
                            if not sub.empty:
                                st.markdown(f"**{status_label}**")
                                st.dataframe(sub[["Player","Team","Position","Play%","News"]].reset_index(drop=True),
                                             use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Injury data unavailable: {e}")

                st.markdown("---")
                st.markdown("### 📈 Ownership Movers")
                players_live = pd.DataFrame(bootstrap["elements"])
                movers = players_live[["web_name","selected_by_percent",
                                        "transfers_in_event","transfers_out_event"]].copy()
                for c in movers.columns[1:]:
                    movers[c] = pd.to_numeric(movers[c], errors="coerce")

                st.markdown("**🔼 Most IN**")
                top_in = movers.nlargest(7,"transfers_in_event")[["web_name","selected_by_percent","transfers_in_event"]]
                top_in.columns = ["Player","Sel%","Transfers In"]
                st.dataframe(top_in, hide_index=True, use_container_width=True)

                st.markdown("**🔽 Most OUT**")
                top_out = movers.nlargest(7,"transfers_out_event")[["web_name","selected_by_percent","transfers_out_event"]]
                top_out.columns = ["Player","Sel%","Transfers Out"]
                st.dataframe(top_out, hide_index=True, use_container_width=True)

    # ── Easy Run Finder ───────────────────────────────────────────────────────
    st.markdown("---")
    st.sidebar.header("Easy Run Finder")
    st.sidebar.info("Find 3+ easy/neutral fixtures (FDR 1-3).")
    teams_to_check = st.sidebar.multiselect("Select teams:", PREMIER_LEAGUE_TEAMS, default=[], key="run_teams")

    st.header("✅ Easy Fixture Runs")
    if teams_to_check:
        rating_dict   = ratings_df.set_index('Team').to_dict('index')
        all_runs      = find_fixture_runs(fixtures_df.to_dict('records'), rating_dict, start_gw)
        results_found = False
        for team in teams_to_check:
            team_runs = all_runs.get(team)
            if team_runs:
                results_found = True
                with st.expander(f"**{team}** ({len(team_runs)} run(s))"):
                    for i, run in enumerate(team_runs):
                        s, e = run[0]['gw'], run[-1]['gw']
                        st.markdown(f"**Run {i+1}: GW{s} – GW{e}**")
                        for fix in run:
                            opp = TEAM_ABBREVIATIONS.get(fix['opp'], '???')
                            st.markdown(f"- **GW{fix['gw']}:** {opp} ({fix['loc']}) — FDR: {fix['fdr']}")
        if not results_found:
            st.warning(f"No easy runs found from GW{start_gw}.")
    else:
        st.info("Select teams from the sidebar to check easy runs.")

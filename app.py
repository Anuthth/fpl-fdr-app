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

def build_live_fixtures_df(bootstrap, raw_fixtures):
    """Returns fixtures_df in the same shape as Fixtures202526.csv."""
    teams = pd.DataFrame(bootstrap["teams"])
    id_to_name  = dict(zip(teams["id"], teams["name"]))
    id_to_short = dict(zip(teams["id"], teams["short_name"]))

    rows = []
    for f in raw_fixtures:
        if f["event"] is None:
            continue
        h_name = TEAM_NAME_MAP.get(id_to_name.get(f["team_h"], ""), id_to_name.get(f["team_h"], "Unknown"))
        a_name = TEAM_NAME_MAP.get(id_to_name.get(f["team_a"], ""), id_to_name.get(f["team_a"], "Unknown"))
        rows.append({
            "GW":           f["event"],
            "Home Team":    h_name,
            "Away Team":    a_name,
            "HomeTeam_std": h_name,
            "AwayTeam_std": a_name,
            "HomeTeamShort": id_to_short.get(f["team_h"], "???"),
            "AwayTeamShort": id_to_short.get(f["team_a"], "???"),
            "HomeTeamID":   f["team_h"],
            "AwayTeamID":   f["team_a"],
            "finished":     f["finished"],
        })
    return pd.DataFrame(rows).sort_values(["GW", "Home Team"]).reset_index(drop=True)

def build_captain_df(bootstrap, raw_fixtures, gw, ratings_dict, top_n=10):
    """Captain candidates ranked by Form × Fixture Ease."""
    POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players = pd.DataFrame(bootstrap["elements"])

    keep = ["id", "web_name", "team", "element_type", "now_cost",
            "total_points", "form", "selected_by_percent",
            "minutes", "goals_scored", "assists"]
    keep = [c for c in keep if c in players.columns]
    players = players[keep].copy()
    players.rename(columns={"team": "team_id", "element_type": "position_id",
                             "now_cost": "price_raw"}, inplace=True)
    players["position"] = players["position_id"].map(POSITION_MAP)
    players["price"]    = players["price_raw"] / 10.0
    players["form"]     = pd.to_numeric(players["form"], errors="coerce")
    players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors="coerce")

    # Build matchups for the GW
    gw_fx = [f for f in raw_fixtures if f["event"] == gw]
    matchups = {}
    for f in gw_fx:
        matchups[f["team_h"]] = (f["team_a_difficulty"], True,  f["team_a"])
        matchups[f["team_a"]] = (f["team_h_difficulty"], False, f["team_h"])

    teams_df = pd.DataFrame(bootstrap["teams"])
    id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))

    df = players[players["position_id"].isin([3, 4])].copy()
    df = df[df["form"] >= 3.0]

    df["difficulty"] = df["team_id"].map(lambda t: matchups.get(t, (3, False, 0))[0])
    df["is_home"]    = df["team_id"].map(lambda t: matchups.get(t, (3, False, 0))[1])
    df["opp_id"]     = df["team_id"].map(lambda t: matchups.get(t, (3, False, 0))[2])
    df["opp_short"]  = df["opp_id"].map(lambda t: id_to_short.get(t, "?"))
    df["home_away"]  = df["is_home"].map({True: "(H)", False: "(A)"})
    df["fixture"]    = df["opp_short"] + " " + df["home_away"]
    df["ease"]       = 6 - df["difficulty"]
    df["score"]      = df["form"] * df["ease"]

    result = df.nlargest(top_n, "score")[[
        "web_name", "position", "price", "form",
        "total_points", "selected_by_percent", "fixture", "difficulty", "score"
    ]].copy()

    result.rename(columns={
        "web_name": "Player", "position": "Pos", "price": "£",
        "form": "Form", "total_points": "Pts",
        "selected_by_percent": "Sel%", "fixture": "Fixture",
        "difficulty": "FDR", "score": "Score"
    }, inplace=True)

    return result.reset_index(drop=True)

def build_dgw_bgw(raw_fixtures, bootstrap, start_gw, end_gw):
    """Returns DGW/BGW info for a GW range."""
    teams_df    = pd.DataFrame(bootstrap["teams"])
    id_to_name  = dict(zip(teams_df["id"], teams_df["name"]))
    all_team_ids = list(teams_df["id"])
    gw_range    = range(start_gw, end_gw + 1)

    counts = {}
    for f in raw_fixtures:
        if f["event"] is None or f["event"] not in gw_range:
            continue
        for tid in [f["team_h"], f["team_a"]]:
            key = (tid, f["event"])
            counts[key] = counts.get(key, 0) + 1

    dgw, bgw = {}, {}
    for tid in all_team_ids:
        name = id_to_name.get(tid, str(tid))
        dgw_gws = [gw for gw in gw_range if counts.get((tid, gw), 0) >= 2]
        bgw_gws = [gw for gw in gw_range if counts.get((tid, gw), 0) == 0]
        if dgw_gws: dgw[name] = dgw_gws
        if bgw_gws: bgw[name] = bgw_gws

    return dgw, bgw

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25
BGW_PENALTY_FDR = 6.0

FDR_THRESHOLDS = {5: 120.0, 4: 108.0, 3: 99.0, 2: 90.0, 1: 0}
FDR_COLORS = {1: '#00ff85', 2: '#50c369', 3: '#D3D3D3', 4: '#9d66a0', 5: '#6f2a74'}

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
}

# --- Helper Functions ---

def get_fdr_score_from_rating(team_rating):
    if pd.isna(team_rating): return 3
    if team_rating >= FDR_THRESHOLDS[5]: return 5
    if team_rating >= FDR_THRESHOLDS[4]: return 4
    if team_rating >= FDR_THRESHOLDS[3]: return 3
    if team_rating >= FDR_THRESHOLDS[2]: return 2
    return 1

@st.cache_data
def load_csv_data():
    """Fallback: load from local CSV files."""
    try:
        ratings_df  = pd.read_csv(RATINGS_CSV_FILE)
        fixtures_df = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error("CSV files not found. Please enable Live Data or ensure CSVs are present.")
        return None, None
    ratings_df['Team'] = ratings_df['Team'].map(TEAM_NAME_MAP).fillna(ratings_df['Team'])
    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    return ratings_df, fixtures_df

@st.cache_data
def create_all_data(fixtures_df_dict, start_gw, end_gw, ratings_df_dict, free_hit_gw=None):
    """Prepares master dataframe. Accepts dicts for cache compatibility."""
    fixtures_df = pd.DataFrame(fixtures_df_dict)
    ratings_df  = pd.DataFrame(ratings_df_dict)

    ratings_dict   = ratings_df.set_index('Team').to_dict('index')
    pl_ratings     = ratings_df[ratings_df['Team'].isin(PREMIER_LEAGUE_TEAMS)]
    avg_off_score  = pl_ratings['Off Score'].mean()
    avg_def_score  = pl_ratings['Def Score'].mean()

    gw_range = range(start_gw, end_gw + 1)
    projection_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team = row.get('HomeTeam_std') or row.get('Home Team', '')
        away_team = row.get('AwayTeam_std') or row.get('Away Team', '')
        gw = f"GW{row['GW']}"

        home_stats = ratings_dict.get(home_team)
        away_stats = ratings_dict.get(away_team)

        if home_stats and away_stats and 'Off Score' in home_stats and 'Def Score' in away_stats:
            home_xg = (home_stats['Off Score'] / avg_off_score) * (avg_def_score / away_stats['Def Score']) * AVG_LEAGUE_HOME_GOALS
            away_xg = (away_stats['Off Score'] / avg_off_score) * (avg_def_score / home_stats['Def Score']) * AVG_LEAGUE_AWAY_GOALS

            if home_team in PREMIER_LEAGUE_TEAMS:
                if gw in projection_data[home_team]:
                    # DGW: combine
                    existing = projection_data[home_team][gw]
                    combined_display = existing['display'] + " + " + f"{TEAM_ABBREVIATIONS.get(away_team, '???')} (H)"
                    avg_fdr = round((existing['fdr'] + get_fdr_score_from_rating(away_stats.get('Final Rating'))) / 2)
                    projection_data[home_team][gw] = {
                        "display": combined_display,
                        "fdr": avg_fdr,
                        "xG": existing['xG'] + home_xg,
                        "CS": existing['CS'] + math.exp(-away_xg)
                    }
                else:
                    projection_data[home_team][gw] = {
                        "display": f"{TEAM_ABBREVIATIONS.get(away_team, '???')} (H)",
                        "fdr": get_fdr_score_from_rating(away_stats.get('Final Rating')),
                        "xG": home_xg, "CS": math.exp(-away_xg)
                    }
            if away_team in PREMIER_LEAGUE_TEAMS:
                if gw in projection_data[away_team]:
                    existing = projection_data[away_team][gw]
                    combined_display = existing['display'] + " + " + f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)"
                    avg_fdr = round((existing['fdr'] + get_fdr_score_from_rating(home_stats.get('Final Rating'))) / 2)
                    projection_data[away_team][gw] = {
                        "display": combined_display,
                        "fdr": avg_fdr,
                        "xG": existing['xG'] + away_xg,
                        "CS": existing['CS'] + math.exp(-home_xg)
                    }
                else:
                    projection_data[away_team][gw] = {
                        "display": f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)",
                        "fdr": get_fdr_score_from_rating(home_stats.get('Final Rating')),
                        "xG": away_xg, "CS": math.exp(-home_xg)
                    }

    df = pd.DataFrame.from_dict(projection_data, orient='index').reindex(columns=[f'GW{i}' for i in gw_range])
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
    fixtures_df = pd.DataFrame(fixtures_df_dict)
    all_fixtures = {team: [] for team in PREMIER_LEAGUE_TEAMS}
    for gw in range(1, 39):
        gw_fixtures = fixtures_df[fixtures_df['GW'] == gw]
        for _, row in gw_fixtures.iterrows():
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Live data toggle
    use_live = st.toggle("📡 Live FPL Data", value=True,
                         help="Pulls fixtures & player data directly from the FPL API. Refreshes every 30 min.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Rerun", use_container_width=True):
            st.rerun()

# ── Load Data ─────────────────────────────────────────────────────────────────
bootstrap     = None
raw_fixtures  = None
live_ok       = False

if use_live:
    try:
        bootstrap    = _fetch_bootstrap()
        raw_fixtures = _fetch_live_fixtures()
        live_ok      = True
        current_gw   = get_current_gw(bootstrap)
        st.sidebar.success(f"✅ Live data | GW{current_gw}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Live data failed ({e}). Falling back to CSV.")
        use_live = False

ratings_df, fixtures_df = load_csv_data()

if live_ok and bootstrap and raw_fixtures:
    fixtures_df = build_live_fixtures_df(bootstrap, raw_fixtures)
    # ratings_df stays as CSV (your custom ratings model)

with st.expander("Glossary & How It Works"):
    st.markdown(f"""
    - **FDR:** Fixture Difficulty Rating (1-5). Lower is better.
    - **xG:** Projected Goals. Higher is better for attackers.
    - **xCS:** Expected Clean Sheets. Higher is better for defenders.
    - **BGW Penalty:** Teams without a fixture in a GW are penalised in Total Difficulty.
    - **Live Data:** {"🟢 Active — fixtures pulled from FPL API" if live_ok else "🔴 Off — using CSV files"}
    """)

if ratings_df is not None and fixtures_df is not None:

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    st.sidebar.header("Controls")
    col_start, col_end = st.sidebar.columns(2)

    default_gw = current_gw if live_ok else 28
    with col_start:
        start_gw = st.number_input("Start GW:", min_value=1, max_value=38, value=default_gw)
    with col_end:
        end_gw = st.number_input("End GW:", min_value=1, max_value=38, value=min(default_gw + 5, 38))

    selected_teams = st.sidebar.multiselect("Select teams to display:", PREMIER_LEAGUE_TEAMS, default=PREMIER_LEAGUE_TEAMS)

    fh_options = [None] + list(range(start_gw, end_gw + 1))
    free_hit_gw = st.sidebar.selectbox(
        "Free Hit GW (optional):",
        options=fh_options,
        format_func=lambda x: "None" if x is None else f"GW{x}"
    )

    # ── Generate Master Data ──────────────────────────────────────────────────
    master_df = create_all_data(
        fixtures_df.to_dict('records'),
        start_gw, end_gw,
        ratings_df.to_dict('records'),
        free_hit_gw
    )

    if selected_teams:
        teams_to_show = [t for t in master_df.index if t in selected_teams]
        master_df = master_df.loc[teams_to_show]

    gw_columns = [f'GW{i}' for i in range(start_gw, end_gw + 1)]
    if free_hit_gw and f'GW{free_hit_gw}' in gw_columns:
        gw_columns.remove(f'GW{free_hit_gw}')

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Fixture Difficulty (FDR)",
        "⚽ Projected Goals (xG)",
        "🧤 Expected Clean Sheets (xCS)",
        "🎯 Captain Picks",
        "📡 Live Radar",
    ])

    # ── Tab 1: FDR ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Fixture Difficulty Rating (Lower = easier)")
        df_display = master_df.sort_values(by='Total Difficulty', ascending=True).reset_index().rename(columns={'index': 'Team'})
        column_order = ['Team', 'Total Difficulty'] + gw_columns
        df_display = df_display[column_order]

        df_for_grid = df_display[['Team', 'Total Difficulty']].copy()
        for col in gw_columns:
            df_for_grid[col] = df_display[col]
            df_for_grid[f'{col}_display'] = df_display[col].apply(
                lambda x: x['display'] if isinstance(x, dict) and 'display' in x else 'BGW'
            )
            df_for_grid[f'{col}_fdr'] = df_display[col].apply(
                lambda x: x['fdr'] if isinstance(x, dict) and 'fdr' in x else BGW_PENALTY_FDR
            )

        gb = GridOptionsBuilder.from_dataframe(df_for_grid)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150, sortable=True)
        gb.configure_column("Total Difficulty", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)

        for col in gw_columns:
            gb.configure_column(f'{col}_display', hide=True)
            gb.configure_column(f'{col}_fdr', hide=True)

        for col in gw_columns:
            value_formatter = f"function(params) {{ return params.data['{col}_display'] || ''; }}"
            jscode_for_col  = f"""function(params) {{
                const display = params.data['{col}_display'];
                if (display === 'BGW') {{
                    return {{'backgroundColor': '#1E1E1E', 'color': '#FF4B4B', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #FF4B4B'}};
                }}
                const fdrValue = params.data['{col}_fdr'];
                if (fdrValue !== undefined && fdrValue !== null) {{
                    const colors = {{1: '#00ff85', 2: '#50c369', 3: '#D3D3D3', 4: '#9d66a0', 5: '#6f2a74'}};
                    const bgColor = colors[fdrValue] || '#444444';
                    const textColor = (fdrValue <= 3) ? '#31333F' : '#FFFFFF';
                    return {{'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold', 'textAlign': 'center'}};
                }}
                return {{'textAlign': 'center', 'backgroundColor': '#444444'}};
            }}"""
            value_getter = f"function(params) {{ return params.data['{col}_fdr']; }}"
            gb.configure_column(col, headerName=col,
                                 valueGetter=JsCode(value_getter),
                                 valueFormatter=JsCode(value_formatter),
                                 cellStyle=JsCode(jscode_for_col),
                                 flex=1, minWidth=90, sortable=True)

        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        AgGrid(df_for_grid, gridOptions=gb.build(), allow_unsafe_jscode=True,
               theme='streamlit-dark', height=(len(df_for_grid) + 1) * 35,
               fit_columns_on_grid_load=True, key=f'fdr_grid_{start_gw}_{end_gw}')

    # ── Tab 2: xG ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Projected Goals (Higher = better for attackers)")
        df_display = master_df.sort_values(by='Total xG', ascending=False).reset_index().rename(columns={'index': 'Team'})
        df_display = df_display[['Team', 'Total xG'] + gw_columns]

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        gb.configure_column("Team", pinned='left', cellStyle={'textAlign': 'left'}, flex=2, minWidth=150, sortable=True)
        gb.configure_column("Total xG", valueFormatter="data['Total xG'].toFixed(2)", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)

        jscode = JsCode("""function(params) {
            const cellData = params.data[params.colDef.field];
            if (cellData && typeof cellData === 'object' && cellData.xG !== undefined) {
                const xG = cellData.xG;
                let bgColor;
                if (xG >= 2.0)      { bgColor = '#63be7b'; }
                else if (xG >= 1.5) { bgColor = '#95d2a6'; }
                else if (xG >= 1.0) { bgColor = '#bfe4cb'; }
                else                { bgColor = '#D3D3D3'; }
                return {'backgroundColor': bgColor, 'color': '#31333F', 'fontWeight': 'bold'};
            }
            return {'textAlign': 'center', 'backgroundColor': '#1E1E1E', 'color': '#FF4B4B', 'fontWeight': 'bold', 'border': '1px solid #FF4B4B'};
        };""")

        comparator_template = """function(valueA, valueB, nodeA, nodeB) {{ const xgA = (nodeA.data['{gw_col}'] && typeof nodeA.data['{gw_col}'] === 'object') ? nodeA.data['{gw_col}'].xG : 0; const xgB = (nodeB.data['{gw_col}'] && typeof nodeB.data['{gw_col}'] === 'object') ? nodeB.data['{gw_col}'].xG : 0; return xgA - xgB; }}"""

        for col in gw_columns:
            gb.configure_column(col, headerName=col,
                                 valueGetter=f"(data['{col}'] && typeof data['{col}'] === 'object') ? data['{col}'].xG.toFixed(2) : 'BGW'",
                                 comparator=JsCode(comparator_template.format(gw_col=col)),
                                 cellStyle=jscode, flex=1, minWidth=90)

        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
               theme='streamlit-dark', height=(len(df_display) + 1) * 35,
               fit_columns_on_grid_load=True, key=f'xg_grid_{start_gw}_{end_gw}')

    # ── Tab 3: xCS ────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Expected Clean Sheets (Higher = better for defenders)")
        df_display = master_df.sort_values(by='xCS', ascending=False).reset_index().rename(columns={'index': 'Team'})
        df_display = df_display[['Team', 'xCS'] + gw_columns]

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150, sortable=True)
        gb.configure_column("xCS", header_name="Expected CS (xCS)", pinned='left',
                             valueFormatter="data['xCS'].toFixed(2)", flex=1.5,
                             type=["numericColumn"], minWidth=140, sortable=True)

        jscode = JsCode("""function(params) {
            const cellData = params.data[params.colDef.field];
            if (cellData && typeof cellData === 'object' && cellData.CS !== undefined) {
                const cs = cellData.CS;
                let bgColor;
                if (cs >= 0.5)       { bgColor = '#00ff85'; }
                else if (cs >= 0.35) { bgColor = '#50c369'; }
                else if (cs >= 0.2)  { bgColor = '#D3D3D3'; }
                else if (cs >= 0.1)  { bgColor = '#9d66a0'; }
                else                 { bgColor = '#6f2a74'; }
                const textColor = (cs >= 0.2 && cs < 0.35) ? '#31333F' : '#FFFFFF';
                return {'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'};
            }
            return {'textAlign': 'center', 'backgroundColor': '#1E1E1E', 'color': '#FF4B4B', 'fontWeight': 'bold', 'border': '1px solid #FF4B4B'};
        };""")

        comparator_template = """function(valueA, valueB, nodeA, nodeB) {{ const csA = (nodeA.data['{gw_col}'] && typeof nodeA.data['{gw_col}'] === 'object') ? nodeA.data['{gw_col}'].CS : 0; const csB = (nodeB.data['{gw_col}'] && typeof nodeB.data['{gw_col}'] === 'object') ? nodeB.data['{gw_col}'].CS : 0; return csA - csB; }}"""

        for col in gw_columns:
            gb.configure_column(col, headerName=col,
                                 valueGetter=f"(data['{col}'] && typeof data['{col}'] === 'object') ? (data['{col}'].CS * 100).toFixed(0) + '%' : 'BGW'",
                                 comparator=JsCode(comparator_template.format(gw_col=col)),
                                 cellStyle=jscode, flex=1, minWidth=90)

        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True,
               theme='streamlit-dark', height=(len(df_display) + 1) * 35,
               key=f'cs_grid_{start_gw}_{end_gw}')

    # ── Tab 4: Captain Picks ──────────────────────────────────────────────────
    with tab4:
        st.subheader("🎯 Captain Picks")

        if not live_ok:
            st.warning("⚠️ Captain Picks requires Live FPL Data. Enable the toggle in the sidebar.")
        else:
            gw_input = st.number_input("GW to analyse:", min_value=1, max_value=38,
                                        value=current_gw, key="captain_gw")

            ratings_dict_for_captain = ratings_df.set_index('Team').to_dict('index')
            captain_df = build_captain_df(bootstrap, raw_fixtures, gw_input, ratings_dict_for_captain)

            st.markdown(f"**Top captain picks for GW{gw_input}** — ranked by Form × Fixture Ease")
            st.caption("Score = Form × (6 - FDR). Higher is better.")

            # Colour the Score column
            styled = captain_df.style \
                .background_gradient(subset=["Score"], cmap="Greens") \
                .background_gradient(subset=["FDR"], cmap="RdYlGn_r") \
                .format({"£": "£{:.1f}m", "Form": "{:.1f}",
                         "Score": "{:.1f}", "Sel%": "{:.1f}%"})

            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 📊 Quick Visual")
            chart_df = captain_df[["Player", "Score"]].set_index("Player")
            st.bar_chart(chart_df)

    # ── Tab 5: Live Radar ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("📡 Live FPL Radar")

        if not live_ok:
            st.warning("⚠️ Live Radar requires Live FPL Data. Enable the toggle in the sidebar.")
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### ⚠️ DGW / BGW Radar")
                radar_end = st.number_input("Show up to GW:", min_value=current_gw,
                                             max_value=38, value=min(current_gw + 5, 38),
                                             key="radar_end_gw")
                dgw, bgw = build_dgw_bgw(raw_fixtures, bootstrap, current_gw, radar_end)

                if dgw:
                    st.markdown("**🟢 Double Gameweeks**")
                    for name, gws in sorted(dgw.items()):
                        st.write(f"- **{name}**: GW{', GW'.join(map(str, gws))}")
                else:
                    st.info("No DGWs detected in this range.")

                if bgw:
                    st.markdown("**🔴 Blank Gameweeks**")
                    for name, gws in sorted(bgw.items()):
                        st.write(f"- **{name}**: GW{', GW'.join(map(str, gws))}")
                else:
                    st.info("No BGWs detected in this range.")

            with col_b:
                st.markdown("### 📅 Upcoming Deadlines")
                events = pd.DataFrame(bootstrap["events"])
                events["deadline"] = pd.to_datetime(events["deadline_time"], utc=True)
                upcoming = events[events["id"] >= current_gw][["id", "deadline", "is_current", "is_next"]].head(8).copy()
                upcoming.rename(columns={"id": "GW"}, inplace=True)
                upcoming["deadline"] = upcoming["deadline"].dt.strftime("%a %d %b — %H:%M UTC")
                st.dataframe(upcoming, use_container_width=True, hide_index=True)

                st.markdown("### 📈 Ownership Movers (Live GW)")
                players_df = pd.DataFrame(bootstrap["elements"])
                movers = players_df[["web_name", "selected_by_percent",
                                      "transfers_in_event", "transfers_out_event"]].copy()
                movers["selected_by_percent"] = pd.to_numeric(movers["selected_by_percent"], errors="coerce")
                movers["transfers_in_event"]  = pd.to_numeric(movers["transfers_in_event"], errors="coerce")
                movers["transfers_out_event"] = pd.to_numeric(movers["transfers_out_event"], errors="coerce")

                st.markdown("**🔼 Most transferred IN this GW**")
                top_in = movers.nlargest(5, "transfers_in_event")[["web_name", "selected_by_percent", "transfers_in_event"]]
                top_in.columns = ["Player", "Sel%", "Transfers In"]
                st.dataframe(top_in, hide_index=True, use_container_width=True)

                st.markdown("**🔽 Most transferred OUT this GW**")
                top_out = movers.nlargest(5, "transfers_out_event")[["web_name", "selected_by_percent", "transfers_out_event"]]
                top_out.columns = ["Player", "Sel%", "Transfers Out"]
                st.dataframe(top_out, hide_index=True, use_container_width=True)

    # ── Easy Run Finder ───────────────────────────────────────────────────────
    st.markdown("---")
    st.sidebar.header("Easy Run Finder")
    st.sidebar.info("Find upcoming periods of 3+ easy/neutral fixtures (FDR 1-3).")
    teams_to_check = st.sidebar.multiselect("Select teams to find runs for:", PREMIER_LEAGUE_TEAMS, default=[])

    st.header("✅ Easy Fixture Runs")
    if teams_to_check:
        rating_dict  = ratings_df.set_index('Team').to_dict('index')
        all_runs     = find_fixture_runs(fixtures_df.to_dict('records'), rating_dict, start_gw)
        results_found = False
        for team in teams_to_check:
            team_runs = all_runs.get(team)
            if team_runs:
                results_found = True
                with st.expander(f"**{team}** ({len(team_runs)} matching run(s) found)"):
                    for i, run in enumerate(team_runs):
                        start_r, end_r = run[0]['gw'], run[-1]['gw']
                        st.markdown(f"**Run {i+1}: GW{start_r} - GW{end_r}**")
                        run_text = ""
                        for fix in run:
                            opp_abbr = TEAM_ABBREVIATIONS.get(fix['opp'], '???')
                            run_text += f"- **GW{fix['gw']}:** {opp_abbr} ({fix['loc']}) - FDR: {fix['fdr']} \n"
                        st.markdown(run_text)
        if not results_found:
            st.warning(f"No upcoming runs of 3+ easy/neutral fixtures found for the selected teams from GW{start_gw}.")
    else:
        st.info("Select one or more teams from the 'Easy Run Finder' in the sidebar.")

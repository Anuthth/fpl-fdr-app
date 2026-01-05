import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import math

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# Constants for the Poisson model
AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25

# Your defined FDR thresholds
FDR_THRESHOLDS = {
    5: 120.0,
    4: 110.0,
    3: 99.0,
    2: 90.0,
    1: 0
}

# User's custom 5-color FDR system
FDR_COLORS = {
    1: '#00ff85',
    2: '#50c369',
    3: '#D3D3D3',
    4: '#9d66a0',
    5: '#6f2a74'
}

# --- Team Lists and Mappings ---
PREMIER_LEAGUE_TEAMS = sorted([
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
    'Man City', 'Man Utd', 'Newcastle', 'Nottm Forest', 'Sunderland',
    'Spurs', 'West Ham', 'Wolves'
])

# This dictionary provides the 3-letter code for every possible team
TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Ipswich': 'IPS', 'Leeds': 'LEE', 
    'Leicester': 'LEI', 'Liverpool': 'LIV', 'Man City': 'MCI', 'Man Utd': 'MUN', 
    'Newcastle': 'NEW', 'Nottm Forest': 'NFO', 'Southampton': 'SOU', 
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL',
    'Tottenham Hotspur': 'TOT', 'Manchester City': 'MCI', 'Manchester United': 'MUN'
}

# This dictionary standardizes team names from your files
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
    "Wolverhampton Wanderers": "Wolves", "Wolves": "Wolves"
}

# --- Helper and Data Processing Functions ---

def get_fdr_score_from_rating(team_rating):
    """Returns an FDR score based on the predefined threshold dictionary."""
    if pd.isna(team_rating): return 3
    if team_rating >= FDR_THRESHOLDS[5]: return 5
    if team_rating >= FDR_THRESHOLDS[4]: return 4
    if team_rating >= FDR_THRESHOLDS[3]: return 3
    if team_rating >= FDR_THRESHOLDS[2]: return 2
    return 1

@st.cache_data
def load_data():
    """Loads and prepares ratings and fixtures data."""
    try:
        ratings_df = pd.read_csv(RATINGS_CSV_FILE)
        fixtures_df = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error("Ensure ratings and fixtures CSV files are in the same folder.")
        return None, None

    ratings_df['Team'] = ratings_df['Team'].map(TEAM_NAME_MAP).fillna(ratings_df['Team'])
    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    return ratings_df, fixtures_df

@st.cache_data
def create_all_data(fixtures_df, start_gw, end_gw, ratings_df, free_hit_gw=None):
    """Prepares a single, comprehensive dataframe with FDR, xG, and CS projections."""
    ratings_dict = ratings_df.set_index('Team').to_dict('index')

    pl_ratings = ratings_df[ratings_df['Team'].isin(PREMIER_LEAGUE_TEAMS)]
    avg_off_score = pl_ratings['Off Score'].mean()
    avg_def_score = pl_ratings['Def Score'].mean()

    gw_range = range(start_gw, end_gw + 1)
    projection_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team, away_team = row['HomeTeam_std'], row['AwayTeam_std']
        gw = f"GW{row['GW']}"

        home_stats = ratings_dict.get(home_team)
        away_stats = ratings_dict.get(away_team)

        if home_stats and away_stats and 'Off Score' in home_stats and 'Def Score' in away_stats:
            home_attack_strength = home_stats['Off Score'] / avg_off_score
            away_defense_weakness = avg_def_score / away_stats['Def Score']
            home_xg = home_attack_strength * away_defense_weakness * AVG_LEAGUE_HOME_GOALS

            away_attack_strength = away_stats['Off Score'] / avg_off_score
            home_defense_weakness = avg_def_score / home_stats['Def Score']
            away_xg = away_attack_strength * home_defense_weakness * AVG_LEAGUE_AWAY_GOALS

            home_cs_prob = math.exp(-away_xg)
            away_cs_prob = math.exp(-home_xg)

            if home_team in PREMIER_LEAGUE_TEAMS:
                projection_data[home_team][gw] = {
                    "display": f"{TEAM_ABBREVIATIONS.get(away_team, '???')} (H)",
                    "fdr": get_fdr_score_from_rating(away_stats.get('Final Rating')),
                    "xG": home_xg, "CS": home_cs_prob
                }
            if away_team in PREMIER_LEAGUE_TEAMS:
                projection_data[away_team][gw] = {
                    "display": f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)",
                    "fdr": get_fdr_score_from_rating(home_stats.get('Final Rating')),
                    "xG": away_xg, "CS": away_cs_prob
                }

    df = pd.DataFrame.from_dict(projection_data, orient='index').reindex(columns=[f'GW{i}' for i in gw_range])

    free_hit_col = f'GW{free_hit_gw}' if free_hit_gw else None

    total_difficulty, total_xg, total_cs = [], [], []
    for index, row in df.iterrows():
        fdr_sum, xg_sum, cs_sum = 0, 0, 0
        for gw_col, cell_data in row.items():
            if gw_col != free_hit_col and isinstance(cell_data, dict):
                fdr_sum += cell_data.get('fdr', 0)
                xg_sum += cell_data.get('xG', 0)
                cs_sum += cell_data.get('CS', 0)
        total_difficulty.append(fdr_sum)
        total_xg.append(xg_sum)
        total_cs.append(cs_sum)

    df['Total Difficulty'] = total_difficulty
    df['Total xG'] = total_xg
    df['xCS'] = total_cs

    return df

@st.cache_data
def find_fixture_runs(fixtures_df, rating_dict, start_gw):
    """Scans for runs of 3+ games with an FDR of 3 or less."""
    all_fixtures = {team: [] for team in PREMIER_LEAGUE_TEAMS}
    for gw in range(1, 39):
        gw_fixtures = fixtures_df[fixtures_df['GW'] == gw]
        for _, row in gw_fixtures.iterrows():
            home_team, away_team = row['HomeTeam_std'], row['AwayTeam_std']
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
                    if team not in good_runs: good_runs[team] = []
                    good_runs[team].append(current_run)
                current_run = []

        if len(current_run) >= 3:
            if team not in good_runs: good_runs[team] = []
            good_runs[team].append(current_run)

    return good_runs

# --- Main Streamlit App ---

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("FPL Fixture Planner")

with st.expander("Glossary & How It Works"):
    st.markdown("""
    - **FDR:** Fixture Difficulty Rating (1-5). Lower is better.
    - **xG:** Projected Goals. Higher is better for attackers.
    - **xCS:** Expected Clean Sheets. Higher is better for defenders.
    """)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Manual clear cache button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Rerun", use_container_width=True):
            st.rerun()

ratings_df, fixtures_df = load_data()

if ratings_df is not None and fixtures_df is not None:
    st.sidebar.header("Controls")
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_gw = st.number_input("Start GW:", min_value=21, max_value=38, value=21)
    with col_end:
        end_gw = st.number_input("End GW:", min_value=21, max_value=38, value=30)

    selected_teams = st.sidebar.multiselect("Select teams to display:", PREMIER_LEAGUE_TEAMS, default=PREMIER_LEAGUE_TEAMS)
    fh_options = [None] + list(range(start_gw, end_gw + 1))
    free_hit_gw = st.sidebar.selectbox(
        "Select Free Hit Gameweek (optional):",
        options=fh_options,
        format_func=lambda x: "None" if x is None else f"GW{x}"
    )

    master_df = create_all_data(fixtures_df, start_gw, end_gw, ratings_df, free_hit_gw)

    if selected_teams:
        teams_to_show = [team for team in master_df.index if team in selected_teams]
        master_df = master_df.loc[teams_to_show]

    tab1, tab2, tab3 = st.tabs(["Fixture Difficulty (FDR)", "Projected Goals (xG)", "Expected Clean Sheets (xCS)"])

    gw_columns = [f'GW{i}' for i in range(start_gw, end_gw + 1)]
    if free_hit_gw:
        gw_columns.remove(f'GW{free_hit_gw}')

    with tab1:
        st.subheader("Fixture Difficulty Rating (Lower score is better)")
        df_display = master_df.sort_values(by='Total Difficulty', ascending=True).reset_index().rename(columns={'index': 'Team'})
        
        column_order = ['Team', 'Total Difficulty'] + gw_columns
        df_display = df_display[column_order]
        
        df_for_grid = df_display[['Team', 'Total Difficulty']].copy()
        
        for col in gw_columns:
            df_for_grid[col] = df_display[col]
            df_for_grid[f'{col}_display'] = df_display[col].apply(
                lambda x: x['display'] if isinstance(x, dict) and 'display' in x else ''
            )
            df_for_grid[f'{col}_fdr'] = df_display[col].apply(
                lambda x: x['fdr'] if isinstance(x, dict) and 'fdr' in x else 3
            )
        
        gb = GridOptionsBuilder.from_dataframe(df_for_grid)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150, sortable=True)
        gb.configure_column("Total Difficulty", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)
        
        for col in gw_columns:
            gb.configure_column(f'{col}_display', hide=True)
            gb.configure_column(f'{col}_fdr', hide=True)
        
        for col in gw_columns:
            value_formatter = f"""function(params) {{
                return params.data['{col}_display'] || '';
            }}"""
            
            jscode_for_col = f"""function(params) {{
                const fdrValue = params.data['{col}_fdr'];
                if (fdrValue !== undefined && fdrValue !== null) {{
                    const colors = {{1: '#00ff85', 2: '#50c369', 3: '#D3D3D3', 4: '#9d66a0', 5: '#6f2a74'}};
                    const bgColor = colors[fdrValue] || '#444444';
                    const textColor = (fdrValue <= 3) ? '#31333F' : '#FFFFFF';
                    return {{'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold', 'textAlign': 'center'}};
                }}
                return {{'textAlign': 'center', 'backgroundColor': '#444444'}};
            }}"""
            
            value_getter = f"""function(params) {{
                return params.data['{col}_fdr'];
            }}"""
            
            gb.configure_column(
                col,
                headerName=col,
                valueGetter=JsCode(value_getter),
                valueFormatter=JsCode(value_formatter),
                cellStyle=JsCode(jscode_for_col),
                flex=1,
                minWidth=90,
                sortable=True
            )
        
        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        
        AgGrid(
            df_for_grid, 
            gridOptions=gb.build(), 
            allow_unsafe_jscode=True, 
            theme='streamlit-dark', 
            height=(len(df_for_grid) + 1) * 35, 
            fit_columns_on_grid_load=True, 
            key=f'fdr_grid_{start_gw}_{end_gw}'
        )
            
    with tab2:
        st.subheader("Projected Goals (Higher is better for attackers)")
        df_display = master_df.sort_values(by='Total xG', ascending=False).reset_index().rename(columns={'index': 'Team'})

        gw_columns_in_df = [col for col in df_display.columns if col.startswith('GW')]
        cols_to_display = ['Team', 'Total xG'] + gw_columns_in_df
        df_display = df_display[cols_to_display]

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])

        gb.configure_column("Team", pinned='left', cellStyle={'textAlign': 'left'}, flex=2, minWidth=150, sortable=True)
        gb.configure_column("Total xG", valueFormatter="data['Total xG'].toFixed(2)", flex=1.5, type=["numericColumn"],minWidth=140, sortable=True)
        gb.configure_column("Total Difficulty", hide=True)
        gb.configure_column("xCS", hide=True)

        jscode = JsCode("""function(params) { const cellData = params.data[params.colDef.field]; if (cellData && cellData.xG !== undefined) { const xG = cellData.xG; let bgColor; if (xG >= 2.0) { bgColor = '#63be7b'; } else if (xG >= 1.5) { bgColor = '#95d2a6'; } else if (xG >= 1.0) { bgColor = '#bfe4cb'; } else if (xG >= 0.5) { bgColor = '#D3D3D3'; } else { bgColor = '#D3D3D3'; } const textColor = (xG >= 0.0 && xG < 5.0) ? '#31333F' : '#FFFFFF'; return {'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'}; } return {'textAlign': 'center', 'backgroundColor': '#444444'}; };""")
        comparator_template = """function(valueA, valueB, nodeA, nodeB) {{ const xgA = nodeA.data['{gw_col}'] ? nodeA.data['{gw_col}'].xG : 0; const xgB = nodeB.data['{gw_col}'] ? nodeB.data['{gw_col}'].xG : 0; return xgA - xgB; }}"""

        for col in gw_columns_in_df:
            gb.configure_column(col, headerName=col, valueGetter=f"data['{col}'] ? data['{col}'].xG.toFixed(2) : ''", comparator=JsCode(comparator_template.format(gw_col=col)), cellStyle=jscode, flex=1, minWidth=90)

        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True, theme='streamlit-dark', height=(len(df_display) + 1) * 35, fit_columns_on_grid_load=True, key=f'xg_grid_{start_gw}_{end_gw}')

    with tab3:
        st.subheader("Expected Clean Sheets (Higher is better for defenders)")
        df_display = master_df.sort_values(by='xCS', ascending=False).reset_index().rename(columns={'index': 'Team'})

        gw_columns_in_df = [col for col in df_display.columns if col.startswith('GW')]
        cols_to_display = ['Team', 'xCS'] + gw_columns_in_df
        df_display = df_display[cols_to_display]

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_column("Team", pinned='left', flex=2, minWidth=150, sortable=True)
        gb.configure_column("xCS", header_name="Expected CS (xCS)", pinned='left', valueFormatter="data['xCS'].toFixed(2)", flex=1.5, type=["numericColumn"], minWidth=140, sortable=True)

        jscode = JsCode("""function(params) { const cellData = params.data[params.colDef.field]; if (cellData && cellData.CS !== undefined) { const cs = cellData.CS; let bgColor; if (cs >= 0.5) { bgColor = '#00ff85'; } else if (cs >= 0.35) { bgColor = '#50c369'; } else if (cs >= 0.2) { bgColor = '#D3D3D3'; } else if (cs >= 0.1) { bgColor = '#9d66a0'; } else { bgColor = '#6f2a74'; } const textColor = (cs >= 0.2 && cs < 0.35) ? '#31333F' : '#FFFFFF'; return {'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'}; } return {'textAlign': 'center', 'backgroundColor': '#444444'}; };""")
        comparator_template = """function(valueA, valueB, nodeA, nodeB) {{ const csA = nodeA.data['{gw_col}'] ? nodeA.data['{gw_col}'].CS : 0; const csB = nodeB.data['{gw_col}'] ? nodeB.data['{gw_col}'].CS : 0; return csA - csB; }}"""

        for col in gw_columns_in_df:
            gb.configure_column(col, headerName=col, valueGetter=f"data['{col}'] ? (data['{col}'].CS * 100).toFixed(0) + '%' : ''", comparator=JsCode(comparator_template.format(gw_col=col)), cellStyle=jscode, flex=1, minWidth=90)

        gb.configure_default_column(resizable=True, sortable=True, filter=False, menuTabs=[])
        AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True, theme='streamlit-dark', height=(len(df_display) + 1) * 35, key=f'cs_grid_{start_gw}_{end_gw}')
    
    # --- Easy Run Finder Feature ---
    st.markdown("---") 
    st.sidebar.header("Easy Run Finder")
    st.sidebar.info("Find upcoming periods of 3+ easy/neutral fixtures (FDR 1-3).")

    teams_to_check = st.sidebar.multiselect("Select teams to find runs for:", PREMIER_LEAGUE_TEAMS, default=[])

    st.header("✅ Easy Fixture Runs")

    if teams_to_check:
        rating_dict = ratings_df.set_index('Team').to_dict('index')
        all_runs = find_fixture_runs(fixtures_df, rating_dict, start_gw)

        results_found = False
        for team in teams_to_check:
            team_runs = all_runs.get(team)
            if team_runs:
                results_found = True
                with st.expander(f"**{team}** ({len(team_runs)} matching run(s) found)"):
                    for i, run in enumerate(team_runs):
                        start_run, end_run = run[0]['gw'], run[-1]['gw']
                        st.markdown(f"**Run {i+1}: GW{start_run} - GW{end_run}**")
                        run_text = ""
                        for fix in run:
                            opp_abbr = TEAM_ABBREVIATIONS.get(fix['opp'], '???')
                            run_text += f"- **GW{fix['gw']}:** {opp_abbr} ({fix['loc']}) - FDR: {fix['fdr']} \n"
                        st.markdown(run_text)

        if not results_found:
            st.warning(f"No upcoming runs of 3+ easy/neutral fixtures found for the selected teams, starting from GW{start_gw}.")
    else:
        st.info("Select one or more teams from the 'Easy Run Finder' in the sidebar to check for their favorable fixture periods.")
else:
    st.error("Data could not be loaded. Please check your CSV files.")

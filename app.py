import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import math

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# Your defined FDR thresholds
FDR_THRESHOLDS = {
    5: 115.0,
    4: 90.0,
    3: 80.0,
    2: 70.0,
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
TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Leeds': 'LEE', 'Liverpool': 'LIV',
    'Man City': 'MCI', 'Man Utd': 'MUN', 'Newcastle': 'NEW', 'Nottm Forest': 'NFO',
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL',
    'Tottenham Hotspur': 'TOT', 'Manchester City': 'MCI', 'Manchester United': 'MUN'
}
TEAM_NAME_MAP = {
    "A.F.C. Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds", "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nottm Forest",
    "Tottenham Hotspur": "Spurs", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
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
    ratings_df['Team'] = ratings_df['Team'].replace(TEAM_NAME_MAP)
    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    return ratings_df, fixtures_df

def create_all_data(fixtures_df, start_gw, end_gw, ratings_df):
    """Prepares a single, comprehensive dataframe with FDR, xG, and CS projections."""
    ratings_dict = ratings_df.set_index('Team').to_dict('index')
    gw_range = range(start_gw, end_gw + 1)
    projection_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}
    
    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team, away_team = row['HomeTeam_std'], row['AwayTeam_std']
        gw = f"GW{row['GW']}"

        home_stats = ratings_dict.get(home_team)
        away_stats = ratings_dict.get(away_team)

        if home_stats and away_stats and 'Off Score' in home_stats and 'Def Score' in away_stats:
            home_xg = home_stats['Off Score'] / away_stats['Def Score']
            away_xg = away_stats['Off Score'] / home_stats['Def Score']
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
    
    df['Total Difficulty'] = df.apply(lambda row: sum(cell['fdr'] for cell in row if isinstance(cell, dict) and 'fdr' in cell), axis=1)
    df['Total xG'] = df.apply(lambda row: sum(cell['xG'] for cell in row if isinstance(cell, dict) and 'xG' in cell), axis=1)
    # MODIFIED: Changed column name from 'Total CS' to 'xCS'
    df['xCS'] = df.apply(lambda row: sum(cell['CS'] for cell in row if isinstance(cell, dict) and 'CS' in cell), axis=1)
    
    return df

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("FPL Fixture Planner")

with st.expander("Glossary & How It Works"):
    st.markdown(f"""
    - **FDR:** Fixture Difficulty Rating (1-5). Lower is better.
    - **xG:** Projected Goals a team might score. Higher is better for attackers.
    - **xCS:** Expected Clean Sheets. The total number of clean sheets a team is expected to keep over the period. Higher is better for defenders.
    """)

ratings_df, fixtures_df = load_data()

if ratings_df is not None and fixtures_df is not None:
    st.sidebar.header("Controls")

    start_gw, end_gw = st.sidebar.slider(
        "Select Gameweek Range:",
        min_value=1,
        max_value=38,
        value=(1, 8)
    )
    
    selected_teams = st.sidebar.multiselect(
        "Select teams to display:",
        options=PREMIER_LEAGUE_TEAMS,
        default=PREMIER_LEAGUE_TEAMS
    )

    master_df = create_all_data(fixtures_df, start_gw, end_gw, ratings_df)
    
    if selected_teams:
        teams_to_show = [team for team in master_df.index if team in selected_teams]
        master_df = master_df.loc[teams_to_show]

    tab1, tab2, tab3 = st.tabs(["Fixture Difficulty (FDR)", "Projected Goals (xG)", "Expected Clean Sheets (xCS)"])

    with tab1:
        st.subheader("Fixture Difficulty Rating (Lower score is better)")
        fdr_df = master_df.sort_values(by='Total Difficulty', ascending=True)
        fdr_df = fdr_df.reset_index().rename(columns={'index': 'Team'})
        
        gb_fdr = GridOptionsBuilder.from_dataframe(fdr_df)
        gb_fdr.configure_column("Team", width=150, pinned='left', cellStyle={'textAlign': 'left'})
        gb_fdr.configure_column("Total Difficulty", width=120)
        gb_fdr.configure_column("Total xG", hide=True); gb_fdr.configure_column("xCS", hide=True)

        jscode_fdr = JsCode(f"""function(params) {{ const cellData = params.data[params.colDef.field]; if (cellData && cellData.fdr !== undefined) {{ const fdr = cellData.fdr; const colors = {FDR_COLORS}; const bgColor = colors[fdr] || '#444444'; const textColor = (fdr <= 3) ? '#31333F' : '#FFFFFF'; return {{'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'}}; }} return {{'textAlign': 'center', 'backgroundColor': '#444444'}}; }};""")
        for gw in range(start_gw, end_gw + 1):
            gw_col = f"GW{gw}"; gb_fdr.configure_column(gw_col, headerName=gw_col, valueGetter=f"data['{gw_col}'] ? data['{gw_col}'].display : ''", cellStyle=jscode_fdr, width=100)
        
        gb_fdr.configure_default_column(resizable=True, sortable=False, filter=False, menuTabs=[])
        AgGrid(fdr_df, gridOptions=gb_fdr.build(), allow_unsafe_jscode=True, theme='streamlit-dark', height=(len(fdr_df) + 1) * 35, fit_columns_on_grid_load=True, key='fdr_grid')

    with tab2:
        st.subheader("Projected Goals (Higher is better for attackers)")
        xg_df = master_df.sort_values(by='Total xG', ascending=False)
        xg_df = xg_df.reset_index().rename(columns={'index': 'Team'})

        gb_xg = GridOptionsBuilder.from_dataframe(xg_df)
        gb_xg.configure_column("Team", width=150, pinned='left', cellStyle={'textAlign': 'left'})
        gb_xg.configure_column("Total xG", width=120, valueFormatter="data['Total xG'].toFixed(2)")
        gb_xg.configure_column("Total Difficulty", hide=True); gb_xg.configure_column("xCS", hide=True)

        jscode_xg = JsCode("""function(params) { const cellData = params.data[params.colDef.field]; if (cellData && cellData.xG !== undefined) { const xG = cellData.xG; let bgColor; if (xG >= 1.8) { bgColor = '#00ff85'; } else if (xG >= 1.2) { bgColor = '#50c369'; } else if (xG >= 0.8) { bgColor = '#D3D3D3'; } else if (xG >= 0.5) { bgColor = '#9d66a0'; } else { bgColor = '#6f2a74'; } const textColor = (xG >= 0.8 && xG < 1.2) ? '#31333F' : '#FFFFFF'; return {'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'}; } return {'textAlign': 'center', 'backgroundColor': '#444444'}; };""")
        for gw in range(start_gw, end_gw + 1):
            gw_col = f"GW{gw}"; gb_xg.configure_column(gw_col, headerName=gw_col, valueGetter=f"data['{gw_col}'] ? data['{gw_col}'].xG.toFixed(2) : ''", cellStyle=jscode_xg, width=100)
        
        gb_xg.configure_default_column(resizable=True, sortable=False, filter=False, menuTabs=[])
        AgGrid(xg_df, gridOptions=gb_xg.build(), allow_unsafe_jscode=True, theme='streamlit-dark', height=(len(xg_df) + 1) * 35, fit_columns_on_grid_load=True, key='xg_grid')
        
    with tab3:
        # MODIFIED: Changed subheader and sort column
        st.subheader("Expected Clean Sheets (Higher is better for defenders)")
        cs_df = master_df.sort_values(by='xCS', ascending=False)
        cs_df = cs_df.reset_index().rename(columns={'index': 'Team'})

        gb_cs = GridOptionsBuilder.from_dataframe(cs_df)
        gb_cs.configure_column("Team", width=150, pinned='left', cellStyle={'textAlign': 'left'})
        # MODIFIED: Changed column name and formatter
        gb_cs.configure_column("xCS", header_name="Expected CS (xCS)", width=120, valueFormatter="data['xCS'].toFixed(2)")
        gb_cs.configure_column("Total Difficulty", hide=True); gb_cs.configure_column("Total xG", hide=True)
        
        jscode_cs = JsCode("""function(params) { const cellData = params.data[params.colDef.field]; if (cellData && cellData.CS !== undefined) { const cs = cellData.CS; let bgColor; if (cs >= 0.5) { bgColor = '#00ff85'; } else if (cs >= 0.35) { bgColor = '#50c369'; } else if (cs >= 0.2) { bgColor = '#D3D3D3'; } else if (cs >= 0.1) { bgColor = '#9d66a0'; } else { bgColor = '#6f2a74'; } const textColor = (cs >= 0.2 && cs < 0.35) ? '#31333F' : '#FFFFFF'; return {'backgroundColor': bgColor, 'color': textColor, 'fontWeight': 'bold'}; } return {'textAlign': 'center', 'backgroundColor': '#444444'}; };""")
        for gw in range(start_gw, end_gw + 1):
            gw_col = f"GW{gw}"; gb_cs.configure_column(gw_col, headerName=gw_col, valueGetter=f"data['{gw_col}'] ? (data['{gw_col}'].CS * 100).toFixed(0) + '%' : ''", cellStyle=jscode_cs, width=100)
        
        gb_cs.configure_default_column(resizable=True, sortable=False, filter=False, menuTabs=[])
        AgGrid(cs_df, gridOptions=gb_cs.build(), allow_unsafe_jscode=True, theme='streamlit-dark', height=(len(cs_df) + 1) * 35, fit_columns_on_grid_load=True, key='cs_grid')

else:
    st.error("Data could not be loaded. Please check your CSV files.")

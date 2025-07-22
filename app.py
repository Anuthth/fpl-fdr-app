import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

def create_fdr_data(fixtures_df, start_gw, end_gw, rating_dict):
    """Prepares the dataframes for the FDR table."""
    gw_range = range(start_gw, end_gw + 1)
    gw_columns = [f'GW{i}' for i in gw_range]

    combined_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}
    fdr_score_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team, away_team, gw = row['HomeTeam_std'], row['AwayTeam_std'], f"GW{row['GW']}"
        if home_team in PREMIER_LEAGUE_TEAMS:
            fdr = get_fdr_score_from_rating(rating_dict.get(away_team))
            combined_data[home_team][gw] = {"display": f"{TEAM_ABBREVIATIONS.get(away_team, '???')} (H)", "fdr": fdr}
            fdr_score_data[home_team][gw] = fdr
        if away_team in PREMIER_LEAGUE_TEAMS:
            fdr = get_fdr_score_from_rating(rating_dict.get(home_team))
            combined_data[away_team][gw] = {"display": f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)", "fdr": fdr}
            fdr_score_data[away_team][gw] = fdr
    
    df = pd.DataFrame.from_dict(combined_data, orient='index').reindex(columns=gw_columns)
    
    fdr_score_df = pd.DataFrame.from_dict(fdr_score_data, orient='index').reindex(columns=gw_columns)
    df['Total Difficulty'] = fdr_score_df.sum(axis=1)
    df.sort_values(by='Total Difficulty', ascending=True, inplace=True)
    
    return df

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("FPL Fixture Difficulty")

with st.expander("Glossary & How It Works"):
    st.markdown(f"""
    - **FDR (Fixture Difficulty Rating):** Each fixture is rated 1-5 based on the opponent's 'Final Rating'.
    - **Your Custom Thresholds:** FDR 5 (Rating ≥ {FDR_THRESHOLDS[5]}), FDR 4 (≥ {FDR_THRESHOLDS[4]}), FDR 3 (≥ {FDR_THRESHOLDS[3]}), FDR 2 (≥ {FDR_THRESHOLDS[2]}), FDR 1 (all others).
    - **Total Difficulty:** The sum of the FDR scores for all fixtures in the selected range. A **lower** number indicates an easier run of matches.
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

    rating_col = 'Hybrid Rating' if 'Hybrid Rating' in ratings_df.columns else 'Final Rating'
    rating_dict = ratings_df.set_index('Team')[rating_col].to_dict()

    fdr_df = create_fdr_data(fixtures_df, start_gw, end_gw, rating_dict)
    
    if selected_teams:
        teams_to_show = [team for team in fdr_df.index if team in selected_teams]
        fdr_df = fdr_df.loc[teams_to_show]

    if not fdr_df.empty:
        fdr_df.reset_index(inplace=True)
        fdr_df.rename(columns={'index': 'Team'}, inplace=True)

        gb = GridOptionsBuilder.from_dataframe(fdr_df)
        
        gb.configure_column("Team", width=150, pinned='left', cellStyle={'textAlign': 'left'})
        gb.configure_column("Total Difficulty", width=120)

        # --- FINAL FIX: Corrected the JavaScript to access the full cell data ---
        jscode = JsCode(f"""
        function(params) {{
            // Get the full data for the cell, not just the display value
            const cellData = params.data[params.colDef.field]; 
            if (cellData && cellData.fdr !== undefined) {{
                const fdr = cellData.fdr;
                const colors = {FDR_COLORS};
                const bgColor = colors[fdr] || '#444444';
                const textColor = (fdr <= 3) ? '#31333F' : '#FFFFFF';
                return {{
                    'backgroundColor': bgColor,
                    'color': textColor,
                    'fontWeight': 'bold'
                }};
            }}
            // Default style for blank cells
            return {{'textAlign': 'center', 'backgroundColor': '#444444'}};
        }};
        """)
        
        comparator_jscode_template = """
        function(valueA, valueB, nodeA, nodeB) {{
            const fdrA = nodeA.data['{gw_col}'] ? nodeA.data['{gw_col}'].fdr : 3;
            const fdrB = nodeB.data['{gw_col}'] ? nodeB.data['{gw_col}'].fdr : 3;
            return fdrA - fdrB;
        }}
        """

        for gw in range(start_gw, end_gw + 1):
            gw_col = f"GW{gw}"
            js_string = comparator_jscode_template.format(gw_col=gw_col)
            comparator_jscode = JsCode(js_string)

            gb.configure_column(
                gw_col,
                headerName=gw_col,
                valueGetter=f"data['{gw_col}'] ? data['{gw_col}'].display : ''",
                comparator=comparator_jscode,
                cellStyle=jscode,
                width=100
            )
        
        gb.configure_default_column(
            resizable=True, sortable=True, filter=False, menuTabs=[]
        )
        
        gridOptions = gb.build()
        
        AgGrid(
            fdr_df,
            gridOptions=gridOptions,
            allow_unsafe_jscode=True,
            theme='streamlit-dark',
            height=(len(fdr_df) + 1) * 35,
            fit_columns_on_grid_load=True
        )

    elif not selected_teams:
        st.warning("Please select at least one team from the sidebar to display the fixtures.")
else:
    st.error("Data could not be loaded. Please check your CSV files.")

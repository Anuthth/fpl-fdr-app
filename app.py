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
BLANK_FIXTURE_COLOR = '#444444'

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

    # This will now hold a dictionary with both display text and the FDR score
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
    
    # We create a multi-level column dataframe for display, then flatten it
    # This is a trick to keep the display text and the sortable FDR value together
    df = pd.DataFrame.from_dict({
        (team, gw): data for team, gw_data in combined_data.items() for gw, data in gw_data.items()
    }, orient='index')
    df = df.unstack(level=1)
    df.columns = df.columns.map('{0[0]}|{0[1]}'.format) # Flatten columns to 'GW1|display', 'GW1|fdr'

    # Create the score dataframe for calculating total difficulty
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

    # Create the combined data
    fdr_df = create_fdr_data(fixtures_df, start_gw, end_gw, rating_dict)
    
    # Filter by selected teams
    if selected_teams:
        teams_to_show = [team for team in fdr_df.index if team in selected_teams]
        fdr_df = fdr_df.loc[teams_to_show]

    if not fdr_df.empty:
        # --- NEW: AG-Grid Implementation ---
        
        # Reset index to make Team a column
        fdr_df.reset_index(inplace=True)
        fdr_df.rename(columns={'index': 'Team'}, inplace=True)

        gb = GridOptionsBuilder.from_dataframe(fdr_df)
        
        # Configure the Total Difficulty column
        gb.configure_column("Total Difficulty", width=90, headerClass='total-difficulty-header')
        
        # Configure the Team column
        gb.configure_column("Team", width=150, pinned='left', headerClass='team-header', cellClass='team-cell')

        # Cell styling logic using JavaScript
        cell_style_js = JsCode(f"""
        function(params) {{
            if (params.data && params.colDef.field.includes('|fdr')) {{
                const fdr = params.value;
                const colors = {FDR_COLORS};
                const textColor = (fdr <= 3) ? '#31333F' : '#FFFFFF';
                return {{
                    'backgroundColor': colors[fdr],
                    'color': textColor,
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }};
            }}
            return {{'textAlign': 'center'}};
        }}
        """)

        # Configure all the gameweek columns
        for gw in range(start_gw, end_gw + 1):
            gw_col = f"GW{gw}|display"
            fdr_col = f"GW{gw}|fdr"
            gb.configure_column(
                gw_col,
                headerName=f"GW{gw}", # Set the visible header name
                valueGetter=f"data['{gw_col}']", # Get the display text
                comparator=JsCode(f"function(valueA, valueB, nodeA, nodeB) {{ return nodeA.data['{fdr_col}'] - nodeB.data['{fdr_col}']; }}"),
                cellStyle=cell_style_js,
                width=80
            )
            # Hide the underlying FDR data column
            gb.configure_column(fdr_col, hide=True)
        
        # Build the grid options
        go = gb.build()
        
        # Define custom CSS for headers and team column
        custom_css = {
            ".ag-theme-streamlit-dark .total-difficulty-header": {"color": "#FFFFFF !important"},
            ".ag-theme-streamlit-dark .team-header": {"color": "#FFFFFF !important"},
            ".ag-theme-streamlit-dark .team-cell": {"text-align": "left !important"},
        }

        # Display the AgGrid table
        AgGrid(
            fdr_df,
            gridOptions=go,
            custom_css=custom_css,
            allow_unsafe_jscode=True,
            theme='streamlit-dark',
            height= (len(fdr_df) + 1) * 35,
            fit_columns_on_grid_load=True # Adjust columns to fit width
        )

    elif not selected_teams:
        st.warning("Please select at least one team from the sidebar to display the fixtures.")
else:
    st.error("Data could not be loaded. Please check your CSV files.")


import pandas as pd
import streamlit as st
import numpy as np
from matplotlib.colors import to_hex

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"
STARTING_GAMEWEEK = 1

# FINAL UI: Your original 5-color FDR system
FDR_COLORS = {
    1: '#2ECC71',  # Very Green (Easiest)
    2: '#90EE90',  # Light Green
    3: '#D3D3D3',  # Light Gray (Neutral)
    4: '#F08080',  # Light Red
    5: '#E74C3C'   # Very Red (Hardest)
}
BLANK_FIXTURE_COLOR = '#444444' # Color for empty cells

# Team Abbreviations and Name Maps
TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Leeds': 'LEE', 'Liverpool': 'LIV',
    'Luton': 'LUT', 'Man City': 'MCI', 'Man Utd': 'MUN', 'Newcastle': 'NEW',
    'Nottm Forest': 'NFO', 'Sheffield Utd': 'SHU', 'Sunderland': 'SUN',
    'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL',
    'Tottenham Hotspur': 'TOT', 'Manchester City': 'MCI', 'Manchester United': 'MUN'
}
PREMIER_LEAGUE_TEAMS = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool', 'Luton',
    'Man City', 'Man Utd', 'Newcastle', 'Nottm Forest', 'Sheffield Utd',
    'Spurs', 'West Ham', 'Wolves'
]
TEAM_NAME_MAP = {
    "A.F.C. Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
    "Luton Town": "Luton", "Leeds United": "Leeds", "Manchester City": "Man City",
    "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottm Forest", "Tottenham Hotspur": "Spurs",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd"
}

# --- Data Processing Functions ---

@st.cache_data
def load_data():
    """Loads and prepares ratings and fixtures data."""
    try:
        ratings_df = pd.read_csv(RATINGS_CSV_FILE)
        fixtures_df = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error("Ensure ratings and fixtures CSV files are in the same folder.")
        return None, None

    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    return ratings_df, fixtures_df

def create_fdr_data(ratings_df, fixtures_df, num_gws, start_gw):
    """Prepares the dataframes needed for the FDR table using the 5-color system."""
    rating_col = 'Hybrid Rating' if 'Hybrid Rating' in ratings_df.columns else 'Final Rating'
    
    # Filter for PL teams and create rating dictionary
    pl_ratings_df = ratings_df[ratings_df['Team'].isin(PREMIER_LEAGUE_TEAMS)]
    rating_dict = pl_ratings_df.set_index('Team')[rating_col].to_dict()

    # Create the 1-5 FDR score function based on quintiles of PL team ratings
    rating_values = sorted(rating_dict.values())
    quintiles = np.percentile(rating_values, [0, 20, 40, 60, 80, 100])
    def get_fdr_score(team_rating):
        if pd.isna(team_rating): return 3
        if team_rating <= quintiles[1]: return 1
        if team_rating <= quintiles[2]: return 2
        if team_rating <= quintiles[3]: return 3
        if team_rating <= quintiles[4]: return 4
        return 5

    gw_range = range(start_gw, start_gw + num_gws)
    gw_columns = [f'GW{i}' for i in gw_range]

    display_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}
    fdr_score_data = {team: {} for team in PREMIER_LEAGUE_TEAMS}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team, away_team = row['HomeTeam_std'], row['AwayTeam_std']
        gw = f"GW{row['GW']}"

        if home_team in PREMIER_LEAGUE_TEAMS:
            opponent_abbr = TEAM_ABBREVIATIONS.get(away_team, '???')
            display_data[home_team][gw] = f"{opponent_abbr} (H)"
            fdr_score_data[home_team][gw] = get_fdr_score(rating_dict.get(away_team))
        
        if away_team in PREMIER_LEAGUE_TEAMS:
            opponent_abbr = TEAM_ABBREVIATIONS.get(home_team, '???')
            display_data[away_team][gw] = f"{opponent_abbr} (A)"
            fdr_score_data[away_team][gw] = get_fdr_score(rating_dict.get(home_team))

    display_df = pd.DataFrame.from_dict(display_data, orient='index').reindex(columns=gw_columns)
    fdr_score_df = pd.DataFrame.from_dict(fdr_score_data, orient='index').reindex(columns=gw_columns)
    
    # Calculate total difficulty score by SUMMING the 1-5 FDR scores
    display_df['Score'] = fdr_score_df.sum(axis=1)
    
    display_df.sort_values(by='Score', ascending=True, inplace=True)
    cols = ['Score'] + [col for col in display_df.columns if col != 'Score']
    display_df = display_df[cols]
    
    return display_df, fdr_score_df.reindex(display_df.index)

# --- Styling Functions ---

def style_fdr_table(display_df, fdr_score_df):
    """Applies CSS styling to the FDR table using the 5-color system."""
    
    def color_cells(fdr_score):
        """Applies background color based on the 1-5 FDR score."""
        if pd.isna(fdr_score):
            return f'background-color: {BLANK_FIXTURE_COLOR}'
        
        # Get color from the predefined dictionary
        color = FDR_COLORS.get(fdr_score, '#FFFFFF') # Default to white
        
        # Determine text color for readability
        text_color = '#31333F' if fdr_score == 3 else '#FFFFFF' # Dark text for grey cells
        return f'background-color: {color}; color: {text_color}'

    styler = display_df.style

    gw_cols = [col for col in display_df.columns if 'GW' in col]
    subset_scores = fdr_score_df[gw_cols]
    
    # Apply the 5-color styling
    styler = styler.apply(lambda x: subset_scores.map(color_cells), axis=None, subset=gw_cols)
    
    # Remove gradient from score and just format it
    styler = styler.format({'Score': '{:.0f}'})

    # Set alignment for the whole table
    styler = styler.set_table_styles([
        {'selector': 'th.row_heading', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])

    return styler

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("FPL Fixture Difficulty")

ratings_df, fixtures_df = load_data()

if ratings_df is not None and fixtures_df is not None:
    st.sidebar.header("Controls")
    num_gws_to_show = st.sidebar.number_input(
        "Select number of gameweeks to view:",
        min_value=1,
        max_value=12,
        value=8,
        step=1
    )

    display_df, fdr_score_df = create_fdr_data(ratings_df, fixtures_df, num_gws_to_show, STARTING_GAMEWEEK)

    display_df.reset_index(inplace=True)
    display_df.rename(columns={'index': 'Team'}, inplace=True)
    
    styled_table = style_fdr_table(display_df.set_index('Team'), fdr_score_df)

    # FINAL UI: Calculate height to show all teams without scrolling
    # (Number of rows + 1 for header) * height per row in pixels
    table_height = (len(display_df) + 1) * 35

    st.dataframe(styled_table, use_container_width=True, height=table_height)

else:
    st.error("Data could not be loaded. Please check your CSV files.")

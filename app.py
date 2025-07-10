import pandas as pd
import streamlit as st
import numpy as np

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"
STARTING_GAMEWEEK = 1

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

def get_fdr_score_from_rating(team_rating, thresholds):
    """Returns an FDR score based on a dynamic threshold dictionary."""
    if pd.isna(team_rating):
        return 3
    # Checks from highest to lowest threshold
    if team_rating >= thresholds[4]: return 5
    if team_rating >= thresholds[3]: return 4
    if team_rating >= thresholds[2]: return 3
    if team_rating >= thresholds[1]: return 2
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

def create_fdr_data(fixtures_df, num_gws, start_gw, rating_dict, fdr_score_func):
    """Prepares the dataframes for the FDR table."""
    gw_range = range(start_gw, start_gw + num_gws)
    gw_columns = [f'GW{i}' for i in gw_range]

    display_data, fdr_score_data = {}, {}
    for team in PREMIER_LEAGUE_TEAMS:
        display_data[team], fdr_score_data[team] = {}, {}

    for _, row in fixtures_df[fixtures_df['GW'].isin(gw_range)].iterrows():
        home_team, away_team, gw = row['HomeTeam_std'], row['AwayTeam_std'], f"GW{row['GW']}"
        if home_team in PREMIER_LEAGUE_TEAMS:
            display_data[home_team][gw] = f"{TEAM_ABBREVIATIONS.get(away_team, '???')} (H)"
            fdr_score_data[home_team][gw] = fdr_score_func(rating_dict.get(away_team))
        if away_team in PREMIER_LEAGUE_TEAMS:
            display_data[away_team][gw] = f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)"
            fdr_score_data[away_team][gw] = fdr_score_func(rating_dict.get(home_team))

    display_df = pd.DataFrame.from_dict(display_data, orient='index').reindex(columns=gw_columns)
    fdr_score_df = pd.DataFrame.from_dict(fdr_score_data, orient='index').reindex(columns=gw_columns)
    
    display_df['Total Difficulty'] = fdr_score_df.sum(axis=1)
    display_df.sort_values(by='Total Difficulty', ascending=True, inplace=True)
    
    cols = ['Total Difficulty'] + [col for col in display_df.columns if col != 'Total Difficulty']
    return display_df[cols], fdr_score_df.reindex(display_df.index)

# --- Styling Function ---
def style_fdr_table(display_df, fdr_score_df):
    """Applies CSS styling to the FDR table."""
    def color_cells(fdr_score):
        if pd.isna(fdr_score): return f'background-color: {BLANK_FIXTURE_COLOR}'
        color = FDR_COLORS.get(fdr_score, '#FFFFFF')
        text_color = '#31333F' if fdr_score <= 3 else '#FFFFFF'
        return f'background-color: {color}; color: {text_color}'

    styler = display_df.style
    gw_cols = [col for col in display_df.columns if 'GW' in col]
    styler = styler.apply(lambda x: fdr_score_df[gw_cols].map(color_cells), axis=None, subset=gw_cols)
    styler = styler.format({'Total Difficulty': '{:.0f}'})
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
    # --- Setup shared logic ---
    rating_col = 'Hybrid Rating' if 'Hybrid Rating' in ratings_df.columns else 'Final Rating'
    rating_dict = ratings_df.set_index('Team')[rating_col].to_dict()
    pl_team_ratings = [r for t, r in rating_dict.items() if t in PREMIER_LEAGUE_TEAMS and r is not None]

    # --- NEW: Dynamic Threshold Calculation ---
    fdr_thresholds = {}
    if len(pl_team_ratings) > 0:
        min_rating = min(pl_team_ratings)
        max_rating = max(pl_team_ratings)
        rating_range = max_rating - min_rating
        interval_size = rating_range / 5
        
        # Create 4 thresholds for 5 groups
        fdr_thresholds[1] = min_rating + interval_size       # Threshold for FDR 2
        fdr_thresholds[2] = min_rating + (2 * interval_size)  # Threshold for FDR 3
        fdr_thresholds[3] = min_rating + (3 * interval_size)  # Threshold for FDR 4
        fdr_thresholds[4] = min_rating + (4 * interval_size)  # Threshold for FDR 5
    
    # Update the glossary dynamically
    with st.expander("Glossary & How It Works"):
        st.markdown("""
        - **FDR (Fixture Difficulty Rating):** Each fixture is rated on a scale of 1 to 5.
        - **How it's calculated:** The app automatically finds the highest and lowest team ratings in your file and divides that range into 5 equal 'difficulty zones'.
        - **Total Difficulty:** The sum of the FDR scores for all fixtures in the selected range. A **lower** number indicates an easier run of matches.
        """)
        if fdr_thresholds:
            st.markdown(f"""
            **Calculated Thresholds:**
            - **FDR 5 (Hardest):** Opponent Rating ≥ {fdr_thresholds[4]:.2f}
            - **FDR 4:** Opponent Rating ≥ {fdr_thresholds[3]:.2f}
            - **FDR 3:** Opponent Rating ≥ {fdr_thresholds[2]:.2f}
            - **FDR 2:** Opponent Rating ≥ {fdr_thresholds[1]:.2f}
            - **FDR 1 (Easiest):** All other ratings
            """)

    st.sidebar.header("Controls")
    num_gws_to_show = st.sidebar.number_input("Select number of gameweeks to view:", min_value=1, max_value=12, value=8, step=1)
    selected_teams = st.sidebar.multiselect("Select teams to display:", options=PREMIER_LEAGUE_TEAMS, default=PREMIER_LEAGUE_TEAMS)

    # Create the FDR score function with the dynamic thresholds
    fdr_score_func = lambda rating: get_fdr_score_from_rating(rating, fdr_thresholds)

    # --- Create and Display Data ---
    display_df, fdr_score_df = create_fdr_data(fixtures_df, num_gws_to_show, STARTING_GAMEWEEK, rating_dict, fdr_score_func)
    
    if selected_teams:
        teams_to_show = [team for team in display_df.index if team in selected_teams]
        display_df, fdr_score_df = display_df.loc[teams_to_show], fdr_score_df.loc[teams_to_show]
    else:
        display_df = pd.DataFrame()

    if not display_df.empty:
        display_df.reset_index(inplace=True); display_df.rename(columns={'index': 'Team'}, inplace=True)
        st.dataframe(
            style_fdr_table(display_df.set_index('Team'), fdr_score_df), 
            use_container_width=True, 
            height=(len(display_df) + 1) * 35
        )
    elif not selected_teams:
        st.warning("Please select at least one team from the sidebar to display the fixtures.")

else:
    st.error("Data could not be loaded. Please check your CSV files.")

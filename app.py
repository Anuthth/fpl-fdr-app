import pandas as pd
import streamlit as st
import numpy as np
from matplotlib.colors import to_hex

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
    
    pl_ratings_df = ratings_df[ratings_df['Team'].isin(PREMIER_LEAGUE_TEAMS)]
    rating_dict = pl_ratings_df.set_index('Team')[rating_col].to_dict()

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
    
    # CHANGED: "Score" is now "Total Difficulty"
    display_df['Total Difficulty'] = fdr_score_df.sum(axis=1)
    
    display_df.sort_values(by='Total Difficulty', ascending=True, inplace=True)
    cols = ['Total Difficulty'] + [col for col in display_df.columns if col != 'Total Difficulty']
    display_df = display_df[cols]
    
    return display_df, fdr_score_df.reindex(display_df.index)

# --- Styling Functions ---

def style_fdr_table(display_df, fdr_score_df):
    """Applies CSS styling to the FDR table using the 5-color system."""
    
    def color_cells(fdr_score):
        if pd.isna(fdr_score):
            return f'background-color: {BLANK_FIXTURE_COLOR}'
        
        color = FDR_COLORS.get(fdr_score, '#FFFFFF')
        text_color = '#31333F' if fdr_score <= 3 else '#FFFFFF'
        return f'background-color: {color}; color: {text_color}'

    styler = display_df.style

    gw_cols = [col for col in display_df.columns if 'GW' in col]
    subset_scores = fdr_score_df[gw_cols]
    
    styler = styler.apply(lambda x: subset_scores.map(color_cells), axis=None, subset=gw_cols)
    
    # CHANGED: Formatting the new "Total Difficulty" column
    styler = styler.format({'Total Difficulty': '{:.0f}'})

    styler = styler.set_table_styles([
        {'selector': 'th.row_heading', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])

    return styler

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("โค้ชFPL Fixture Difficulty")

# NEW: Added a collapsible glossary
with st.expander("Glossary & How It Works"):
    st.markdown("""
    - **FDR (Fixture Difficulty Rating):** Each fixture is rated on a scale of 1 to 5, where 1 is the easiest and 5 is the hardest.
    - **How it's calculated:** Team ratings are divided into five groups (quintiles). Playing a team in the easiest group of opponents gives an FDR of 1; playing a team in the hardest group gives an FDR of 5.
    - **Total Difficulty:** This is the sum of the FDR scores for all fixtures in the selected range. A **lower** number indicates an easier run of matches.
    - **(H)** denotes a Home fixture.
    - **(A)** denotes an Away fixture.
    """)

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
    
    selected_teams = st.sidebar.multiselect(
        "Select teams to display:",
        options=PREMIER_LEAGUE_TEAMS,
        default=PREMIER_LEAGUE_TEAMS
    )

    display_df, fdr_score_df = create_fdr_data(ratings_df, fixtures_df, num_gws_to_show, STARTING_GAMEWEEK)

    if selected_teams:
        teams_to_show = [team for team in display_df.index if team in selected_teams]
        display_df = display_df.loc[teams_to_show]
        fdr_score_df = fdr_score_df.loc[teams_to_show]
    else:
        display_df = pd.DataFrame()

    if not display_df.empty:
        display_df.reset_index(inplace=True)
        display_df.rename(columns={'index': 'Team'}, inplace=True)
        
        styled_table = style_fdr_table(display_df.set_index('Team'), fdr_score_df)

        table_height = (len(display_df) + 1) * 35
        st.dataframe(styled_table, use_container_width=True, height=table_height)
    elif not selected_teams:
        st.warning("Please select at least one team from the sidebar to display the fixtures.")
    else:
        st.error("Data could not be generated for the selected teams.")

else:
    st.error("Data could not be loaded. Please check your CSV files.")

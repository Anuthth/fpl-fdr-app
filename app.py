import pandas as pd
import streamlit as st
import numpy as np

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# User's custom 5-color FDR system
FDR_COLORS = {
    1: '#00ff85',
    2: '#50c369',
    3: '#D3D3D3',
    4: '#9d66a0',
    5: '#6f2a74'
}
BLANK_FIXTURE_COLOR = '#444444'

# Your defined FDR thresholds
FDR_THRESHOLDS = {
    5: 115.0,
    4: 90.0,
    3: 80.0,
    2: 70.0,
    1: 0
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

def create_fdr_data(fixtures_df, num_gws, start_gw, rating_dict):
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
            fdr_score_data[home_team][gw] = get_fdr_score_from_rating(rating_dict.get(away_team))
        if away_team in PREMIER_LEAGUE_TEAMS:
            display_data[away_team][gw] = f"{TEAM_ABBREVIATIONS.get(home_team, '???')} (A)"
            fdr_score_data[away_team][gw] = get_fdr_score_from_rating(rating_dict.get(home_team))

    display_df = pd.DataFrame.from_dict(display_data, orient='index').reindex(columns=gw_columns)
    fdr_score_df = pd.DataFrame.from_dict(fdr_score_data, orient='index').reindex(columns=gw_columns)
    
    display_df['Total Difficulty'] = fdr_score_df.sum(axis=1)
    display_df.sort_values(by='Total Difficulty', ascending=True, inplace=True)
    
    cols = ['Total Difficulty'] + [col for col in display_df.columns if col != 'Total Difficulty']
    return display_df[cols], fdr_score_df.reindex(display_df.index)

def find_fixture_runs(fixtures_df, rating_dict, start_gw):
    """Scans for runs of 3+ games with an FDR of 3 or less."""
    all_fixtures = {team: [] for team in PREMIER_LEAGUE_TEAMS}
    for gw in range(1, 39):
        gw_fixtures = fixtures_df[fixtures_df['GW'] == gw]
        for _, row in gw_fixtures.iterrows():
            home_team, away_team = row['HomeTeam_std'], row['AwayTeam_std']
            if home_team in PREMIER_LEAGUE_TEAMS:
                all_fixtures[home_team].append({"gw": gw, "opp": away_team, "loc": "H", "fdr": get_fdr_score_from_rating(rating_dict.get(away_team))})
            if away_team in PREMIER_LEAGUE_TEAMS:
                all_fixtures[away_team].append({"gw": gw, "opp": home_team, "loc": "A", "fdr": get_fdr_score_from_rating(rating_dict.get(home_team))})

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

with st.expander("Glossary & How It Works"):
    st.markdown(f"""
    - **FDR (Fixture Difficulty Rating):** Each fixture is rated 1-5 based on the opponent's 'Final Rating'.
    - **Your Custom Thresholds:** FDR 5 (Rating ≥ {FDR_THRESHOLDS[5]}), FDR 4 (≥ {FDR_THRESHOLDS[4]}), FDR 3 (≥ {FDR_THRESHOLDS[3]}), FDR 2 (≥ {FDR_THRESHOLDS[2]}), FDR 1 (all others).
    - **Total Difficulty:** The sum of the FDR scores for all fixtures in the selected range. A **lower** number indicates an easier run of matches.
    """)

ratings_df, fixtures_df = load_data()

if ratings_df is not None and fixtures_df is not None:
    st.sidebar.header("Controls")

    # --- NEW: Gameweek Range Slider ---
    start_gw, end_gw = st.sidebar.slider(
        "Select Gameweek Range:",
        min_value=1,
        max_value=38,
        value=(1, 8)  # Default range GW1 to GW8
    )
    
    selected_teams = st.sidebar.multiselect(
        "Select teams to display:",
        options=PREMIER_LEAGUE_TEAMS,
        default=PREMIER_LEAGUE_TEAMS
    )
    
    st.sidebar.header("Fixture Run Finder")
    st.sidebar.info("Find a team's upcoming periods of 3 or more easy/neutral fixtures (FDR 1-3).")
    team_to_check = st.sidebar.selectbox(
        "Select a team:",
        options=[None] + PREMIER_LEAGUE_TEAMS,
        format_func=lambda x: "Choose a team..." if x is None else x
    )

    rating_col = 'Hybrid Rating' if 'Hybrid Rating' in ratings_df.columns else 'Final Rating'
    rating_dict = ratings_df.set_index('Team')[rating_col].to_dict()

    # Calculate number of gameweeks from the slider
    num_gws_to_show = end_gw - start_gw + 1

    # Create and Display Main Table
    display_df, fdr_score_df = create_fdr_data(fixtures_df, num_gws_to_show, start_gw, rating_dict)
    
    if selected_teams:
        teams_to_show = [team for team in display_df.index if team in selected_teams]
        display_df, fdr_score_df = display_df.loc[teams_to_show], fdr_score_df.loc[teams_to_show]
    else:
        display_df = pd.DataFrame()

    if not display_df.empty:
        display_df.reset_index(inplace=True); display_df.rename(columns={'index': 'Team'}, inplace=True)
        st.dataframe(style_fdr_table(display_df.set_index('Team'), fdr_score_df), use_container_width=True, height=(len(display_df) + 1) * 35)
    elif not selected_teams:
        st.warning("Please select at least one team from the sidebar to display the fixtures.")
    
    st.markdown("---") 

    # Display Fixture Run Finder Results
    st.header("✅ Easy Fixture Runs")
    
    if team_to_check:
        # Pass the start of the selected range to the finder
        all_runs = find_fixture_runs(fixtures_df, rating_dict, start_gw)
        team_runs = all_runs.get(team_to_check)

        if team_runs:
            st.subheader(f"Found {len(team_runs)} matching run(s) for {team_to_check} (starting from GW{start_gw})")
            for i, run in enumerate(team_runs):
                start, end = run[0]['gw'], run[-1]['gw']
                st.markdown(f"**Run {i+1}: GW{start} - GW{end}**")
                run_text = ""
                for fix in run:
                    opp_abbr = TEAM_ABBREVIATIONS.get(fix['opp'], '???')
                    run_text += f"- **GW{fix['gw']}:** {opp_abbr} ({fix['loc']}) - FDR: {fix['fdr']} \n"
                st.markdown(run_text)
        else:
            st.warning(f"No upcoming runs of 3+ easy/neutral fixtures found for {team_to_check} starting from GW{start_gw}.")
    else:
        st.info("Select a team from the sidebar dropdown to check for their easy fixture runs.")

else:
    st.error("Data could not be loaded. Please check your CSV files.")

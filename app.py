import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import math

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FPL Fixture Planner")

RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# Constants for the Poisson model
AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25

# Your defined FDR thresholds
FDR_THRESHOLDS = {
    5: 120.0,
    4: 108.0,
    3: 99.0,
    2: 90.0,
    1: 0
}

# User's custom 5-color FDR system
FDR_COLORS = {
    1: '#00ff85',
    2: '#50c369',
    3: '#D3D3D3',  # Grey
    4: '#9d66a0',
    5: '#6f2a74',
    'BGW': '#e0e0e0' # Added specific color for blanks (Light Grey)
}

# --- Team Lists and Mappings ---
# Exact list from your code
PREMIER_LEAGUE_TEAMS = sorted([
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
    'Man City', 'Man Utd', 'Newcastle', 'Nottm Forest', 'Sunderland',
    'Spurs', 'West Ham', 'Wolves'
])

# Mapping CSV names to your Short Names
CSV_TO_SHORT_NAME = {
    'Arsenal': 'Arsenal', 
    'Aston Villa': 'Aston Villa', 
    'A.F.C. Bournemouth': 'Bournemouth', 
    'Brentford': 'Brentford', 
    'Brighton & Hove Albion': 'Brighton', 
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea', 
    'Crystal Palace': 'Crystal Palace', 
    'Everton': 'Everton', 
    'Fulham': 'Fulham', 
    'Leeds United': 'Leeds', 
    'Liverpool': 'Liverpool',
    'Manchester City': 'Man City', 
    'Manchester United': 'Man Utd', 
    'Newcastle United': 'Newcastle', 
    'Nottingham Forest': 'Nottm Forest', 
    'Sunderland': 'Sunderland',
    'Tottenham Hotspur': 'Spurs', 
    'West Ham United': 'West Ham', 
    'Wolverhampton Wanderers': 'Wolves'
}

# Abbreviations for display
TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Leeds': 'LEE', 'Liverpool': 'LIV',
    'Man City': 'MCI', 'Man Utd': 'MUN', 'Newcastle': 'NEW', 'Nottm Forest': 'NFO',
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL'
}

# --- Helper Functions ---

@st.cache_data
def load_data():
    try:
        fixtures = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error(f"Could not find {FIXTURES_CSV_FILE}.")
        return None, None

    try:
        ratings = pd.read_csv(RATINGS_CSV_FILE)
        if 'Team' not in ratings.columns:
            ratings.rename(columns={'team_name': 'Team'}, inplace=True)
    except FileNotFoundError:
        st.warning(f"Could not find {RATINGS_CSV_FILE}. Using default ratings.")
        ratings = pd.DataFrame({'Team': PREMIER_LEAGUE_TEAMS, 'Att': 1.0, 'Def': 1.0})
    
    return fixtures, ratings

def calculate_match_fdr(home_team, away_team, ratings_dict):
    h_att = ratings_dict.get(home_team, {}).get('Att', 1.0)
    h_def = ratings_dict.get(home_team, {}).get('Def', 1.0)
    a_att = ratings_dict.get(away_team, {}).get('Att', 1.0)
    a_def = ratings_dict.get(away_team, {}).get('Def', 1.0)

    # Your original calculation logic
    home_fdr_score = (a_att * a_def * 100) 
    away_fdr_score = (h_att * h_def * 100) * 1.1 
    
    return home_fdr_score, away_fdr_score

def get_fdr_category(score):
    for cat, threshold in FDR_THRESHOLDS.items():
        if score >= threshold:
            return cat
    return 1

def get_color_for_category(cat):
    return FDR_COLORS.get(cat, '#ffffff')

def find_fixture_runs(fixtures_df, ratings_dict, start_gw, min_length=3):
    # Original logic for finding easy runs
    runs = {}
    processed_matches = []
    
    for _, row in fixtures_df.iterrows():
        gw = row['GW']
        if gw < start_gw: continue
        
        home = CSV_TO_SHORT_NAME.get(row['Home Team'], row['Home Team'])
        away = CSV_TO_SHORT_NAME.get(row['Away Team'], row['Away Team'])
        
        if home not in PREMIER_LEAGUE_TEAMS or away not in PREMIER_LEAGUE_TEAMS: continue
        
        h_score, a_score = calculate_match_fdr(home, away, ratings_dict)
        h_cat = get_fdr_category(h_score)
        a_cat = get_fdr_category(a_score)
        
        processed_matches.append({'Team': home, 'gw': gw, 'opp': away, 'loc': 'H', 'fdr': h_cat})
        processed_matches.append({'Team': away, 'gw': gw, 'opp': home, 'loc': 'A', 'fdr': a_cat})
        
    df = pd.DataFrame(processed_matches)
    
    for team in PREMIER_LEAGUE_TEAMS:
        team_fixtures = df[df['Team'] == team].sort_values('gw')
        current_run = []
        team_runs = []
        
        gws = team_fixtures['gw'].unique()
        
        for gw in gws:
            matches = team_fixtures[team_fixtures['gw'] == gw]
            is_easy = all(m <= 3 for m in matches['fdr'])
            
            if is_easy:
                current_run.extend(matches.to_dict('records'))
            else:
                if len(set(m['gw'] for m in current_run)) >= min_length:
                    team_runs.append(current_run)
                current_run = []
                
        if len(set(m['gw'] for m in current_run)) >= min_length:
            team_runs.append(current_run)
            
        if team_runs:
            runs[team] = team_runs
            
    return runs

def process_fixtures_for_grid(fixtures_df, ratings_df):
    """
    Transforms fixtures into grid format, aggregating Double Gameweeks.
    """
    ratings_df['Team'] = ratings_df['Team'].map(lambda x: CSV_TO_SHORT_NAME.get(x, x))
    ratings_dict = ratings_df.set_index('Team').to_dict('index')

    all_matches = []
    
    for _, row in fixtures_df.iterrows():
        gw = row['GW']
        home = CSV_TO_SHORT_NAME.get(row['Home Team'], row['Home Team'])
        away = CSV_TO_SHORT_NAME.get(row['Away Team'], row['Away Team'])
        
        if home not in PREMIER_LEAGUE_TEAMS or away not in PREMIER_LEAGUE_TEAMS:
            continue

        h_score, a_score = calculate_match_fdr(home, away, ratings_dict)
        h_cat = get_fdr_category(h_score)
        a_cat = get_fdr_category(a_score)
        
        all_matches.append({
            'Team': home, 'GW': gw, 'Opponent': away, 'Loc': 'H',
            'Cat': h_cat, 'Color': get_color_for_category(h_cat)
        })
        all_matches.append({
            'Team': away, 'GW': gw, 'Opponent': home, 'Loc': 'A',
            'Cat': a_cat, 'Color': get_color_for_category(a_cat)
        })

    matches_df = pd.DataFrame(all_matches)
    
    # Build the Grid Data
    grid_rows = []
    
    for team in PREMIER_LEAGUE_TEAMS:
        row_data = {'Team': team}
        # We also store a parallel "color row" to help the JS renderer
        color_data = {}
        
        team_matches = matches_df[matches_df['Team'] == team]
        
        for gw in range(1, 39):
            gw_matches = team_matches[team_matches['GW'] == gw]
            col_name = f"GW{gw}"
            color_col_name = f"GW{gw}_color"
            
            count = len(gw_matches)
            
            if count == 0:
                # Blank Gameweek
                row_data[col_name] = "-"
                row_data[color_col_name] = FDR_COLORS['BGW']
            elif count == 1:
                # Single Gameweek
                m = gw_matches.iloc[0]
                opp = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3])
                row_data[col_name] = f"{opp}({m['Loc']})"
                row_data[color_col_name] = m['Color']
            else:
                # Double Gameweek
                labels = []
                colors = []
                for _, m in gw_matches.iterrows():
                    opp = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3])
                    labels.append(f"{opp}({m['Loc']})")
                    colors.append(m['Color'])
                
                row_data[col_name] = ", ".join(labels)
                
                # Create CSS Gradient for the background
                c1 = colors[0]
                c2 = colors[1] if len(colors) > 1 else c1
                # 50/50 split gradient
                row_data[color_col_name] = f"linear-gradient(90deg, {c1} 50%, {c2} 50%)"
        
        grid_rows.append(row_data)

    return pd.DataFrame(grid_rows)

# --- Main App ---

st.title("⚽ FPL Fixture Planner")

fixtures_df, ratings_df = load_data()

if fixtures_df is not None and ratings_df is not None:
    
    # --- Sidebar (Reverted to Original Style) ---
    st.sidebar.header("Settings")
    
    # Added Button to Clear Cache and Rerun
    if st.sidebar.button("Clear Cache & Rerun"):
        st.cache_data.clear()
        st.rerun()
    
    # Original sliders for Start and End GW
    start_gw = st.sidebar.slider("Start Gameweek", 25, 38, 25)
    end_gw = st.sidebar.slider("End Gameweek", 25, 38, 34)
    
    selected_teams = st.sidebar.multiselect("Filter Teams", PREMIER_LEAGUE_TEAMS, default=[])

    # --- Easy Run Finder (Preserved) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Easy Run Finder")
    st.sidebar.info("Find upcoming periods of 3+ easy/neutral fixtures (FDR 1-3).")
    
    teams_to_check = st.sidebar.multiselect("Select teams to find runs for:", PREMIER_LEAGUE_TEAMS, default=[], key='runs_multiselect')
    
    if teams_to_check:
        st.header("✅ Easy Fixture Runs")
        rating_dict = ratings_df.set_index('Team').to_dict('index')
        all_runs = find_fixture_runs(fixtures_df, rating_dict, start_gw)
        results_found = False
        for team in teams_to_check:
            team_runs = all_runs.get(team)
            if team_runs:
                results_found = True
                with st.expander(f"**{team}** ({len(team_runs)} matching run(s) found)"):
                    for i, run in enumerate(team_runs):
                        run_start, run_end = run[0]['gw'], run[-1]['gw']
                        st.markdown(f"**Run {i+1}: GW{run_start} - GW{run_end}**")
                        run_text = ""
                        for fix in run:
                            opp_abbr = TEAM_ABBREVIATIONS.get(fix['opp'], '???')
                            run_text += f"- **GW{fix['gw']}:** {opp_abbr} ({fix['loc']}) - FDR: {fix['fdr']} \n"
                        st.markdown(run_text)
        if not results_found:
            st.warning("No upcoming runs of 3+ easy/neutral fixtures found.")
        st.markdown("---")

    # --- Grid Logic ---
    final_df = process_fixtures_for_grid(fixtures_df, ratings_df)
    
    # Prepare Columns (Using start_gw and end_gw sliders)
    visible_cols = ['Team'] + [f"GW{i}" for i in range(start_gw, end_gw + 1)]
    # We need the hidden color columns available in the dataframe passed to AgGrid
    color_cols = [f"GW{i}_color" for i in range(start_gw, end_gw + 1)]
    all_cols = visible_cols + color_cols
    
    display_df = final_df[all_cols].copy()
    
    if selected_teams:
        display_df = display_df[display_df['Team'].isin(selected_teams)]
        
    # Default Sort by Team (Alphabetical)
    display_df = display_df.sort_values('Team')
    
    # --- AgGrid Construction ---
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("Team", pinned="left", width=120, cellStyle={'fontWeight': 'bold'})
    
    # Hide the technical color columns
    for c in color_cols:
        gb.configure_column(c, hide=True)
        
    # Apply Styling to GW columns
    for i in range(start_gw, end_gw + 1):
        col_name = f"GW{i}"
        
        # This JS function reads the value from the hidden 'GWx_color' column
        # and applies it as the background for the visible 'GWx' column.
        js_style = JsCode("""
        function(params) {
            var colorCol = params.colDef.field + "_color";
            var colorVal = params.data[colorCol];
            return {
                'background': colorVal,
                'color': 'black',
                'border-right': '1px solid #ddd',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'font-size': '12px',
                'white-space': 'nowrap' 
            };
        }
        """)
        
        gb.configure_column(col_name, width=100, cellStyle=js_style)

    gridOptions = gb.build()
    
    AgGrid(
        display_df, 
        gridOptions=gridOptions, 
        allow_unsafe_jscode=True, 
        height=600,
        theme='streamlit'
    )
    
    # Legend
    st.markdown("### FDR Key")
    cols = st.columns(6)
    keys = [(1, "Easy"), (2, "Good"), (3, "Neutral"), (4, "Hard"), (5, "Very Hard"), ('BGW', "Blank")]
    
    for i, (k, label) in enumerate(keys):
        c = FDR_COLORS[k]
        cols[i].markdown(
            f"<div style='background:{c};padding:8px;border-radius:4px;text-align:center;color:black;'>{label}</div>", 
            unsafe_allow_html=True
        )

else:
    st.info("Please ensure both CSV files are uploaded.")

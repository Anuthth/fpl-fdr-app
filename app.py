import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FPL Fixture Planner")

RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# Constants for the Poisson model
AVG_LEAGUE_HOME_GOALS = 1.55
AVG_LEAGUE_AWAY_GOALS = 1.25

# FDR Thresholds
FDR_THRESHOLDS = {
    5: 120.0,
    4: 108.0,
    3: 99.0,
    2: 90.0,
    1: 0
}

# FDR Colors
FDR_COLORS = {
    1: '#375523',  # Dark Green (Easiest)
    2: '#01fc7a',  # Bright Green
    3: '#e7e7e7',  # Grey (Neutral)
    4: '#ff1751',  # Pink/Red (Hard)
    5: '#80072d',  # Dark Red (Hardest)
    'BGW': '#555555' # Dark Grey for Blanks
}

# --- Team Mappings ---
# Maps CSV full names to internal short names
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
    'Wolverhampton Wanderers': 'Wolves',
    'Sheffield United': 'Sheff Utd',
    'Luton Town': 'Luton',
    'Leicester City': 'Leicester',
    'Ipswich Town': 'Ipswich',
    'Southampton': 'Southampton'
}

# List of teams for dropdowns (using short names)
PREMIER_LEAGUE_TEAMS = sorted(list(set(CSV_TO_SHORT_NAME.values())))

# Abbreviations for display in grid (Short Name -> Abbr)
TEAM_ABBREVIATIONS = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Burnley': 'BUR', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Leeds': 'LEE', 'Liverpool': 'LIV',
    'Man City': 'MCI', 'Man Utd': 'MUN', 'Newcastle': 'NEW', 'Nottm Forest': 'NFO',
    'Sunderland': 'SUN', 'Spurs': 'TOT', 'West Ham': 'WHU', 'Wolves': 'WOL',
    'Sheff Utd': 'SHU', 'Luton': 'LUT', 'Leicester': 'LEI', 'Ipswich': 'IPS', 'Southampton': 'SOU'
}

# --- Helper Functions ---

def load_data():
    """Loads fixtures and ratings."""
    try:
        fixtures = pd.read_csv(FIXTURES_CSV_FILE)
    except FileNotFoundError:
        st.error(f"Could not find {FIXTURES_CSV_FILE}. Please upload it.")
        return None, None

    try:
        ratings = pd.read_csv(RATINGS_CSV_FILE)
        # Normalize columns if needed
        if 'Team' not in ratings.columns:
            ratings.rename(columns={'team_name': 'Team'}, inplace=True)
    except FileNotFoundError:
        st.warning(f"Could not find {RATINGS_CSV_FILE}. Using default neutral ratings.")
        # Create dummy ratings
        ratings = pd.DataFrame({
            'Team': PREMIER_LEAGUE_TEAMS,
            'Att': 1.0,
            'Def': 1.0
        })
    
    return fixtures, ratings

def calculate_match_fdr(home_team, away_team, ratings_dict):
    """Calculates FDR for Home and Away perspectives."""
    h_att = ratings_dict.get(home_team, {}).get('Att', 1.0)
    h_def = ratings_dict.get(home_team, {}).get('Def', 1.0)
    a_att = ratings_dict.get(away_team, {}).get('Att', 1.0)
    a_def = ratings_dict.get(away_team, {}).get('Def', 1.0)

    # Simplified Poisson-like strength calculation
    # Home Difficulty = Away Attack strength * Home Defense weakness (inverted)
    # Actually, standard FDR is about how hard the OPPONENT is.
    
    # FDR for Home Team (facing Away Team): Based on Away Team's Strength
    # Away Team Strength = A_Att * A_Def * (Base Difficulty)
    # Let's use a simpler heuristic based on the user's thresholds
    # High score = Hard match.
    
    # Score for Home Team = Away Team Strength factor
    home_fdr_score = (a_att * a_def * 100) # Arbitrary scaling to fit 0-150 range
    
    # Score for Away Team = Home Team Strength factor * Home Advantage
    away_fdr_score = (h_att * h_def * 100) * 1.1 # 10% boost for home advantage difficulty
    
    return home_fdr_score, away_fdr_score

def get_fdr_category(score):
    """Maps a numeric score to 1-5 category."""
    for cat, threshold in FDR_THRESHOLDS.items():
        if score >= threshold:
            return cat
    return 1

def get_color_for_category(cat):
    return FDR_COLORS.get(cat, '#ffffff')

def process_fixtures(fixtures_df, ratings_df):
    """
    Transforms raw fixtures into a grid-ready structure handling DGWs and BGWs.
    """
    # 1. Create Lookup for Ratings
    # Standardize team names in ratings if needed
    ratings_df['Team'] = ratings_df['Team'].map(lambda x: CSV_TO_SHORT_NAME.get(x, x))
    ratings_dict = ratings_df.set_index('Team').to_dict('index')

    # 2. Parse Fixtures into a list of Match Objects
    all_matches = []
    
    for _, row in fixtures_df.iterrows():
        gw = row['GW']
        home_full = row['Home Team']
        away_full = row['Away Team']
        
        home_short = CSV_TO_SHORT_NAME.get(home_full, home_full)
        away_short = CSV_TO_SHORT_NAME.get(away_full, away_full)
        
        # skip if mapping fails to find a valid team (optional safety)
        if home_short not in PREMIER_LEAGUE_TEAMS or away_short not in PREMIER_LEAGUE_TEAMS:
            continue

        h_score, a_score = calculate_match_fdr(home_short, away_short, ratings_dict)
        h_cat = get_fdr_category(h_score)
        a_cat = get_fdr_category(a_score)
        
        # Record match for Home Team
        all_matches.append({
            'Team': home_short,
            'GW': gw,
            'Opponent': away_short,
            'Loc': 'H',
            'FDR': h_score,
            'Cat': h_cat,
            'Color': get_color_for_category(h_cat)
        })
        
        # Record match for Away Team
        all_matches.append({
            'Team': away_short,
            'GW': gw,
            'Opponent': home_short,
            'Loc': 'A',
            'FDR': a_score,
            'Cat': a_cat,
            'Color': get_color_for_category(a_cat)
        })

    # 3. Aggregate by Team and GW
    matches_df = pd.DataFrame(all_matches)
    
    # We need a grid of all Teams x all GWs (1-38) to catch BGWs
    teams = PREMIER_LEAGUE_TEAMS
    gws = range(1, 39)
    
    grid_data = []
    
    for team in teams:
        row_dict = {'Team': team}
        color_row_dict = {'Team': team}
        
        team_matches = matches_df[matches_df['Team'] == team]
        
        for gw in gws:
            gw_matches = team_matches[team_matches['GW'] == gw]
            count = len(gw_matches)
            
            col_name = f"GW{gw}"
            color_col_name = f"GW{gw}_color"
            
            if count == 0:
                # BGW
                row_dict[col_name] = "-"
                color_row_dict[color_col_name] = FDR_COLORS['BGW']
            elif count == 1:
                # Normal GW
                m = gw_matches.iloc[0]
                opp_abbr = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3].upper())
                row_dict[col_name] = f"{opp_abbr}({m['Loc']})"
                color_row_dict[color_col_name] = m['Color']
            else:
                # DGW (or TGW)
                texts = []
                colors = []
                for _, m in gw_matches.iterrows():
                    opp_abbr = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3].upper())
                    texts.append(f"{opp_abbr}({m['Loc']})")
                    colors.append(m['Color'])
                
                # Join text
                row_dict[col_name] = ", ".join(texts)
                
                # Create CSS Gradient for background
                if len(colors) == 2:
                    grad = f"linear-gradient(90deg, {colors[0]} 50%, {colors[1]} 50%)"
                elif len(colors) >= 3:
                    # Just in case of TGW, split 3 ways
                    grad = f"linear-gradient(90deg, {colors[0]} 33%, {colors[1]} 33% 66%, {colors[2]} 66%)"
                else:
                    grad = colors[0]
                    
                color_row_dict[color_col_name] = grad

        # Merge color data into the main row dict for the dataframe
        # We put colors in separate columns to hide them later
        full_row = {**row_dict, **color_row_dict}
        grid_data.append(full_row)

    final_df = pd.DataFrame(grid_data)
    return final_df

# --- Main App Logic ---

st.title("⚽ FPL Fixture Planner (DGW & BGW Support)")

fixtures_df, ratings_df = load_data()

if fixtures_df is not None and ratings_df is not None:
    
    # --- Sidebar Controls ---
    st.sidebar.header("Settings")
    
    # GW Range Slider
    min_gw = int(fixtures_df['GW'].min())
    max_gw = int(fixtures_df['GW'].max())
    gw_range = st.sidebar.slider("Gameweek Range", min_gw, 38, (min_gw, min(min_gw + 10, 38)))
    
    # Team Filter
    selected_teams = st.sidebar.multiselect("Filter Teams", PREMIER_LEAGUE_TEAMS, default=[])
    
    # Sort Logic
    sort_options = ["Fixture Difficulty (Easy to Hard)", "Alphabetical"]
    sort_choice = st.sidebar.selectbox("Sort By", sort_options)

    # --- Processing ---
    final_df = process_fixtures(fixtures_df, ratings_df)
    
    # Filter Columns (GW Range)
    cols_to_keep = ['Team']
    cols_to_hide = [] # We want to keep color columns in the DF but hide them in grid
    
    for gw in range(gw_range[0], gw_range[1] + 1):
        cols_to_keep.append(f"GW{gw}")
        cols_to_keep.append(f"GW{gw}_color")
        cols_to_hide.append(f"GW{gw}_color")
        
    display_df = final_df[cols_to_keep].copy()
    
    # Filter Rows (Teams)
    if selected_teams:
        display_df = display_df[display_df['Team'].isin(selected_teams)]
        
    # Sorting (Simplified - Logic for sorting by difficulty is complex with DGWs, 
    # so we'll stick to simple alphabetical or just preserve index for now unless easy mode requested)
    if sort_choice == "Alphabetical":
        display_df = display_df.sort_values('Team')
    
    # --- AgGrid Setup ---
    gb = GridOptionsBuilder.from_dataframe(display_df)
    
    # Pin Team Column
    gb.configure_column("Team", pinned="left", width=120, cellStyle={'fontWeight': 'bold'})
    
    # Hide Color Columns
    for col in cols_to_hide:
        gb.configure_column(col, hide=True)
        
    # Configure GW Columns with JS Injection for Styling
    # We iterate over the visible GW columns
    visible_gw_cols = [c for c in cols_to_keep if "color" not in c and c != "Team"]
    
    for col in visible_gw_cols:
        # The JS function looks for the column name + "_color" in the data row
        # and applies it as the background.
        js_style = JsCode("""
        function(params) {
            var colorCol = params.colDef.field + "_color";
            var colorVal = params.data[colorCol];
            if (colorVal) {
                return {
                    'background': colorVal, 
                    'color': 'black', 
                    'border-right': '1px solid #ddd',
                    'font-size': '12px',
                    'text-align': 'center'
                };
            }
            return null;
        }
        """)
        
        gb.configure_column(col, 
                            cellStyle=js_style, 
                            width=100,
                            suppressMenu=True,
                            sortable=False)

    gridOptions = gb.build()
    
    # Render Grid
    st.markdown("### Fixture Grid")
    st.markdown("Double Gameweeks are split-colored. Blank Gameweeks are grey.")
    
    AgGrid(
        display_df,
        gridOptions=gridOptions,
        allow_unsafe_jscode=True,
        height=600,
        theme="streamlit", # or 'balham'
        fit_columns_on_grid_load=False
    )
    
    # --- Legend ---
    st.markdown("#### FDR Key")
    cols = st.columns(6)
    keys = [(1, "Easy"), (2, "Good"), (3, "Neutral"), (4, "Hard"), (5, "Very Hard"), ('BGW', "Blank")]
    
    for i, (cat, label) in enumerate(keys):
        color = FDR_COLORS[cat]
        cols[i].markdown(
            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: black; border: 1px solid #ccc;'>{label}</div>", 
            unsafe_allow_html=True
        )

else:
    st.info("Please ensure both 'Fixtures202526.csv' and 'final_team_ratings_with_components_new.csv' are in the directory.")

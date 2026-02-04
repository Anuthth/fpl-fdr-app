import pandas as pd
import streamlit as st
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FPL Fixture Planner")

RATINGS_CSV_FILE = "final_team_ratings_with_components_new.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"

# FDR Thresholds (From your original settings)
FDR_THRESHOLDS = {
    5: 120.0,
    4: 108.0,
    3: 99.0,
    2: 90.0,
    1: 0
}

# FDR Colors (Reverted to your Green/Purple scheme)
FDR_COLORS = {
    1: '#00ff85',  # Bright Green
    2: '#50c369',  # Medium Green
    3: '#D3D3D3',  # Grey
    4: '#9d66a0',  # Purple
    5: '#6f2a74',  # Dark Purple
    'BGW': '#e0e0e0' # Light Grey for Blanks
}

# --- Team Mappings ---
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

PREMIER_LEAGUE_TEAMS = sorted(list(set(CSV_TO_SHORT_NAME.values())))

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
        if 'Team' not in ratings.columns:
            ratings.rename(columns={'team_name': 'Team'}, inplace=True)
    except FileNotFoundError:
        st.warning(f"Could not find {RATINGS_CSV_FILE}. Using default neutral ratings.")
        ratings = pd.DataFrame({'Team': PREMIER_LEAGUE_TEAMS, 'Att': 1.0, 'Def': 1.0})
    
    return fixtures, ratings

def calculate_match_fdr(home_team, away_team, ratings_dict):
    """Calculates FDR score."""
    h_att = ratings_dict.get(home_team, {}).get('Att', 1.0)
    h_def = ratings_dict.get(home_team, {}).get('Def', 1.0)
    a_att = ratings_dict.get(away_team, {}).get('Att', 1.0)
    a_def = ratings_dict.get(away_team, {}).get('Def', 1.0)

    # Simplified Strength Calculation
    home_fdr_score = (a_att * a_def * 100) 
    away_fdr_score = (h_att * h_def * 100) * 1.1 
    
    return home_fdr_score, away_fdr_score

def get_fdr_category(score):
    """Maps score to 1-5 category."""
    for cat, threshold in FDR_THRESHOLDS.items():
        if score >= threshold:
            return cat
    return 1

def get_color_for_category(cat):
    return FDR_COLORS.get(cat, '#ffffff')

def process_fixtures(fixtures_df, ratings_df):
    """
    Aggregates fixtures to handle DGWs and BGWs correctly.
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
        
        # Add Home Record
        all_matches.append({
            'Team': home, 'GW': gw, 'Opponent': away, 'Loc': 'H',
            'Cat': h_cat, 'Color': get_color_for_category(h_cat)
        })
        # Add Away Record
        all_matches.append({
            'Team': away, 'GW': gw, 'Opponent': home, 'Loc': 'A',
            'Cat': a_cat, 'Color': get_color_for_category(a_cat)
        })

    matches_df = pd.DataFrame(all_matches)
    
    # Create the display grid
    teams = PREMIER_LEAGUE_TEAMS
    gws = range(1, 39)
    grid_data = []
    
    for team in teams:
        row = {'Team': team}
        # Hidden dict for colors
        colors = {'Team': team}
        
        team_matches = matches_df[matches_df['Team'] == team]
        
        for gw in gws:
            gw_matches = team_matches[team_matches['GW'] == gw]
            count = len(gw_matches)
            col_name = f"GW{gw}"
            color_name = f"GW{gw}_color"
            
            if count == 0:
                row[col_name] = "-"
                colors[color_name] = FDR_COLORS['BGW']
            elif count == 1:
                m = gw_matches.iloc[0]
                opp = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3])
                row[col_name] = f"{opp}({m['Loc']})"
                colors[color_name] = m['Color']
            else:
                # DGW Logic: Join text and split colors
                texts = []
                match_colors = []
                for _, m in gw_matches.iterrows():
                    opp = TEAM_ABBREVIATIONS.get(m['Opponent'], m['Opponent'][:3])
                    texts.append(f"{opp}({m['Loc']})")
                    match_colors.append(m['Color'])
                
                row[col_name] = ", ".join(texts)
                
                # Create gradient for DGW
                c1 = match_colors[0]
                c2 = match_colors[1] if len(match_colors) > 1 else c1
                colors[color_name] = f"linear-gradient(90deg, {c1} 50%, {c2} 50%)"

        grid_data.append({**row, **colors})

    return pd.DataFrame(grid_data)

# --- Main App ---

st.title("⚽ FPL Fixture Planner")

fixtures_df, ratings_df = load_data()

if fixtures_df is not None and ratings_df is not None:
    
    st.sidebar.header("Settings")
    
    min_gw = int(fixtures_df['GW'].min())
    gw_range = st.sidebar.slider("Gameweek Range", min_gw, 38, (min_gw, min(min_gw + 10, 38)))
    selected_teams = st.sidebar.multiselect("Filter Teams", PREMIER_LEAGUE_TEAMS)
    sort_choice = st.sidebar.selectbox("Sort By", ["Alphabetical", "Fixture Difficulty"])

    final_df = process_fixtures(fixtures_df, ratings_df)
    
    # Filter Columns
    cols = ['Team'] + [f"GW{i}" for i in range(gw_range[0], gw_range[1] + 1)]
    # Keep color cols for visible GWs only
    color_cols = [f"GW{i}_color" for i in range(gw_range[0], gw_range[1] + 1)]
    
    display_df = final_df[cols + color_cols].copy()
    
    if selected_teams:
        display_df = display_df[display_df['Team'].isin(selected_teams)]
        
    if sort_choice == "Alphabetical":
        display_df = display_df.sort_values('Team')

    # --- AgGrid ---
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("Team", pinned="left", width=120)
    
    # Hide the technical color columns
    for c in color_cols:
        gb.configure_column(c, hide=True)
    
    # Apply styling to GW columns
    for i in range(gw_range[0], gw_range[1] + 1):
        col = f"GW{i}"
        gb.configure_column(col, width=100, cellStyle=JsCode("""
            function(params) {
                var colorCol = params.colDef.field + "_color";
                var color = params.data[colorCol];
                return {
                    'background': color,
                    'color': 'black',
                    'border-right': '1px solid #ddd',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center',
                    'font-size': '12px'
                };
            }
        """))

    AgGrid(display_df, gridOptions=gb.build(), allow_unsafe_jscode=True, height=600, theme='streamlit')

    # Legend
    st.markdown("### FDR Key")
    legend_cols = st.columns(6)
    for i, (k, label) in enumerate([(1, "Easy"), (2, "Good"), (3, "Neutral"), (4, "Hard"), (5, "Very Hard"), ('BGW', "Blank")]):
        legend_cols[i].markdown(f"<div style='background:{FDR_COLORS[k]};padding:5px;text-align:center;border-radius:4px;'>{label}</div>", unsafe_allow_html=True)

else:
    st.info("Upload 'Fixtures202526.csv' and 'final_team_ratings_with_components_new.csv'.")

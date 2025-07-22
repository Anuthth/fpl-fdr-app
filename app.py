import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

# --- Configuration ---
RATINGS_CSV_FILE = "final_team_ratings_with_components.csv"
FIXTURES_CSV_FILE = "Fixtures202526.csv"
OUTPUT_FDR_IMAGE = "FDR_Schedule_Range.png"

# --- HIGHLIGHT: Select your Gameweek range here ---
START_GW = 4  # The first gameweek you want to see
END_GW = 8    # The last gameweek you want to see

# --- Chart Text ---
CHART_TITLE = f"Premier League Fixture Difficulty (GW{START_GW}-GW{END_GW})"
CREATOR_TEXT = "create by CoachFPL"

# --- FDR Colors (from easiest to hardest) ---
FDR_COLORS = {
    1: '#2ECC71',  # Green
    2: '#90EE90',  # Light Green
    3: '#D3D3D3',  # Light Gray (Neutral)
    4: '#F08080',  # LightCoral
    5: '#E74C3C'   # Red (Hardest)
}

# --- Data Correction & Naming ---
TEAM_NAME_MAP = {
    "A.F.C. Bournemouth": "AFC Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottm Forest",
    "Tottenham Hotspur": "Spurs",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Ipswich Town": "Ipswich",
    "Sheffield United": "Sheff Utd"
}

PREMIER_LEAGUE_TEAMS = [
    'Arsenal', 'Aston Villa', 'AFC Bournemouth', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool',
    'Manchester City', 'Manchester United', 'Newcastle', 'Nottingham Forest',
    'Sunderland', 'Tottenham Hotspur', 'West Ham United', 'Wolverhampton Wanderers'
]

TEAM_DISPLAY_NAMES = {
    "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
    "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nottm Forest",
    "Tottenham Hotspur": "Spurs", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Leeds United": "Leeds",
}

# --- Function Definitions ---

def load_and_prepare_data(ratings_file, fixtures_file):
    """Loads and prepares both the ratings and fixtures data."""
    print("Loading data...")
    try:
        ratings_df = pd.read_csv(ratings_file)
        fixtures_df = pd.read_csv(fixtures_file)
    except FileNotFoundError as e:
        print(f"Error: File not found. Details: {e}"); return None, None
    
    fixtures_df.dropna(how='all', inplace=True)
    fixtures_df['GW'] = fixtures_df['GW'].astype(int)
    fixtures_df['HomeTeam_std'] = fixtures_df['Home Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Home Team'])
    fixtures_df['AwayTeam_std'] = fixtures_df['Away Team'].map(TEAM_NAME_MAP).fillna(fixtures_df['Away Team'])
    print("Data loaded and prepared successfully.")
    return ratings_df, fixtures_df

def create_fdr_schedule(ratings_df, fixtures_df, start_gw, end_gw, pl_team_list):
    """Creates the master FDR schedule for all teams and sorts it by difficulty for the specified GW range."""
    print(f"Calculating FDR for all teams from GW{start_gw} to GW{end_gw}...")
    
    pl_ratings = ratings_df[ratings_df['Team'].isin(pl_team_list)].copy()
    
    if len(pl_ratings) < len(pl_team_list):
        missing = [team for team in pl_team_list if team not in pl_ratings['Team'].values]
        print(f"Warning: Could not find ratings for all specified PL teams. Missing: {missing}")

    rating_col_to_use = 'Hybrid Rating' if 'Hybrid Rating' in pl_ratings.columns else 'Final Rating'
    print(f"Using '{rating_col_to_use}' to determine fixture difficulty.")
    rating_dict = pl_ratings.set_index('Team')[rating_col_to_use].to_dict()

    rating_values = sorted(rating_dict.values())
    quintiles = np.percentile(rating_values, [0, 20, 40, 60, 80, 100])
    
    def get_fdr_score(team_rating):
        if team_rating is None or pd.isna(team_rating): return 3
        if team_rating <= quintiles[1]: return 1
        if team_rating <= quintiles[2]: return 2
        if team_rating <= quintiles[3]: return 3
        if team_rating <= quintiles[4]: return 4
        return 5

    fdr_schedule = {team: {} for team in pl_team_list}

    for _, row in fixtures_df.iterrows():
        gw, home_team, away_team = row['GW'], row['HomeTeam_std'], row['AwayTeam_std']
        
        if home_team in pl_team_list and away_team in pl_team_list:
            away_rating, home_rating = rating_dict.get(away_team), rating_dict.get(home_team)
            if home_rating is None or away_rating is None: continue
            
            fdr_home = get_fdr_score(away_rating)
            fdr_away = get_fdr_score(home_rating)

            fdr_schedule[home_team][f'GW{gw}'] = (away_team, 'H', fdr_home)
            fdr_schedule[away_team][f'GW{gw}'] = (home_team, 'A', fdr_away)
    
    fdr_df = pd.DataFrame.from_dict(fdr_schedule, orient='index').reindex(pl_team_list)
    
    # HIGHLIGHT: Calculate difficulty score only for the selected GW range
    gw_columns_in_range = [f'GW{i}' for i in range(start_gw, end_gw + 1) if f'GW{i}' in fdr_df.columns]
    fdr_df['DifficultyScore'] = fdr_df.apply(lambda row: sum(row[gw][2] for gw in gw_columns_in_range if pd.notna(row[gw])), axis=1)
    fdr_df.sort_values(by='DifficultyScore', ascending=True, inplace=True)
    
    fdr_df = fdr_df[fdr_df['DifficultyScore'] > 0]
    
    return fdr_df

def plot_fdr_grid(fdr_df, title, creator_text, output_filename, start_gw, end_gw, team_display_names):
    """Creates and saves the final FDR grid visualization for the specified GW range."""
    print(f"Generating FDR grid visualization for GW{start_gw}-GW{end_gw}...")
    
    # HIGHLIGHT: Filter columns to plot based on the start and end GW
    columns_to_plot = [f'GW{i}' for i in range(start_gw, end_gw + 1) if f'GW{i}' in fdr_df.columns]
    fdr_df_subset = fdr_df[columns_to_plot]

    num_teams, num_gameweeks_to_plot = fdr_df_subset.shape

    BG_COLOR, TEXT_COLOR_LIGHT, TEXT_COLOR_DARK = '#1E1E1E', '#FFFFFF', '#000000'
    FONT_FAMILY = 'DejaVu Sans'
    
    fig_width = max(10, num_gameweeks_to_plot * 0.8) 
    fig_height = max(8, num_teams * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    fig.subplots_adjust(left=0.25) 

    for r, team_name in enumerate(fdr_df_subset.index):
        display_name = team_display_names.get(team_name, team_name)
        ax.text(-0.1, r, display_name, va='center', ha='right', fontsize=9, color=TEXT_COLOR_LIGHT, fontweight='bold')
        
        for c, gw_name in enumerate(fdr_df_subset.columns):
            cell_data = fdr_df_subset.loc[team_name, gw_name]
            if pd.isna(cell_data): continue
            
            opponent, location, fdr_score = cell_data
            
            cell_color = FDR_COLORS.get(fdr_score, '#FFFFFF')
            ax.add_patch(Rectangle((c, r - 0.4), 1, 0.8, color=cell_color, ec=BG_COLOR, lw=2))
            
            text_color = TEXT_COLOR_DARK if fdr_score <= 3 else TEXT_COLOR_LIGHT
            opponent_display_name = team_display_names.get(opponent, opponent) 
            cell_text = f"{opponent_display_name}\n({location})"
            ax.text(c + 0.5, r, cell_text, ha='center', va='center', fontsize=7, color=text_color, fontweight='bold', linespacing=0.9)

    ax.set_xlim(-0.2, num_gameweeks_to_plot)
    ax.set_ylim(-0.5, num_teams - 0.5)
    ax.set_xticks(np.arange(num_gameweeks_to_plot) + 0.5)
    ax.set_xticklabels([f"GW{int(col.replace('GW', ''))}" for col in fdr_df_subset.columns], fontsize=8, color=TEXT_COLOR_LIGHT)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0); ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()

    for spine in ax.spines.values(): spine.set_visible(False)

    fig.text(0.5, 0.95, title, ha='center', va='center', fontsize=18, color=TEXT_COLOR_LIGHT, fontweight='bold')
    fig.text(0.99, 0.01, creator_text, ha='right', va='bottom', fontsize=8, color='#AAAAAA', style='italic')

    try:
        plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"\nFDR visualization saved as {output_filename}")
    except Exception as e:
        print(f"\nError saving FDR visualization: {e}")
    plt.close(fig)

# --- Main script ---
if __name__ == "__main__":
    ratings_df, fixtures_df = load_and_prepare_data(RATINGS_CSV_FILE, FIXTURES_CSV_FILE)
    if ratings_df is None or fixtures_df is None: exit()
    
    fdr_schedule = create_fdr_schedule(ratings_df, fixtures_df, START_GW, END_GW, PREMIER_LEAGUE_TEAMS)
    
    if fdr_schedule.empty: 
        print("\nCould not generate the FDR schedule.")
        exit()
        
    if len(fdr_schedule) < 20:
        print(f"\nWarning: FDR schedule was generated for only {len(fdr_schedule)} of 20 teams.")

    plot_fdr_grid(fdr_schedule, 
                  title=CHART_TITLE, 
                  creator_text=CREATOR_TEXT, 
                  output_filename=OUTPUT_FDR_IMAGE,
                  start_gw=START_GW,
                  end_gw=END_GW,
                  team_display_names=TEAM_DISPLAY_NAMES)

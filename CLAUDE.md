# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (default port 8501)
streamlit run app.py

# Run without CORS / XSRF protection (used in devcontainer / Codespaces)
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

There are no tests or linters configured. The entire application is a single file (`app.py`).

## Architecture

The app is a **single-file Streamlit application** (~3 500 lines). All logic — data fetching, processing, and UI rendering — lives in `app.py`. There is no package structure, no modules, and no build step.

### Data sources (two tiers)

**Live FPL API** (toggled via "📡 Live FPL Data" sidebar switch):
- `bootstrap-static/` — teams, players, element types, GW events
- `fixtures/` — all season fixtures
- `event/{gw}/live/` — live GW points, BPS, bonus, stats
- `entry/{id}/`, `entry/{id}/event/{gw}/picks/`, `entry/{id}/history/` — My FPL team
- `leagues-classic/{id}/standings/` — mini-league

**CSV files** (must be placed alongside `app.py`):
| File | Content |
|------|---------|
| `final_team_ratings_with_components_new.csv` | Custom Off Score / Def Score / Final Rating per team (drives FDR) |
| `Fixtures202526.csv` | Season schedule — used as **source of truth** for DGW/BGW (live API may lag) |
| `projections.csv` | Solio Analytics: columns `Pos,ID,Name,BV,SV,Team,{GW}_xMins,{GW}_Pts` |
| `EO%.csv` | Solio Analytics: columns `Pos,ID,Name,BV,SV,Team,{GW}_xMins,{GW}_eo` |
| `pl_teams_stats_2025_2026.csv` | Season team xG stats (fallback for Stats tab) |
| `pl_players_stats_2025_2026.csv` | Season player stats (fallback for Stats tab) |
| `Solio_Logo_Neg_RGB.png` | Branding logo embedded as base64 in HTML exports |

**External APIs** (fetched at runtime, cached):
- Fotmob (`fotmob.com/api/leagues`, `fotmob.com/api/leagueseasondeepstats`) — richer team/player stats
- FPL ID Map (`github.com/ChrisMusson/FPL-ID-Map`) — maps FPL `code` → FBRef / Understat / Transfermarkt IDs
- Premier League CDN (`resources.premierleague.com`) — player photos, club badges

### Core data pipeline

1. `load_csv_data()` → loads ratings + fixtures CSVs, normalises team names via `TEAM_NAME_MAP`
2. `create_all_data(fixtures, start_gw, end_gw, ratings)` → builds `master_df` — a DataFrame indexed by team name where each `GWN` cell is a dict: `{display, fdr, xG, CS, count, xG_parts, CS_parts, fdr_parts}`. DGW cells have `count >= 2`; BGW cells are `NaN`.
3. `master_df` is built twice: once for the full remaining season (`master_df_full`, used for captain lookups), once for the selected GW range (`master_df`, used for heatmaps).

### FDR model

Team strength ratings (`Off Score`, `Def Score`, `Final Rating`) come from the CSV. The model:
- Normalises off/def relative to league average
- Applies a `HOME_ADVANTAGE = 1.10` multiplier
- Uses Dixon-Coles-style xG: `home_xg = home_att × (1/away_def) × 1.36 × 1.10`
- CS probability: `P(Poisson(xG)=0) = exp(-xG_against)`
- Maps `Final Rating` to FDR 1–5 via `FDR_THRESHOLDS`
- DGW cells use `color_val = avg_fdr - 6` (always sorts above single-game cells 1–5)
- BGW cells use sort value `10` (always sorts to bottom)

### Team name normalisation

`TEAM_NAME_MAP` (defined at the top of `app.py`) is the single source of truth for mapping any variant of a team name (API name, short code, common abbreviation) to the canonical internal name (e.g. `"Manchester City"` → `"Man City"`). Always run team names through this map before lookups.

### Navigation model

The sidebar radio `nav_cat` selects a top-level section. Each section conditionally renders its own set of `st.tabs()`:
- **🔴 Live GW** — no sub-tabs; fixture cards with goals/BPS/defcon/bonus, expandable player detail tables
- **📊 Planning** — FDR / xG / xCS / Team Ratings heatmap tables (all use `_heatmap_table()`)
- **🎯 Captain & Picks** — Captain Picks, Captain Matrix, Cheatsheet, Differentials (all use Solio CSV data)
- **🏟️ Stats** — Team Stats and Player Stats (Fotmob API primary, FPL API fallback, CSV fallback)
- **👕 My FPL** — Live Radar (squad on pitch with EV/FDR), Mini-League standings

### HTML rendering pattern

All tables and cards are hand-crafted HTML strings rendered via `components.html()` (for interactive elements like sortable columns) or `st.markdown(..., unsafe_allow_html=True)` (for static cards). The colour scheme is consistently dark (`#0d1117` background) with club colours applied via `CLUB_COLORS` / `club_style()`.

`_heatmap_table()` is the shared renderer for all Planning tab grids. It injects a vanilla JS `sortHT()` function that sorts rows client-side by clicking column headers.

### Solio data integration

Projections (`projections.csv`) and expected ownership (`EO%.csv`) are keyed by **Solio player ID** (column `ID`), which matches FPL element IDs directly. Photo lookups use `id_to_code[solio_id]` first, falling back to fuzzy name matching via `_fuzzy_code()` / `add_fpl_positions()`.

EO values in the Solio files are stored as **raw fractions** (e.g. `0.30` = 30%), but some code paths note "EO already in %" — this refers to the fact that values like `1.74` mean 1.74% (i.e. the files use percent-as-decimal, not 0–1 fractions). Multiply by 100 for display.

### Caching strategy

- `@st.cache_data(ttl=300)` — FPL bootstrap, fixtures, live GW data (5 min)
- `@st.cache_data(ttl=60)` — live GW stats (1 min, refreshes during active GW)
- `@st.cache_data(ttl=3600)` — player photo verification, FPL ID map (1 hr)
- `@st.cache_data` (no TTL) — CSV loads, `create_all_data` (indefinite until cache cleared)
- The "🗑️ Clear Cache" button calls `st.cache_data.clear()` + `st.cache_resource.clear()` and clears `st.session_state`

### Key constants (top of file)

- `FDR_THRESHOLDS` — maps Final Rating ranges to FDR 1–5
- `BGW_PENALTY_FDR = 3.0` — BGW counts as neutral difficulty in totals
- `DGW_BONUS_FDR = 3.0` — DGW reduces difficulty contribution (`max(0, fdr - 3)`)
- `FDR_BG` / `FDR_FG` — colour maps for FDR cells (-2 to 6)
- `PREMIER_LEAGUE_TEAMS` — canonical 20-team list for the current season
- `TEAM_ABBREVIATIONS` — 3-letter codes for fixture display strings like `"LIV (H)"`

"""
fpl_api.py — Live FPL Data Module for CoachFPL Command Center
============================================================
Drop this file next to your app.py.
Import with: from fpl_api import FPLData

Usage:
    fpl = FPLData()
    fixtures_df = fpl.get_fixtures_df()   # Replaces your Fixtures202526.csv
    teams_df    = fpl.get_teams_df()      # Team names, strengths
    players_df  = fpl.get_players_df()    # All players with stats
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# ── FPL API endpoints ─────────────────────────────────────────────────────────
BASE_URL       = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL  = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL   = f"{BASE_URL}/fixtures/"
LIVE_URL       = f"{BASE_URL}/event/{{gw}}/live/"   # format with gw number


class FPLData:
    """
    Fetches and caches all FPL data from the official API.
    Data is cached for 30 minutes using Streamlit's cache.
    """

    # ── Fetch raw bootstrap data (teams + players + events) ──────────────────
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner="Fetching live FPL data...")
    def _fetch_bootstrap() -> dict:
        resp = requests.get(BOOTSTRAP_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Fetch all fixtures ────────────────────────────────────────────────────
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner="Fetching fixtures...")
    def _fetch_fixtures() -> list:
        resp = requests.get(FIXTURES_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Fetch live GW points ──────────────────────────────────────────────────
    @staticmethod
    @st.cache_data(ttl=300, show_spinner="Fetching live GW data...")
    def _fetch_live(gw: int) -> dict:
        resp = requests.get(LIVE_URL.format(gw=gw), timeout=15)
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def __init__(self):
        self._bootstrap = self._fetch_bootstrap()
        self._fixtures   = self._fetch_fixtures()

    # ── Teams ─────────────────────────────────────────────────────────────────
    def get_teams_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with team info.
        Columns: id, name, short_name, strength, strength_attack_home,
                 strength_attack_away, strength_defence_home, strength_defence_away
        """
        teams = self._bootstrap["teams"]
        df = pd.DataFrame(teams)[[
            "id", "name", "short_name",
            "strength",
            "strength_attack_home",  "strength_attack_away",
            "strength_defence_home", "strength_defence_away"
        ]]
        return df.set_index("id")

    # ── Fixtures → same shape as your CSV ────────────────────────────────────
    def get_fixtures_df(self) -> pd.DataFrame:
        """
        Returns fixtures in the same format as your Fixtures202526.csv so
        you can drop this in as a replacement with zero other changes.

        Columns: GW, HomeTeam, AwayTeam, HomeTeamID, AwayTeamID,
                 finished, kickoff_time
        """
        teams_df = self.get_teams_df().reset_index()
        id_to_name  = dict(zip(teams_df["id"], teams_df["name"]))
        id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))

        rows = []
        for f in self._fixtures:
            if f["event"] is None:          # postponed / unscheduled
                continue
            rows.append({
                "GW":           f["event"],
                "HomeTeam":     id_to_name.get(f["team_h"], "Unknown"),
                "AwayTeam":     id_to_name.get(f["team_a"], "Unknown"),
                "HomeTeamShort": id_to_short.get(f["team_h"], "???"),
                "AwayTeamShort": id_to_short.get(f["team_a"], "???"),
                "HomeTeamID":   f["team_h"],
                "AwayTeamID":   f["team_a"],
                "HomeTeamDifficulty": f["team_h_difficulty"],   # FPL's own rating
                "AwayTeamDifficulty": f["team_a_difficulty"],
                "finished":     f["finished"],
                "kickoff_time": f.get("kickoff_time"),
            })

        df = pd.DataFrame(rows)
        if "kickoff_time" in df.columns:
            df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
        return df.sort_values(["GW", "HomeTeam"]).reset_index(drop=True)

    # ── Players ───────────────────────────────────────────────────────────────
    def get_players_df(self) -> pd.DataFrame:
        """
        Returns all players with key FPL stats.
        Columns: id, web_name, team_id, position, price, total_points,
                 form, selected_by_percent, transfers_in_event,
                 transfers_out_event, minutes, goals_scored, assists,
                 clean_sheets, expected_goals, expected_assists
        """
        POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

        players = self._bootstrap["elements"]
        df = pd.DataFrame(players)

        keep = [
            "id", "web_name", "team", "element_type",
            "now_cost", "total_points", "form",
            "selected_by_percent",
            "transfers_in_event", "transfers_out_event",
            "minutes", "goals_scored", "assists", "clean_sheets",
            "expected_goals", "expected_assists",
            "expected_goals_conceded",
            "chance_of_playing_next_round",
        ]
        # Only keep columns that actually exist (API can vary)
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        df.rename(columns={
            "team":         "team_id",
            "element_type": "position_id",
            "now_cost":     "price_raw",
        }, inplace=True)

        df["position"] = df["position_id"].map(POSITION_MAP)
        df["price"]    = df["price_raw"] / 10.0     # convert to £m

        # Numeric conversions
        for col in ["form", "selected_by_percent",
                    "expected_goals", "expected_assists",
                    "expected_goals_conceded"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.reset_index(drop=True)

    # ── Gameweek info ─────────────────────────────────────────────────────────
    def get_current_gw(self) -> int:
        """Returns the current (or next upcoming) gameweek number."""
        events = self._bootstrap["events"]
        for ev in events:
            if ev["is_current"]:
                return ev["id"]
        for ev in events:
            if ev["is_next"]:
                return ev["id"]
        return 1

    def get_gw_deadlines(self) -> pd.DataFrame:
        """Returns a DataFrame of GW → deadline_time for all gameweeks."""
        events = self._bootstrap["events"]
        rows = [{"GW": ev["id"],
                 "deadline": ev["deadline_time"],
                 "is_current": ev["is_current"],
                 "is_next": ev["is_next"]}
                for ev in events]
        df = pd.DataFrame(rows)
        df["deadline"] = pd.to_datetime(df["deadline"], utc=True)
        return df

    # ── DGW / BGW detection ───────────────────────────────────────────────────
    def get_dgw_bgw(self, start_gw: int = None, end_gw: int = None) -> dict:
        """
        Returns two dicts:
          dgw[team_id] = list of GWs where team plays twice
          bgw[team_id] = list of GWs where team has no fixture

        Optionally filter by start_gw / end_gw range.
        """
        fixtures_df = self.get_fixtures_df()
        teams_df    = self.get_teams_df()

        if start_gw:
            fixtures_df = fixtures_df[fixtures_df["GW"] >= start_gw]
        if end_gw:
            fixtures_df = fixtures_df[fixtures_df["GW"] <= end_gw]

        all_gws    = sorted(fixtures_df["GW"].unique())
        all_teams  = list(teams_df.index)

        # Count fixtures per team per GW
        counts = {}   # counts[(team_id, gw)] = number of fixtures
        for _, row in fixtures_df.iterrows():
            for team_id in [row["HomeTeamID"], row["AwayTeamID"]]:
                key = (team_id, row["GW"])
                counts[key] = counts.get(key, 0) + 1

        dgw = {t: [] for t in all_teams}
        bgw = {t: [] for t in all_teams}

        for team_id in all_teams:
            for gw in all_gws:
                n = counts.get((team_id, gw), 0)
                if n >= 2:
                    dgw[team_id].append(gw)
                elif n == 0:
                    bgw[team_id].append(gw)

        return {"dgw": dgw, "bgw": bgw}

    # ── Captain suggestions ───────────────────────────────────────────────────
    def get_captain_candidates(self,
                               gw: int,
                               top_n: int = 10) -> pd.DataFrame:
        """
        Returns top captain candidates for a given GW ranked by:
          form × fixture_ease  (higher = better captain pick)

        fixture_ease = 6 - opponent_difficulty  (so 1 = hardest, 5 = easiest)
        """
        fixtures_df = self.get_fixtures_df()
        players_df  = self.get_players_df()
        teams_df    = self.get_teams_df().reset_index()

        gw_fixtures = fixtures_df[fixtures_df["GW"] == gw]

        # Build: team_id → (opponent_short, difficulty, is_home)
        matchups = {}
        for _, f in gw_fixtures.iterrows():
            matchups[f["HomeTeamID"]] = (f["AwayTeamShort"], f["HomeTeamDifficulty"], True)
            matchups[f["AwayTeamID"]] = (f["HomeTeamShort"], f["AwayTeamDifficulty"], False)

        # Filter to attackers + key mids with decent form
        attack_pos = [3, 4]   # MID, FWD position_ids
        df = players_df[players_df["position_id"].isin(attack_pos)].copy()
        df = df[pd.to_numeric(df["form"], errors="coerce") >= 3.0]

        df["opponent"]    = df["team_id"].map(lambda t: matchups.get(t, ("?", 3, False))[0])
        df["difficulty"]  = df["team_id"].map(lambda t: matchups.get(t, ("?", 3, False))[1])
        df["is_home"]     = df["team_id"].map(lambda t: matchups.get(t, ("?", 3, False))[2])
        df["fixture_ease"]= 6 - df["difficulty"]

        df["captain_score"] = pd.to_numeric(df["form"], errors="coerce") * df["fixture_ease"]

        result = df.nlargest(top_n, "captain_score")[[
            "web_name", "position", "price",
            "form", "total_points", "selected_by_percent",
            "opponent", "is_home", "difficulty", "captain_score"
        ]].copy()

        result["home_away"] = result["is_home"].map({True: "(H)", False: "(A)"})
        result["fixture"]   = result["opponent"] + " " + result["home_away"]
        result.drop(columns=["is_home", "opponent", "home_away"], inplace=True)

        result.rename(columns={
            "web_name":            "Player",
            "position":            "Pos",
            "price":               "£",
            "form":                "Form",
            "total_points":        "Pts",
            "selected_by_percent": "Sel%",
            "fixture":             "Fixture",
            "difficulty":          "FDR",
            "captain_score":       "Score",
        }, inplace=True)

        return result.reset_index(drop=True)


# =============================================================================
# STREAMLIT HELPER — paste this into your app.py
# =============================================================================
def render_fpl_api_tab(fpl: FPLData):
    """
    Example tab renderer. Add this as a new tab in your existing app.py:

        tab_fdr, tab_live, tab_captain = st.tabs([
            "Fixture Difficulty (FDR)",
            "📡 Live Data",
            "🎯 Captain Picks"
        ])
        with tab_live:
            render_fpl_api_tab(fpl)
    """
    st.subheader("📡 Live FPL Data")

    current_gw = fpl.get_current_gw()
    st.info(f"Current GW: **{current_gw}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Captain Picks")
        gw_input = st.number_input("GW to analyse",
                                   min_value=1, max_value=38,
                                   value=current_gw)
        captain_df = fpl.get_captain_candidates(gw_input, top_n=10)
        st.dataframe(
            captain_df.style.background_gradient(
                subset=["Score"], cmap="Greens"
            ).format({
                "£": "£{:.1f}m",
                "Form": "{:.1f}",
                "Score": "{:.1f}",
                "Sel%": "{:.1f}%"
            }),
            use_container_width=True
        )

    with col2:
        st.markdown("### ⚠️ DGW / BGW Radar")
        dgw_bgw = fpl.get_dgw_bgw(start_gw=current_gw,
                                   end_gw=min(current_gw + 5, 38))
        teams_df = fpl.get_teams_df()

        dgw_teams = [(teams_df.loc[t, "name"], gws)
                     for t, gws in dgw_bgw["dgw"].items() if gws]
        bgw_teams = [(teams_df.loc[t, "name"], gws)
                     for t, gws in dgw_bgw["bgw"].items() if gws]

        if dgw_teams:
            st.markdown("**🟢 Double Gameweeks (next 5 GWs)**")
            for name, gws in sorted(dgw_teams):
                st.write(f"- {name}: GW{', GW'.join(map(str, gws))}")
        else:
            st.write("No DGWs detected in next 5 GWs.")

        if bgw_teams:
            st.markdown("**🔴 Blank Gameweeks (next 5 GWs)**")
            for name, gws in sorted(bgw_teams):
                st.write(f"- {name}: GW{', GW'.join(map(str, gws))}")

    st.markdown("---")
    st.markdown("### 📅 GW Deadlines")
    deadlines = fpl.get_gw_deadlines()
    upcoming  = deadlines[deadlines["GW"] >= current_gw].head(10)
    upcoming  = upcoming.copy()
    upcoming["deadline"] = upcoming["deadline"].dt.strftime("%a %d %b %Y — %H:%M UTC")
    st.dataframe(upcoming[["GW", "deadline", "is_current", "is_next"]],
                 use_container_width=True, hide_index=True)

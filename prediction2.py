import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# ==============================
# 1. Setup & Config
# ==============================

load_dotenv()
STRATZ_TOKEN = os.getenv("STRATZ_TOKEN")
if not STRATZ_TOKEN:
    raise ValueError("STRATZ_TOKEN not found. Please add it to your .env file.")

TEAM_IDS = {
    "Xtreme Gaming": 8261500,
    "BB Team": 8255888,
    "Team Tidebound": 9640842,
    "Heroic": 9303484,
    "Pvision": 9572001,
    "Nigma Galaxy": 7554697,
    "Tundra Esports": 8291895,
    "Team Falcons": 9247354
}

cache_file = "dota2_stratz_match_data3.csv"


# ==============================
# 2. Fetch Data from Stratz
# ==============================

def get_team_match_history_stratz(team_name, team_id, take=50, skip=0):
    api_url = "https://api.stratz.com/graphql"
    headers = {
        "Authorization": f"Bearer {STRATZ_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"
    }

    query = f"""
    {{
      team(teamId: {team_id}) {{
        matches(request: {{ take: {take}, skip: {skip} }}) {{
          id
          didRadiantWin
          startDateTime
          radiantTeamId
          direTeamId
        }}
      }}
    }}
    """

    try:
        response = requests.post(api_url, headers=headers, json={"query": query})
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            print(f"GraphQL errors for {team_name}: {data['errors']}")
            return []

        matches = []
        raw_matches = data.get("data", {}).get("team", {}).get("matches", [])

        for match in raw_matches:
            match_id = match.get("id")
            if not match_id:
                continue

            radiant_team_id = match["radiantTeamId"]
            dire_team_id = match["direTeamId"]
            radiant_win = match["didRadiantWin"]
            start_time = datetime.fromtimestamp(match["startDateTime"])

            radiant_team_name = next((name for name, id in TEAM_IDS.items() if id == radiant_team_id),
                                     f"Team_{radiant_team_id}")
            dire_team_name = next((name for name, id in TEAM_IDS.items() if id == dire_team_id),
                                  f"Team_{dire_team_id}")

            winner = radiant_team_name if radiant_win else dire_team_name

            matches.append({
                "Team1": radiant_team_name,
                "Team2": dire_team_name,
                "Winner": winner,
                "Date": start_time
            })

        return matches

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {team_name}: {e}")
        return []


# ==============================
# 3. Load or Fetch Data
# ==============================

if os.path.exists(cache_file):
    print(f"Found cache file '{cache_file}'. Loading data...")
    df = pd.read_csv(cache_file)
    df["Date"] = pd.to_datetime(df["Date"])
else:
    print("No cache found. Fetching from API...")
    all_matches = []
    for team_name, team_id in TEAM_IDS.items():
        print(f"Fetching matches for {team_name}...")
        matches = get_team_match_history_stratz(team_name, team_id, take=50)
        all_matches.extend(matches)
        time.sleep(2)

    df = pd.DataFrame(all_matches)
    if not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"Saved {len(df)} matches to cache.")

if df.empty:
    raise ValueError("No data available. Cannot continue.")

df.drop_duplicates(subset=["Team1", "Team2", "Date"], inplace=True)


# ==============================
# 4. Feature Engineering
# ==============================

team_stats = {}
teams = list(TEAM_IDS.keys())

# Overall win rate
for team in teams:
    total_matches = len(df[(df["Team1"] == team) | (df["Team2"] == team)])
    wins = len(df[df["Winner"] == team])
    win_rate = wins / total_matches if total_matches > 0 else 0.5
    team_stats[team] = {"overall_wr": win_rate}

# Recent win rate (last 30 days)
recent_cutoff = datetime.now() - pd.Timedelta(days=30)
recent_df = df[df["Date"] > recent_cutoff]
for team in teams:
    total_matches = len(recent_df[(recent_df["Team1"] == team) | (recent_df["Team2"] == team)])
    wins = len(recent_df[recent_df["Winner"] == team])
    win_rate = wins / total_matches if total_matches > 0 else 0.5
    team_stats[team]["recent_wr"] = win_rate

# Head-to-head win rate
def head_to_head_wr(team1, team2, lookback_days=180):
    cutoff = datetime.now() - pd.Timedelta(days=lookback_days)
    h2h_matches = df[
        (((df["Team1"] == team1) & (df["Team2"] == team2)) |
         ((df["Team1"] == team2) & (df["Team2"] == team1)))
        & (df["Date"] > cutoff)
    ]
    if len(h2h_matches) == 0:
        return 0.5
    wins = len(h2h_matches[h2h_matches["Winner"] == team1])
    return wins / len(h2h_matches)


# Dataset
X, y = [], []
for _, row in df.iterrows():
    team1, team2 = row["Team1"], row["Team2"]

    overall_wr_diff = team_stats.get(team1, {}).get("overall_wr", 0.5) - team_stats.get(team2, {}).get("overall_wr", 0.5)
    recent_wr_diff = team_stats.get(team1, {}).get("recent_wr", 0.5) - team_stats.get(team2, {}).get("recent_wr", 0.5)
    h2h_wr = head_to_head_wr(team1, team2)

    X.append([overall_wr_diff, recent_wr_diff, h2h_wr])
    y.append(1 if row["Winner"] == team1 else 0)

X = np.array(X)
y = np.array(y)


# ==============================
# 5. Train & Evaluate
# ==============================

if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}\n")
else:
    raise ValueError("Not enough data to train.")


# ==============================
# 6. Prediction Helper
# ==============================

def predict_match(team1_name, team2_name, model, team_stats):
    overall_wr_diff = team_stats[team1_name]["overall_wr"] - team_stats[team2_name]["overall_wr"]
    recent_wr_diff = team_stats[team1_name]["recent_wr"] - team_stats[team2_name]["recent_wr"]
    h2h_wr = head_to_head_wr(team1_name, team2_name)

    feature_vector = np.array([overall_wr_diff, recent_wr_diff, h2h_wr]).reshape(1, -1)
    prediction = model.predict(feature_vector)
    probability = model.predict_proba(feature_vector)[0]

    if prediction[0] == 1:
        return team1_name, probability[1]
    else:
        return team2_name, probability[0]


# ==============================
# 7. Bracket Simulation
# ==============================

def simulate_ti_bracket():
    # Upper Bracket Quarterfinals
    ub_qf = [
        ("Xtreme Gaming", "Tundra Esports"),
        ("Pvision", "Heroic"),
        ("Team Tidebound", "Team Falcons"),
        ("BB Team", "Nigma Galaxy")
    ]

    print("==== TI14 Bracket Simulation ====\n")
    winners = []
    losers = []

    print("Upper Bracket Quarterfinals:")
    for t1, t2 in ub_qf:
        winner, conf = predict_match(t1, t2, model, team_stats)
        loser = t1 if winner == t2 else t2
        winners.append(winner)
        losers.append(loser)
        print(f"{t1} vs {t2} -> {winner} (Conf {conf:.2f})")

    # UB Semifinals
    print("\nUpper Bracket Semifinals:")
    ub_sf_pairs = [(winners[0], winners[1]), (winners[2], winners[3])]
    ub_sf_winners = []
    for t1, t2 in ub_sf_pairs:
        winner, conf = predict_match(t1, t2, model, team_stats)
        loser = t1 if winner == t2 else t2
        ub_sf_winners.append(winner)
        losers.append(loser)
        print(f"{t1} vs {t2} -> {winner} (Conf {conf:.2f})")

    # UB Final
    print("\nUpper Bracket Final:")
    t1, t2 = ub_sf_winners
    ub_winner, conf = predict_match(t1, t2, model, team_stats)
    ub_loser = t1 if ub_winner == t2 else t2
    losers.append(ub_loser)
    print(f"{t1} vs {t2} -> {ub_winner} (Conf {conf:.2f})")

    # Lower Bracket Rounds
    print("\nLower Bracket:")
    lb_active = losers.copy()
    while len(lb_active) > 1:
        next_round = []
        for i in range(0, len(lb_active), 2):
            if i + 1 >= len(lb_active):
                next_round.append(lb_active[i])
                continue
            t1, t2 = lb_active[i], lb_active[i + 1]
            winner, conf = predict_match(t1, t2, model, team_stats)
            loser = t1 if winner == t2 else t2
            next_round.append(winner)
            print(f"{t1} vs {t2} -> {winner} (Conf {conf:.2f})")
        lb_active = next_round

    lb_winner = lb_active[0]
    print(f"\nLower Bracket Winner: {lb_winner}")

    # Grand Final
    print("\nGrand Final:")
    t1, t2 = ub_winner, lb_winner
    gf_winner, conf = predict_match(t1, t2, model, team_stats)
    print(f"{t1} vs {t2} -> {gf_winner} (Conf {conf:.2f})")

    print("\n==== Simulation Complete ====")


simulate_ti_bracket()

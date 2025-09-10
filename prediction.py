import pandas as pd
import requests
import numpy as np
from datetime import datetime
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os

# -----------------------------
# 1. User Settings
# -----------------------------

# Load variables from .env file
load_dotenv()

# Get your Stratz token
STRATZ_TOKEN = os.getenv("STRATZ_TOKEN")

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

CACHE_FILE = "dota2_stratz_match_data.csv"

# -----------------------------
# 2. Fetch Team Match History
# -----------------------------
def get_team_match_history_stratz(team_name, team_id, take=50, skip=0):
    if not STRATZ_TOKEN or STRATZ_TOKEN == "YOUR_STRATZ_API_TOKEN":
        print("API token not set. Please replace STRATZ_TOKEN with your actual token.")
        return []

    api_url = "https://api.stratz.com/graphql"
    headers = {
        "Authorization": f"Bearer {STRATZ_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"
    }

    query = f"""
    {{
      team(teamId: {team_id}) {{
        matches(request: {{ skip: {skip}, take: {take} }}) {{
          id
          didRadiantWin
          startDateTime
          radiantTeam {{
            id
            name
          }}
          direTeam {{
            id
            name
          }}
        }}
      }}
    }}
    """

    payload = {"query": query}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if 'errors' in data:
            print(f"GraphQL errors for {team_name}: {data['errors']}")
            return []

        matches_data = data.get('data', {}).get('team', {}).get('matches', [])
        if not matches_data:
            print(f"No matches found for {team_name}")
            return []

        matches = []
        for match in matches_data:
            radiant_name = match['radiantTeam']['name'] if match.get('radiantTeam') else f"Team_{match['radiantTeam']['id']}"
            dire_name = match['direTeam']['name'] if match.get('direTeam') else f"Team_{match['direTeam']['id']}"
            radiant_win = match['didRadiantWin']
            winner = radiant_name if radiant_win else dire_name
            start_time = datetime.fromtimestamp(match['startDateTime'])

            matches.append({
                'Team1': radiant_name,
                'Team2': dire_name,
                'Winner': winner,
                'Date': start_time
            })

        return matches

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {team_name}: {e}")
        return []

# -----------------------------
# 3. Load or Fetch Match Data
# -----------------------------
if os.path.exists(CACHE_FILE):
    df = pd.read_csv(CACHE_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} matches from cache.")
else:
    all_matches = []
    for name, team_id in TEAM_IDS.items():
        print(f"Fetching matches for {name}...")
        matches = get_team_match_history_stratz(name, team_id, take=50)
        if matches:
            all_matches.extend(matches)
        time.sleep(1)  # avoid rate limits

    df = pd.DataFrame(all_matches)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
    print(f"Fetched {len(df)} total matches.")

# Remove duplicates
df.drop_duplicates(subset=['Team1','Team2','Date'], inplace=True)

if df.empty:
    print("No match data available. Exiting.")
    exit()

# -----------------------------
# 4. Compute Team Stats (Win Rates)
# -----------------------------
team_stats = {}
teams = list(TEAM_IDS.keys())

for team in teams:
    total = len(df[(df['Team1']==team) | (df['Team2']==team)])
    wins = len(df[df['Winner']==team])
    team_stats[team] = wins/total if total>0 else 0.5
    print(f"{team}: {wins}/{total} wins (Win Rate: {team_stats[team]:.3f})")

# -----------------------------
# 5. Prepare ML Dataset
# -----------------------------
X, y = [], []

for _, row in df.iterrows():
    t1, t2 = row['Team1'], row['Team2']
    feature = [team_stats.get(t1,0.5) - team_stats.get(t2,0.5)]
    label = 1 if row['Winner']==t1 else 0
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)

if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test,y_pred):.2f}")
else:
    print("Not enough data to train the model.")
    exit()

# -----------------------------
# 6. Helper Functions
# -----------------------------
def predict_match(team1, team2):
    """Return winner, loser, and confidence"""
    if team2 is None:
        # bye
        return team1, None, 1.0
    if team1 not in team_stats or team2 not in team_stats:
        return None, None, 0.5
    feature = np.array([team_stats[team1]-team_stats[team2]]).reshape(1,-1)
    pred = model.predict(feature)[0]
    prob = model.predict_proba(feature)[0]
    if pred==1:
        return team1, team2, prob[1]
    else:
        return team2, team1, prob[0]

def pair_teams(teams):
    """Return list of match tuples, handle odd numbers by giving a bye."""
    matches = []
    i = 0
    while i < len(teams):
        if i+1 < len(teams):
            matches.append((teams[i], teams[i+1]))
            i += 2
        else:
            print(f"{teams[i]} gets a bye this round")
            matches.append((teams[i], None))  # None means bye
            i += 1
    return matches

def simulate_round(matches, round_name, bracket_type="UB"):
    next_round = []
    losers = []
    print(f"\n--- {bracket_type} {round_name} ---")
    for t1, t2 in matches:
        winner, loser, confidence = predict_match(t1, t2)
        if t2 is None:
            next_round.append(winner)
            continue
        print(f"{t1} vs {t2} -> Winner: {winner} (Confidence: {confidence:.3f})")
        print(f"  Win Rates: {t1}({team_stats.get(t1,0.5):.3f}) vs {t2}({team_stats.get(t2,0.5):.3f})\n")
        next_round.append(winner)
        losers.append(loser)
    return next_round, losers

# -----------------------------
# 7. Define TI Bracket
# -----------------------------
ub_round1 = [
    ("Xtreme Gaming", "Tundra Esports"),
    ("Pvision", "Heroic"),
    ("Team Tidebound", "Team Falcons"),
    ("BB Team", "Nigma Galaxy")
]

# -----------------------------
# 8. Simulate Tournament
# -----------------------------
ub_current = ub_round1
lb_teams = []

ub_winners = []

round_num = 1
while len(ub_current) > 0:
    winners, losers = simulate_round(ub_current, f"Round {round_num}", "UB")
    ub_winners.append(winners)
    lb_teams.extend(losers)
    if len(winners) > 1:
        ub_current = pair_teams(winners)
    else:
        ub_current = []
    round_num += 1

# Simulate Lower Bracket
lb_matches = pair_teams(lb_teams)
lb_winners = []
lb_round_num = 1
while len(lb_matches) > 0:
    winners, eliminated = simulate_round(lb_matches, f"LB Round {lb_round_num}", "LB")
    lb_winners = winners
    if len(winners) > 1:
        lb_matches = pair_teams(winners)
    else:
        break
    lb_round_num += 1

# Grand Final
if ub_winners and lb_winners:
    ub_final_winner = ub_winners[-1][0]
    lb_final_winner = lb_winners[0]
    print("\n--- Grand Final ---")
    winner, loser, confidence = predict_match(ub_final_winner, lb_final_winner)
    print(f"{ub_final_winner} vs {lb_final_winner} -> Winner: {winner} (Confidence: {confidence:.3f})")

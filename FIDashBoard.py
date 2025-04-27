import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from statsbombpy import sb

# Constants
PITCH_LENGTH = 105
PITCH_WIDTH = 68

# Helper function for coordinate transformation
def transform_coords(statsbomb_x, statsbomb_y):
    """Transform StatsBomb coordinates (120x80) to pitch coordinates (105x68)."""
    return (statsbomb_x / 120 * PITCH_LENGTH, statsbomb_y / 80 * PITCH_WIDTH)

# -------------------------
# Data Acquisition from StatsBomb
# -------------------------
match_events = sb.events(match_id=3869685)
# match_events = sb.events(match_id=3869685)

# Extract shot data for xG model
shot_data = match_events[match_events['type'] == 'Shot'].copy()
shot_data['shot_distance'] = np.sqrt((120 - shot_data['location'].apply(lambda x: x[0])) ** 2 +
                                     (40 - shot_data['location'].apply(lambda x: x[1])) ** 2)
shot_data['shot_angle'] = np.arctan2(40 - shot_data['location'].apply(lambda x: x[1]),
                                     120 - shot_data['location'].apply(lambda x: x[0])).abs() * 180 / np.pi
shot_data['is_goal'] = shot_data['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

# Train xG model
features = shot_data[['shot_distance', 'shot_angle']]
target = shot_data['is_goal']
xg_classifier = LogisticRegression().fit(features, target)
shot_data['xG_value'] = xg_classifier.predict_proba(features)[:, 1]

# Convert shot locations to pitch coordinates
shot_data['pitch_x'] = shot_data['location'].apply(lambda loc: transform_coords(loc[0], loc[1])[0])
shot_data['pitch_y'] = shot_data['location'].apply(lambda loc: transform_coords(loc[0], loc[1])[1])

# Calculate optimal positions based on high xG shots
arg_high_xg = shot_data[(shot_data['team'] == 'Argentina') & (shot_data['xG_value'] > 0.3)]
if not arg_high_xg.empty:
    arg_optimal_x = arg_high_xg['pitch_x'].mean()
    arg_optimal_y = arg_high_xg['pitch_y'].mean()
else:
    arg_optimal_x, arg_optimal_y = 90, 34

fra_high_xg = shot_data[(shot_data['team'] == 'France') & (shot_data['xG_value'] > 0.3)]
if not fra_high_xg.empty:
    fra_optimal_x = fra_high_xg['pitch_x'].mean()
    fra_optimal_y = fra_high_xg['pitch_y'].mean()
else:
    fra_optimal_x, fra_optimal_y = 10, 34

# Team statistics
team_stats = pd.DataFrame({
    'Squad': ['Argentina', 'France'],
    'Attack_xG': [shot_data[shot_data['team'] == 'Argentina']['xG_value'].sum(),
                  shot_data[shot_data['team'] == 'France']['xG_value'].sum()],
    'Defense_xG': [shot_data[shot_data['team'] == 'France']['xG_value'].sum(),
                   shot_data[shot_data['team'] == 'Argentina']['xG_value'].sum()]
})
team_stats['xG_balance'] = team_stats['Attack_xG'] - team_stats['Defense_xG']

# -------------------------
# Player Data and Starting Positions
# -------------------------
arg_squad = [
    "Emiliano Martinez", "Nahuel Molina", "Cristian Romero", "Nicolas Otamendi",
    "Nicolas Tagliafico", "Rodrigo De Paul", "Leandro Paredes", "Alexis Mac Allister",
    "Angel Di Maria", "Lionel Messi", "Julian Alvarez"
]

arg_formations = {
    "Emiliano Martinez": (5, 34),
    "Nahuel Molina": (20, 10),
    "Cristian Romero": (20, 30),
    "Nicolas Otamendi": (20, 50),
    "Nicolas Tagliafico": (20, 70),
    "Rodrigo De Paul": (40, 20),
    "Leandro Paredes": (45, 34),
    "Alexis Mac Allister": (40, 50),
    "Angel Di Maria": (60, 20),
    "Lionel Messi": (65, 34),
    "Julian Alvarez": (70, 50)
}

fra_squad = [
    "Hugo Lloris", "Jules Koundé", "Raphaël Varane", "Dayot Upamecano", "Theo Hernández",
    "Aurélien Tchouaméni", "Adrien Rabiot", "Ousmane Dembélé", "Antoine Griezmann",
    "Kylian Mbappé", "Olivier Giroud"
]

fra_formations = {
    "Hugo Lloris": (100, 34),
    "Jules Koundé": (85, 60),
    "Raphaël Varane": (85, 40),
    "Dayot Upamecano": (85, 28),
    "Theo Hernández": (85, 8),
    "Aurélien Tchouaméni": (75, 34),
    "Adrien Rabiot": (70, 34),
    "Ousmane Dembélé": (60, 60),
    "Antoine Griezmann": (60, 34),
    "Kylian Mbappé": (60, 8),
    "Olivier Giroud": (55, 34)
}

player_locations = {**arg_formations, **fra_formations}

# Player performance metrics
player_metrics = {
    "Lionel Messi": {"games": 6, "scores": 5, "assists": 3, "passes_completed": 320, "rating": 8.4},
    "Julian Alvarez": {"games": 6, "scores": 4, "assists": 0, "passes_completed": 180, "rating": 7.9},
    "Angel Di Maria": {"games": 4, "scores": 1, "assists": 1, "passes_completed": 120, "rating": 7.5},
    "Rodrigo De Paul": {"games": 6, "scores": 0, "assists": 1, "passes_completed": 290, "rating": 7.6},
    "Leandro Paredes": {"games": 5, "scores": 0, "assists": 0, "passes_completed": 200, "rating": 7.3},
    "Alexis Mac Allister": {"games": 6, "scores": 1, "assists": 1, "passes_completed": 250, "rating": 7.7},
    "Emiliano Martinez": {"games": 6, "scores": 0, "assists": 0, "passes_completed": 50, "rating": 7.2},
    "Cristian Romero": {"games": 6, "scores": 0, "assists": 0, "passes_completed": 120, "rating": 7.5},
    "Nicolas Otamendi": {"games": 6, "scores": 0, "assists": 0, "passes_completed": 180, "rating": 7.4},
    "Nicolas Tagliafico": {"games": 5, "scores": 0, "assists": 0, "passes_completed": 150, "rating": 7.3},
    "Nahuel Molina": {"games": 6, "scores": 1, "assists": 1, "passes_completed": 170, "rating": 7.4}
}

# Player ID to name mapping
team_lineups = sb.lineups(match_id=3869685)
id_to_player = {}
for team in team_lineups:
    for _, row in team_lineups[team].iterrows():
        id_to_player[row['player_id']] = row['player_name']

# Player roles
player_roles_dict = {
    "Emiliano Martinez": "GK",
    "Nahuel Molina": "DEF",
    "Cristian Romero": "DEF",
    "Nicolas Otamendi": "DEF",
    "Nicolas Tagliafico": "DEF",
    "Rodrigo De Paul": "MID",
    "Leandro Paredes": "MID",
    "Alexis Mac Allister": "MID",
    "Angel Di Maria": "FWD",
    "Lionel Messi": "FWD",
    "Julian Alvarez": "FWD",
    "Hugo Lloris": "GK",
    "Jules Koundé": "DEF",
    "Raphaël Varane": "DEF",
    "Dayot Upamecano": "DEF",
    "Theo Hernández": "DEF",
    "Aurélien Tchouaméni": "MID",
    "Adrien Rabiot": "MID",
    "Ousmane Dembélé": "MID",
    "Antoine Griezmann": "MID",
    "Kylian Mbappé": "FWD",
    "Olivier Giroud": "FWD"
}

# -------------------------
# Utility Functions
# -------------------------
def get_movement_bounds(player_name):
    """Define movement boundaries based on player role and team."""
    role = player_roles_dict[player_name]
    squad = 'Argentina' if player_name in arg_squad else 'France'
    if role == 'GK':
        return (0, 18, 22, 46) if squad == 'Argentina' else (87, 105, 22, 46)
    elif role == 'DEF':
        return (0, 52.5, 0, 68) if squad == 'Argentina' else (52.5, 105, 0, 68)
    elif role == 'MID':
        return (0, 105, 0, 68)
    elif role == 'FWD':
        return (52.5, 105, 0, 68) if squad == 'Argentina' else (0, 52.5, 0, 68)

def adjust_player_locations(step):
    """Adjust player locations with smooth transitions and occasional dynamic moves."""
    current_action = match_events.iloc[step % len(match_events)]
    transition_rate = 0.3
    dynamic_move_prob = 0.2

    updated_players = []
    if 'player_id' in current_action and current_action['player_id'] in id_to_player:
        player_name = id_to_player[current_action['player_id']]
        if player_name in player_locations and 'location' in current_action:
            sb_x, sb_y = current_action['location']
            dest_x, dest_y = transform_coords(sb_x, sb_y)
            curr_x, curr_y = player_locations[player_name]
            new_x = curr_x + transition_rate * (dest_x - curr_x)
            new_y = curr_y + transition_rate * (dest_y - curr_y)
            player_locations[player_name] = (new_x, new_y)
            updated_players.append(player_name)

    if current_action['type'] == 'Pass' and 'pass_recipient_id' in current_action:
        recipient_id = current_action['pass_recipient_id']
        if recipient_id in id_to_player:
            recipient_name = id_to_player[recipient_id]
            if recipient_name in player_locations and 'pass_end_location' in current_action:
                sb_x, sb_y = current_action['pass_end_location']
                dest_x, dest_y = transform_coords(sb_x, sb_y)
                curr_x, curr_y = player_locations[recipient_name]
                new_x = curr_x + transition_rate * (dest_x - curr_x)
                new_y = curr_y + transition_rate * (dest_y - curr_y)
                player_locations[recipient_name] = (new_x, new_y)
                updated_players.append(recipient_name)

    for player_name in player_locations:
        if player_name not in updated_players:
            curr_x, curr_y = player_locations[player_name]
            role = player_roles_dict[player_name]
            if role != 'GK' and random.random() < dynamic_move_prob:
                dx = random.uniform(-2.0, 2.0)
                dy = random.uniform(-2.0, 2.0)
            else:
                dx = random.uniform(-0.2, 0.2) if role == 'GK' else random.uniform(-0.5, 0.5)
                dy = random.uniform(-0.2, 0.2) if role == 'GK' else random.uniform(-0.5, 0.5)
            target_x = curr_x + dx
            target_y = curr_y + dy
            new_x = curr_x + transition_rate * (target_x - curr_x)
            new_y = curr_y + transition_rate * (target_y - curr_y)
            x_min, x_max, y_min, y_max = get_movement_bounds(player_name)
            new_x = max(x_min, min(new_x, x_max))
            new_y = max(y_min, min(new_y, y_max))
            player_locations[player_name] = (new_x, new_y)

def tactical_advice(player_name, squad):
    """Provide tactical advice and optimal positioning based on role and team stats."""
    role = player_roles_dict[player_name]
    player_stats = player_metrics.get(player_name, {})
    team_data = team_stats[team_stats['Squad'] == squad].iloc[0]
    curr_x, curr_y = player_locations[player_name]

    if role == 'FWD':
        opt_x, opt_y = (arg_optimal_x, arg_optimal_y) if squad == 'Argentina' else (fra_optimal_x, fra_optimal_y)
    elif role == 'DEF':
        opt_x, opt_y = (fra_optimal_x, fra_optimal_y) if squad == 'Argentina' else (arg_optimal_x, arg_optimal_y)
    elif role == 'MID':
        opt_x = (arg_optimal_x + fra_optimal_x) / 2
        opt_y = (arg_optimal_y + fra_optimal_y) / 2
    elif role == 'GK':
        opt_x, opt_y = (5, 34) if squad == 'Argentina' else (100, 34)

    if role == 'FWD':
        if team_data['xG_balance'] > 0.5:
            if curr_x < opt_x - 10:
                advice = f"Charge to ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—exploit gaps with speed."
            elif abs(curr_y - 34) > 15:
                advice = f"Move centrally to ({opt_x:.1f}, {opt_y:.1f}). Edge (xG balance {team_data['xG_balance']:.2f})—aim for goal."
            elif curr_x > 85:
                advice = f"Stay at ({opt_x:.1f}, {opt_y:.1f}). Advantage (xG balance {team_data['xG_balance']:.2f})—lure defenders."
            elif random.random() < 0.3:
                advice = f"Drop from ({curr_x:.1f}, {curr_y:.1f}) to link. Lead (xG balance {team_data['xG_balance']:.2f})—create space."
            else:
                advice = f"Attack ({opt_x:.1f}, {opt_y:.1f}). Dominance (xG balance {team_data['xG_balance']:.2f})—score in box."
        else:
            if curr_x < opt_x - 10:
                advice = f"Advance to ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—seek openings."
            elif abs(curr_y - 34) > 15:
                advice = f"Drift to ({opt_x:.1f}, {opt_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—use wide gaps."
            elif curr_x > 85:
                advice = f"Hold at ({opt_x:.1f}, {opt_y:.1f}). Even (xG balance {team_data['xG_balance']:.2f})—wait for counter."
            elif random.random() < 0.3:
                advice = f"Support from ({curr_x:.1f}, {curr_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—aid midfield."
            else:
                advice = f"Move to ({opt_x:.1f}, {opt_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—strike smartly."

    elif role == 'DEF':
        if team_data['xG_balance'] > 0.5:
            if curr_x > opt_x + 10:
                advice = f"Press to ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—mark forwards."
            elif abs(curr_y - opt_y) > 10:
                advice = f"Cover ({opt_x:.1f}, {opt_y:.1f}). Edge (xG balance {team_data['xG_balance']:.2f})—block wings."
            elif curr_x < 20:
                advice = f"Hold at ({curr_x:.1f}, {curr_y:.1f}). Advantage (xG balance {team_data['xG_balance']:.2f})—stop counters."
            elif random.random() < 0.3:
                advice = f"Intercept from ({curr_x:.1f}, {curr_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—break play."
            else:
                advice = f"Lock ({opt_x:.1f}, {opt_y:.1f}). Dominance (xG balance {team_data['xG_balance']:.2f})—secure danger area."
        else:
            if curr_x > opt_x + 10:
                advice = f"Retreat to ({opt_x:.1f}, {opt_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—stay tight."
            elif abs(curr_y - opt_y) > 10:
                advice = f"Adjust to ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—watch wingers."
            elif curr_x < 20:
                advice = f"Guard at ({curr_x:.1f}, {curr_y:.1f}). Even (xG balance {team_data['xG_balance']:.2f})—protect box."
            elif random.random() < 0.3:
                advice = f"Hold at ({curr_x:.1f}, {curr_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—track runners."
            else:
                advice = f"Defend ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—block shots."

    elif role == 'MID':
        if team_data['xG_balance'] > 0.5:
            if curr_x < 40:
                advice = f"Advance to ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—push play."
            elif abs(curr_y - 34) > 20:
                advice = f"Move to ({opt_x:.1f}, {opt_y:.1f}). Edge (xG balance {team_data['xG_balance']:.2f})—use flanks."
            elif curr_x > 70:
                advice = f"Support at ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—feed attackers."
            elif random.random() < 0.3:
                advice = f"Recycle from ({curr_x:.1f}, {curr_y:.1f}). Advantage (xG balance {team_data['xG_balance']:.2f})—keep ball."
            else:
                advice = f"Control ({opt_x:.1f}, {opt_y:.1f}). Dominance (xG balance {team_data['xG_balance']:.2f})—break lines."
        else:
            if curr_x < 40:
                advice = f"Push to ({opt_x:.1f}, {opt_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—link play."
            elif abs(curr_y - 34) > 20:
                advice = f"Cover ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—shield flanks."
            elif curr_x > 70:
                advice = f"Hold at ({opt_x:.1f}, {opt_y:.1f}). Even (xG balance {team_data['xG_balance']:.2f})—aid counters."
            elif random.random() < 0.3:
                advice = f"Stay at ({curr_x:.1f}, {curr_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—disrupt press."
            else:
                advice = f"Balance ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—maintain shape."

    elif role == 'GK':
        if team_data['xG_balance'] > 0.5:
            if curr_x > 10 and squad == 'Argentina':
                advice = f"Move to ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—distribute boldly."
            elif curr_x < 95 and squad == 'France':
                advice = f"Shift to ({opt_x:.1f}, {opt_y:.1f}). Edge (xG balance {team_data['xG_balance']:.2f})—launch attacks."
            elif abs(curr_y - 34) > 5:
                advice = f"Adjust to ({opt_x:.1f}, {opt_y:.1f}). Lead (xG balance {team_data['xG_balance']:.2f})—cover angles."
            elif random.random() < 0.3:
                advice = f"Organize from ({curr_x:.1f}, {curr_y:.1f}). Advantage (xG balance {team_data['xG_balance']:.2f})—lead defense."
            else:
                advice = f"Command ({opt_x:.1f}, {opt_y:.1f}). Dominance (xG balance {team_data['xG_balance']:.2f})—pass accurately."
        else:
            if curr_x > 10 and squad == 'Argentina':
                advice = f"Stay at ({opt_x:.1f}, {opt_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—be ready."
            elif curr_x < 95 and squad == 'France':
                advice = f"Move to ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—watch shots."
            elif abs(curr_y - 34) > 5:
                advice = f"Shift to ({opt_x:.1f}, {opt_y:.1f}). Even (xG balance {team_data['xG_balance']:.2f})—guard crosses."
            elif random.random() < 0.3:
                advice = f"Hold at ({curr_x:.1f}, {curr_y:.1f}). Tight (xG balance {team_data['xG_balance']:.2f})—organize backline."
            else:
                advice = f"Protect ({opt_x:.1f}, {opt_y:.1f}). Close (xG balance {team_data['xG_balance']:.2f})—make saves."

    if player_stats.get('scores', 0) > 2:
        player_tip = "Your scoring form is hot—take more shots and challenge the keeper."
    elif player_stats.get('assists', 0) > 1:
        player_tip = "Your playmaking shines—find teammates with precise passes."
    elif player_stats.get('passes_completed', 0) > 100:
        player_tip = "You control the game—dictate tempo with sharp passing."
    elif player_stats.get('rating', 0) > 7.5:
        player_tip = "You’re a star—lead the team and make the difference."
    else:
        player_tip = "Stay focused—work with teammates to shift momentum."

    return f"{advice} {player_tip}", (opt_x, opt_y)

def generate_pitch_visual(selected_player=None):
    """Generate pitch visualization with player positions and field markings."""
    plot_traces = []

    # Pitch markings
    plot_traces.append(go.Scatter(x=[0, 105, 105, 0, 0], y=[0, 0, 68, 68, 0], mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[52.5, 52.5], y=[0, 68], mode="lines",
                                  line=dict(color="white", width=2, dash="dash"), showlegend=False))
    theta = np.linspace(0, 2*np.pi, 100)
    center_x = 52.5 + 9.15 * np.cos(theta)
    center_y = 34 + 9.15 * np.sin(theta)
    plot_traces.append(go.Scatter(x=center_x, y=center_y, mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[0, 16.5, 16.5, 0, 0], y=[13.84, 13.84, 54.16, 54.16, 13.84], mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[88.5, 105, 105, 88.5, 88.5], y=[13.84, 13.84, 54.16, 54.16, 13.84], mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[0, 5.5, 5.5, 0, 0], y=[24.84, 24.84, 43.16, 43.16, 24.84], mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[99.5, 105, 105, 99.5, 99.5], y=[24.84, 24.84, 43.16, 43.16, 24.84], mode="lines",
                                  line=dict(color="white", width=2), showlegend=False))
    plot_traces.append(go.Scatter(x=[0, 0], y=[30.34, 37.66], mode="lines",
                                  line=dict(color="white", width=4), showlegend=False))
    plot_traces.append(go.Scatter(x=[105, 105], y=[30.34, 37.66], mode="lines",
                                  line=dict(color="white", width=4), showlegend=False))

    # Player positions
    for player_name, (x, y) in player_locations.items():
        marker_color = "blue" if player_name in arg_squad else "red"
        plot_traces.append(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[player_name], textposition="top center",
            marker=dict(size=12, color=marker_color, line=dict(width=2, color='black')),
            customdata=[player_name], hovertemplate=f"<b>{player_name}</b><br>x: %{{x:.2f}}, y: %{{y:.2f}}<extra></extra>"
        ))

        if selected_player == player_name and player_name in arg_squad:
            squad = 'Argentina'
            _, (opt_x, opt_y) = tactical_advice(player_name, squad)
            plot_traces.append(go.Scatter(
                x=[x, opt_x], y=[y, opt_y], mode="lines+markers",
                line=dict(color="yellow", width=2, dash="dash"),
                marker=dict(size=8, color="yellow")
            ))

    plot_layout = go.Layout(
        xaxis=dict(range=[0, PITCH_LENGTH], showgrid=False, zeroline=True, visible=False),
        yaxis=dict(range=[0, PITCH_WIDTH], showgrid=False, zeroline=True, visible=False),
        plot_bgcolor="green", height=500, margin=dict(l=20, r=20, t=20, b=20),
        title="."
    )
    return go.Figure(data=plot_traces, layout=plot_layout)

# -------------------------
# Dash Application Setup
# -------------------------
dash_app = dash.Dash(__name__)
dash_app.title = "France vs Argentina 2022 World Cup Interactive Dashboard"

dash_app.layout = html.Div([
    html.H1("France vs Argentina 2022 World Cup Final"),
    dcc.Interval(id='update-timer', interval=2000, n_intervals=0),
    html.Div([
        html.Div([dcc.Graph(id='field-visual')],
                 style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Player Insights"),
            html.Div(id='player-insights', children="Select hunting://Select a player to view stats.",
                     style={'border': '2px solid #333', 'padding': '15px', 'backgroundColor': '#f9f9f9',
                            'borderRadius': '10px', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ]),
    html.Div([
        html.Span("Argentina", style={'color': 'blue', 'fontSize': '20px', 'position': 'absolute', 'left': '10%', 'bottom': '5px'}),
        html.Span("France", style={'color': 'red', 'fontSize': '20px', 'position': 'absolute', 'right': '45%', 'bottom': '5px'})
    ], style={'position': 'relative', 'height': '40px'})
], style={'padding': '20px'})

# -------------------------
# Callbacks
# -------------------------
@dash_app.callback(
    Output('field-visual', 'figure'),
    [Input('update-timer', 'n_intervals'),
     Input('field-visual', 'clickData')]
)
def refresh_field(step, click_data):
    adjust_player_locations(step)
    selected_player = click_data['points'][0]['customdata'] if click_data else None
    return generate_pitch_visual(selected_player)

@dash_app.callback(
    Output('player-insights', 'children'),
    Input('field-visual', 'clickData')
)
def show_player_insights(click_data):
    if not click_data:
        return "Click a player to view stats."
    
    player_name = click_data['points'][0]['customdata']
    stats = player_metrics.get(player_name, {})
    curr_x, curr_y = player_locations[player_name]
    squad = 'Argentina' if player_name in arg_squad else 'France'
    advice, (opt_x, opt_y) = tactical_advice(player_name, squad)
    
    insights = [
        html.H4(player_name, style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P(f"Position: x={curr_x:.1f}, y={curr_y:.1f}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Games: {stats.get('games', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Scores: {stats.get('scores', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Assists: {stats.get('assists', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Passes: {stats.get('passes_completed', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Rating: {stats.get('rating', 0.0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.Div([
            html.Strong("Tactical Advice: ", style={'color': '#e74c3c'}),
            html.Span(advice, style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '5px 10px',
                                     'borderRadius': '5px', 'display': 'inline-block'})
        ], style={'marginTop': '15px', 'fontSize': '14px'}),
        html.P(f"Target Position: x={opt_x:.1f}, y={opt_y:.1f}", style={'fontSize': '14px', 'color': '#555', 'marginTop': '10px'})
    ]
    return insights

# -------------------------
# Launch Application
# -------------------------
if __name__ == '__main__':
    dash_app.run(debug=True)
class TeamPossessionAssigner:
    def __init__(self):
        self.team_possession = []

    def update(self, players, assigned_player_id):
        team = 0
        if self.team_possession:
            team = self.team_possession[-1]
        
        if assigned_player_id is not None:
            player = players.get(assigned_player_id)
            if player is not None:
                team = None
                if player is not None:
                    team = player.get("team")

        self.team_possession.append(team)       
        return team
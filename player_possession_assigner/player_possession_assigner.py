import sys
sys.path.append('../')
from utilities import get_center_bbox, measure_distance

class PlayerPossessionAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_possession_to_player(self, players, ball_bbox):
        ball_position = get_center_bbox(ball_bbox)

        # Big enough number in order for it to get overwritten on the first chance
        minimum_distance = 999999
        assigned_player = None

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player


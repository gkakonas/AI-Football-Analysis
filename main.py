from utilities import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_possession_assigner import PlayerPossessionAssigner
from team_possession_assigner import TeamPossessionAssigner

def main():
    # Read the video
    video_frames = read_video('input/08fd33_4.mp4')

    # Initialize the tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Interpolate missing ball positions
    tracks["ball"] = tracker.ball_interpolation(tracks["ball"])

    # Assign a team to each player
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign possession to player who currently holds the ball
    player_possession_assigner = PlayerPossessionAssigner()
    team_possession_assigner = TeamPossessionAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_possession_assigner.assign_possession_to_player(player_track, ball_bbox)

        if assigned_player is not None:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

        team_possession_assigner.update(player_track, assigned_player)

    # Draw the circles under the players and refs on the video frames
    output_video_frames = tracker.draw_tracking_markers(video_frames, tracks, team_possession_assigner.team_possession)

    # Save the video
    save_video(output_video_frames, 'output/08fd33_4_output.mp4')

if __name__ == '__main__':
    main()
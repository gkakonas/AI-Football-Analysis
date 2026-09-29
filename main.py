from utilities import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # Read the video
    video_frames = read_video('input/08fd33_4.mp4')

    # Initialize the tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Assign a team to each player
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw the circles under the players and refs on the video frames
    output_video_frames = tracker.draw_tracking_markers(video_frames, tracks)

    # Save the video
    save_video(output_video_frames, 'output/08fd33_4_output.mp4')

if __name__ == '__main__':
    main()
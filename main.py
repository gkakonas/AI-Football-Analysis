from utilities import read_video, save_video
from trackers import Tracker

def main():
    # Read the video
    video_frames = read_video('input/08fd33_4.mp4')

    # Initialize the tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Draw the circles under the players and refs on the video frames
    output_video_frames = tracker.draw_circles(video_frames, tracks)

    # Save the video
    save_video(output_video_frames, 'output/08fd33_4_output.mp4')

if __name__ == '__main__':
    main()
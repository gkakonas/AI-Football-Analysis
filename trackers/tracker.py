from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utilities import get_center_bbox, get_width_bbox

class Tracker:
    def __init__(self, model_path):
        self.model= YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1,)
            detections += detections_batch
        
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v: k for k, v in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player, since we don't have any goalkeeper specific stats
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = class_names_inv["player"]

        # Tracker
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if read_from_stub:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        
        return tracks

    def draw_player_circle(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center = get_center_bbox(bbox)[0]
        width = get_width_bbox(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw the rectangle with the track id below the player's circle

        rectangle_width = 40
        rectangle_height = 15
        x1_rectangle = x_center - rectangle_width // 2
        x2_rectangle = x_center + rectangle_width // 2
        y1_rectangle = (y2 - rectangle_height) + 15
        y2_rectangle = (y2 + rectangle_height) + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rectangle), int(y1_rectangle)),
                          (int(x2_rectangle), int(y2_rectangle)),
                          color,
                          cv2.FILLED)
            
            # Draw the track id text inside the rectangle

            x1_text = x1_rectangle + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rectangle + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,)


        return frame

    
    def draw_circles(self, video_frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dictionary = tracks["players"][frame_num]
            referee_dictionary = tracks["referees"][frame_num]
            ball_dictionary = tracks["ball"][frame_num]

            # Draw the circle under each player
            for track_id, player in player_dictionary.items():
                frame = self.draw_player_circle(frame, player["bbox"], (0, 165, 255), track_id)

            # Draw the circle under each referee (We do not care about the track id for referees, so we will not draw it)
            for _, referee in referee_dictionary.items():
                frame = self.draw_player_circle(frame, referee["bbox"], (255, 0, 0), None)

            output_frames.append(frame)
            
        return output_frames
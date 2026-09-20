"""
Systems to Detect and Alarm Probability of Accidents in Real Time

Description:
    Real-time vision-based accident probability detection system.
    Monitors designated hazard zones (e.g., swimming pool perimeter, transit zones),
    calculates human velocity and sensory latency compensation, predicts trajectory 
    intercept thresholds, and triggers multi-tier audiovisual and push notifications.
"""


import cv2
import numpy as np
import time
import math
import threading
from ultralytics import YOLO
from plyer import notification
import pygame

# -------------------------------------------------------------------------
# SYSTEM CONFIGURATION & CALIBRATION
# -------------------------------------------------------------------------
CAMERA_INDEX = 0             # Camera stream (webcam, RTSP, or recorded video)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Calibrated sensory latency constant (pipeline processing + transmission latency)
SENSORY_LATENCY = 0.35       # seconds (~350 ms forward projection window)

# Risk boundary thresholds (in frame pixels)
CRITICAL_DISTANCE_PX = 70    # Immediate perimeter breach
WARNING_DISTANCE_PX = 150    # Pre-impact velocity vector warning

# Initialize Audio Alert Engine
pygame.mixer.init()

def trigger_audio_alarm():
    """Generates an immediate synthesized acoustic pulse alert."""
    try:
        sample_rate = 44100
        duration = 0.30
        freq = 920  # Audible alert frequency (Hz)
        n_samples = int(sample_rate * duration)
        buf = np.sin(2 * np.pi * np.arange(n_samples) * freq / sample_rate)
        stereo_buf = np.repeat(buf.reshape(-1, 1), 2, axis=1)
        audio_array = (stereo_buf * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(audio_array)
        sound.play()
    except Exception as e:
        print(f"[ALARM ERROR] {e}")

def trigger_desktop_notification(title, message):
    """Dispatches an asynchronous system alert."""
    def _notify():
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="Patent-Intervention-System",
                timeout=2
            )
        except Exception as e:
            print(f"[NOTIFICATION ERROR] {e}")
    threading.Thread(target=_notify, daemon=True).start()


# -------------------------------------------------------------------------
# KINEMATIC LATENCY & TRAJECTORY PREDICTOR
# -------------------------------------------------------------------------
class KinematicTrajectoryPredictor:
    """
    Computes real-time velocity vectors and projects ground-contact trajectory
    across the calibrated sensory latency window (P_proj = P_now + V * delta_t).
    """
    def __init__(self, latency=SENSORY_LATENCY):
        self.latency = latency
        self.last_positions = {}   # Track history: {track_id: (x, y, timestamp)}
        self.velocities = {}       # Smoothed velocity: {track_id: (vx, vy)}

    def update_and_project(self, track_id, current_pos, timestamp):
        proj_pos = current_pos
        velocity_mag = 0.0

        if track_id in self.last_positions:
            last_pos, last_t = self.last_positions[track_id]
            dt = timestamp - last_t

            if dt > 1e-4:
                raw_vx = (current_pos[0] - last_pos[0]) / dt
                raw_vy = (current_pos[1] - last_pos[1]) / dt

                # Exponential smoothing filter
                alpha = 0.70
                if track_id in self.velocities:
                    vx = alpha * raw_vx + (1 - alpha) * self.velocities[track_id][0]
                    vy = alpha * raw_vy + (1 - alpha) * self.velocities[track_id][1]
                else:
                    vx, vy = raw_vx, raw_vy

                self.velocities[track_id] = (vx, vy)
                velocity_mag = math.sqrt(vx**2 + vy**2)

                # Project forward over the sensory latency delay window
                proj_x = int(current_pos[0] + vx * self.latency)
                proj_y = int(current_pos[1] + vy * self.latency)
                proj_pos = (proj_x, proj_y)

        self.last_positions[track_id] = (current_pos, timestamp)
        return proj_pos, velocity_mag


def point_to_polygon_distance(point, polygon):
    """Computes signed distance from point to hazard perimeter."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), measureDist=True)


# -------------------------------------------------------------------------
# MAIN INFERENCE PIPELINE
# -------------------------------------------------------------------------
def main():
    print("[SYSTEM] Initializing YOLO Neural Network & Tracking Engine...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    predictor = KinematicTrajectoryPredictor(latency=SENSORY_LATENCY)

    # Designated hazard perimeter polygon (e.g., pool perimeter relative to 1280x720 frame)
    hazard_zone = np.array([
        [400, 320],
        [880, 320],
        [1020, 680],
        [260, 680]
    ], np.int32).reshape((-1, 1, 2))

    last_alarm_time = 0.0
    alarm_cooldown = 1.8  # seconds

    print("==================================================================")
    print("Artina Belivan - Real-Time Accident Probability Model")
    print("Running high-throughput tracker. Press 'q' to exit.")
    print("==================================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        overlay = frame.copy()

        # Render hazard boundary polygon
        cv2.polylines(frame, [hazard_zone], isClosed=True, color=(255, 140, 0), thickness=3)
        cv2.fillPoly(overlay, [hazard_zone], color=(255, 140, 0))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # Execute YOLO tracking: class 0 is 'person'
        results = model.track(frame, classes=[0], persist=True, verbose=False)

        system_status = "SAFE: NO PERIMETER THREAT"
        status_color = (40, 180, 40)

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Ground-plane contact point (feet centroid)
                ground_x = int((x1 + x2) / 2.0)
                ground_y = int(y2)
                current_pos = (ground_x, ground_y)

                # Kinematic latency projection
                projected_pos, speed = predictor.update_and_project(track_id, current_pos, current_time)

                # Distance calculation
                dist_current = point_to_polygon_distance(current_pos, hazard_zone)
                dist_projected = point_to_polygon_distance(projected_pos, hazard_zone)
                effective_dist = min(dist_current, dist_projected)

                # Risk classification
                if dist_current <= 0 or dist_projected <= 0 or effective_dist <= CRITICAL_DISTANCE_PX:
                    system_status = f"CRITICAL HAZARD INTERVENTION (ID: {track_id})"
                    status_color = (0, 0, 255)
                    box_color = (0, 0, 255)

                    if (current_time - last_alarm_time) > alarm_cooldown:
                        trigger_audio_alarm()
                        trigger_desktop_notification(
                            "CRITICAL ACCIDENT RISK DETECTED",
                            f"Subject ID {track_id} breached projected safety threshold."
                        )
                        last_alarm_time = current_time

                elif effective_dist <= WARNING_DISTANCE_PX:
                    system_status = f"WARNING: APPROACH VECTOR DETECTED (ID: {track_id})"
                    status_color = (0, 140, 255)
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 255, 0)

                # Visual overlays
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.circle(frame, current_pos, 5, (0, 255, 255), -1)
                cv2.circle(frame, projected_pos, 6, (0, 0, 255), -1)
                cv2.arrowedLine(frame, current_pos, projected_pos, (0, 0, 255), 2, tipLength=0.25)

                cv2.putText(frame, f"ID:{track_id} | V:{int(speed)}px/s | D:{int(dist_current)}px",
                            (int(x1), max(25, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

        # Telemetry Banner
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 48), status_color, -1)
        cv2.putText(frame, f"STATUS: {system_status}", (20, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)

        cv2.imshow("Artina Belivan - Real-Time Accident Probability Intervention Model", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

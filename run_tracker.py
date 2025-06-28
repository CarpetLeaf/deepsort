import os
import time
import cv2

from detectors.yolov5_detector import YOLOv5Detector
from detectors.yolov8_detector import YOLOv8Detector
from detectors.ssd_detector import SSDDetector

from reid_models.torchreid_model import TorchReID
from reid_models.original_deepsort import OriginalReID
from reid_models.resnet18_reid import ResNet18ReID

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

from utils.draw_tracks import draw_tracks
from utils.parser import load_config_file
from utils.save_mot_txt import write_mot_format


def create_detector(cfg, device):
    detector_type = cfg.name.lower()
    if detector_type == "yolov5":
        return YOLOv5Detector(cfg.model_path, cfg.confidence, device=device)
    if detector_type == "yolov8":
        return YOLOv8Detector(cfg.model_path, cfg.confidence, device=device)
    if detector_type == "ssd":
        return SSDDetector(cfg.confidence, device=device)
    raise ValueError(f"Unsupported detector type: {detector_type}")


def create_reid_model(cfg, device):
    reid_type = cfg.name.lower()
    if reid_type == "torchreid":
        return TorchReID(cfg.model_name, device=device)
    if reid_type == "original":
        return OriginalReID(cfg.model_path, device=device)
    if reid_type == "resnet18":
        return ResNet18ReID(cfg.model_path, device=device)
    raise ValueError(f"Unsupported ReID model type: {reid_type}")


def run_tracking(video_path, video_out_path, txt_out_path, display=True, config_path="configs/deepsort_config.yaml"):
    config = load_config_file(config_path)
    device = "cpu"
    print(f"[INFO] Using device: {device}")

    detector = create_detector(config.detector, device)
    reid_model = create_reid_model(config.reid, device)

    metric = NearestNeighborDistanceMetric(
        metric=config.tracker.metric,
        matching_threshold=config.tracker.max_iou_distance,
        budget=config.tracker.nn_budget
    )
    tracker = Tracker(metric)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(txt_out_path), exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Failed to create output video: {video_out_path}")
        return

    frame_idx = 1
    track_log = []
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        detections_raw = detector.detect(frame)
        features = []
        for (x, y, w, h, conf) in detections_raw:
            crop = frame[int(y):int(y + h), int(x):int(x + w)]
            features.append(reid_model.extract_features(crop))

        detections = [
            Detection([x, y, w, h], conf, feat)
            for (x, y, w, h, conf), feat in zip(detections_raw, features)
        ]

        # Tracking
        tracker.predict()
        tracker.update(detections)

        # Drawing
        frame = draw_tracks(frame, tracker.tracks)

        elapsed = time.time() - start
        curr_fps = frame_idx / elapsed
        cv2.putText(frame, f"FPS: {curr_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                bbox = track.to_tlwh()
                track_log.append((frame_idx, track.track_id, bbox, 1.0))

        frame_idx += 1

        if display:
            cv2.imshow("DeepSORT Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    write_mot_format(txt_out_path, track_log)
    print(f"[INFO] Tracking completed. Results saved to: {txt_out_path}")


def main():
    cfg = load_config_file("configs/deepsort_config.yaml")
    video_input = cfg.input.video_path
    video_output = cfg.input.output_video
    seq_name = os.path.splitext(os.path.basename(video_input))[0]
    txt_output = f"outputs/mot_challenge/{seq_name}/det.txt"

    run_tracking(video_input, video_output, txt_output, cfg.input.display)


if __name__ == "__main__":
    main()

# save_mot_txt.py
import os

def write_mot_format(filepath, tracks):
    """Сохраняет треки в формате MOTChallenge."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as file:
        for frame_idx, object_id, bbox, confidence in tracks:
            x, y, w, h = bbox
            line = f"{frame_idx},{object_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{confidence:.2f},-1,-1\n"
            file.write(line)

if __name__ == "__main__":
    # Пример данных для теста
    example_tracks = [
        (1, 1, [100, 50, 80, 200], 1.0),
        (1, 2, [400, 60, 90, 190], 1.0),
        (2, 1, [102, 53, 80, 200], 1.0),
    ]

    output_file = "trackers/mot_challenge/MOT16-01/ours/det.txt"
    write_mot_format(output_file, example_tracks)


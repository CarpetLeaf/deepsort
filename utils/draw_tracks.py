import cv2

def draw_tracks(frame, tracks):
    for track in tracks:
        x, y, w, h = track.to_tlwh()
        tid = track.track_id
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, str(tid), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame
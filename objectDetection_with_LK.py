import cv2
import numpy as np
import os

frame_dir = "/content/frames"
ann_dir = "/content/annotation"
output_video = "output_tracking_v2.mp4"

frames = sorted(os.listdir(frame_dir), key=lambda x: int(x.split('.')[0]))

sample_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w = sample_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20, (w, h))

lk_params = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

prev_gray = None
p_prev = None
bbox = None          # (x, y, w, h) of object
obj_mask_shape = None  # original mask for shape reference

def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (x1, y1, x2 - x1, y2 - y1)

def sample_points_in_mask(mask, n=300):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = np.random.choice(len(xs), min(n, len(xs)), replace=False)
    pts = np.vstack((xs[idx], ys[idx])).T.astype(np.float32)
    return pts.reshape(-1, 1, 2)

def draw_filled_contour(frame, points, color=(0, 255, 0), alpha=0.4):
    if len(points) < 3:
        return frame
    overlay = frame.copy()
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillPoly(overlay, [hull], color)
    cv2.polylines(overlay, [hull], True, (0, 200, 0), 2)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

# =========================
# MAIN LOOP
# =========================
for f in frames:
    frame_path = os.path.join(frame_dir, f)
    frame = cv2.imread(frame_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp = f.split('.')[0]

    mask_name = f"100_{timestamp}_mask.jpg"
    mask_path = os.path.join(ann_dir, mask_name)

    
    # CASE 1: ANNOTATION EXISTS → RESET
    if os.path.exists(mask_path):
        print(f"[RESET] {mask_name}")

        mask = cv2.imread(mask_path, 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        bbox = get_bbox_from_mask(mask)
        p_prev = sample_points_in_mask(mask, n=400)
        obj_mask_shape = mask.copy()

        if p_prev is None:
            prev_gray = gray
            continue

        # Draw on annotated frame
        output = draw_filled_contour(frame, p_prev.reshape(-1, 2), color=(0, 255, 0))
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(output, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(output, "OBJECT", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

  
    # CASE 2: TRACK WITH LK
    elif prev_gray is not None and p_prev is not None:

        p_next, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p_prev, None, **lk_params
        )

        good_new = p_next[st == 1]
        good_old = p_prev[st == 1]

        if len(good_new) < 10:
            print(f"[WARNING] Only {len(good_new)} points — lost tracking")
            prev_gray = gray
            continue

        # Estimate motion: median shift of all tracked points
        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])

        # Move bbox by the median shift
        if bbox:
            x, y, bw, bh = bbox
            x = int(x + dx)
            y = int(y + dy)
            x = max(0, min(x, w - bw))
            y = max(0, min(y, h - bh))
            bbox = (x, y, bw, bh)

        # Draw filled convex hull around tracked points
        output = draw_filled_contour(frame, good_new, color=(0, 120, 255))

        # Draw bounding box
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(output, (x, y), (x+bw, y+bh), (0, 200, 255), 2)
            cv2.putText(output, "TRACKING", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        p_prev = good_new.reshape(-1, 1, 2)

    else:
        prev_gray = gray
        continue

    # SAVE FRAME
    out.write(output)
    prev_gray = gray.copy()

out.release()
cv2.destroyAllWindows()
print("✅ Saved:", output_video)
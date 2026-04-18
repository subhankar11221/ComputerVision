import cv2
import numpy as np
import os

frame_dir = "/content/frames"
ann_dir = "/content/annotations"
output_video = "output_tracking.mp4"

frames = sorted(os.listdir(frame_dir), key=lambda x: int(x.split('.')[0]))

sample_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w = sample_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20, (w, h))

lk_params = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

prev_gray   = None
p_prev      = None
bbox        = None
orig_mask   = None  
tracking_lost = False

def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min()),
            int(ys.max() - ys.min()))

def sample_points_in_mask(mask, n=500):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = np.random.choice(len(xs), min(n, len(xs)), replace=False)
    pts = np.vstack((xs[idx], ys[idx])).T.astype(np.float32)
    return pts.reshape(-1, 1, 2)

def warp_mask_to_position(orig_mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(orig_mask, M, (orig_mask.shape[1], orig_mask.shape[0]))

def draw_object_overlay(frame, mask, color_bgr, label, alpha=0.45):
    overlay = frame.copy()

    # Filled color inside mask
    colored = np.zeros_like(frame)
    colored[mask > 0] = color_bgr
    cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)

    # Contour outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 2)

    # Label near top of object
    ys, xs = np.where(mask > 0)
    if len(ys):
        tx, ty = int(xs.mean()), max(int(ys.min()) - 10, 15)
        cv2.putText(overlay, label, (tx - 40, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color_bgr, 2, cv2.LINE_AA)

    return overlay

def draw_tracking_lost(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), 6)
    cv2.putText(overlay, "⚠ TRACKING LOST", (w//2 - 160, h//2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return overlay


total_dx = 0.0
total_dy = 0.0


for f in frames:
    frame_path = os.path.join(frame_dir, f)
    frame      = cv2.imread(frame_path)
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp  = f.split('.')[0]

    mask_name = f"100_{timestamp}_mask.jpg"
    mask_path = os.path.join(ann_dir, mask_name)


    if os.path.exists(mask_path):
        print(f"[RESET] {mask_name}")

        mask = cv2.imread(mask_path, 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        orig_mask   = mask.copy()
        bbox        = get_bbox_from_mask(mask)
        p_prev      = sample_points_in_mask(mask, n=500)
        total_dx    = 0.0
        total_dy    = 0.0
        tracking_lost = False

        if p_prev is None:
            prev_gray = gray
            continue

        output = draw_object_overlay(frame, mask,
                                     color_bgr=(0, 220, 0),
                                     label="OBJECT")

    
    elif prev_gray is not None and p_prev is not None and not tracking_lost:

        p_next, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p_prev, None, **lk_params
        )

        if p_next is None or st is None:
            tracking_lost = True
        else:
            good_new = p_next[st == 1]
            good_old = p_prev[st == 1]

            min_points  = max(15, int(0.1 * len(p_prev)))   
            point_ratio = len(good_new) / max(len(p_prev), 1)

            if len(good_new) < min_points or point_ratio < 0.08:
                tracking_lost = True
            else:
                
                dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
                dy = float(np.median(good_new[:, 1] - good_old[:, 1]))

            
                if abs(dx) > w * 0.20 or abs(dy) > h * 0.20:
                    tracking_lost = True
                else:
                    total_dx += dx
                    total_dy += dy

                   
                    curr_mask = warp_mask_to_position(orig_mask,
                                                      int(round(total_dx)),
                                                      int(round(total_dy)))

                    if curr_mask.sum() < 100:
                        tracking_lost = True
                    else:
                        output = draw_object_overlay(frame, curr_mask,
                                                     color_bgr=(0, 140, 255),
                                                     label="TRACKING LOST")
                        p_prev = good_new.reshape(-1, 1, 2)

        if tracking_lost:
            print(f"[LOST] Tracking lost at frame {timestamp}")
            output = draw_tracking_lost(frame)


    elif tracking_lost:
        output = draw_tracking_lost(frame)

    else:
        prev_gray = gray
        continue

    out.write(output)
    prev_gray = gray.copy()

out.release()
cv2.destroyAllWindows()
print("✅ Saved:", output_video)
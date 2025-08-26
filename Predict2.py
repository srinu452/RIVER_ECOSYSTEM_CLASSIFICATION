import cv2  # OpenCV for image and video processing
import numpy as np  # Numerical operations and array manipulations
import pandas as pd  # For storing results in tabular format
from shapely.geometry import Polygon, Point  # To calculate geometric distances
from shapely.ops import nearest_points  # To find nearest points between polygons
from ultralytics import YOLO  # YOLOv8 for object detection and segmentation

# Configuration settings for video, model, and thresholds
VIDEO_PATH     = 'clip_000_3840x2160.mp4'  # Input video path
OUTPUT_VIDEO   = 'annotated_outputv7.mp4'  # Output annotated video file
OUTPUT_CSV     = 'tracked_video_metrics_v6.csv'  # Output CSV log
MODEL_PATH     = 'runs/segment/yolo9class-finetuned2/weights/best.pt'  # Path to trained YOLOv8 segmentation model
SCALE_M_PER_PX = 0.05  # Scale factor: meters per pixel
CONF_THRESH    = 0.25  # Confidence threshold for YOLO detections
IOU_THRESH     = 0.45  # IOU threshold for tracker
TRACKER_CFG    = 'bytetrack.yaml'  # ByteTrack configuration
DRONE_SPEED_M_S= 2  # Drone speed in meters/second

# Color mapping for each class
COLORS = {
    'Building': (139, 0, 0),
    'bridge': (0, 165, 255), 
    'Natural-drinage': (255, 255, 255),
    'Manmade-drinage': (255, 204, 229),
    'vegitation inside the river': (144, 238, 255)
}

# Convert binary mask to polygon using OpenCV + Shapely
def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if largest.shape[0] < 3:
        return None
    return Polygon(largest.squeeze())

# Load model and get video properties
model = YOLO(MODEL_PATH).to('cuda')
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

rows = []
frame_idx = 0
seen_ids = set()

# Track summary metrics
summary_totals = {
    'Building': {},
    'bridge': 0,
    'Natural-drinage': {'count': 0},
    'Manmade-drinage': {'count': 0}
}
building_counter = 0

# Run YOLO tracking inference
streamed = model.track(
    source=VIDEO_PATH, conf=CONF_THRESH, iou=IOU_THRESH,
    tracker=TRACKER_CFG, stream=True, show=False, save=False
)

# Process each frame
for res in streamed:
    time_s = round(frame_idx / fps, 3)
    distance_from_start_m = round(time_s * DRONE_SPEED_M_S, 2)

    if res.masks is None or res.boxes.id is None:
        frame_idx += 1
        continue

    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    ids = res.boxes.id.cpu().numpy().astype(int)
    boxes = res.boxes.xyxy.cpu().numpy()
    masks = res.masks.data.cpu().numpy()

    vis = res.orig_img.copy()
    overlay = vis.copy()
    mask_overlay = np.zeros_like(vis)

    # Extract river polygons
    river_polygons = []
    for i, cls_id in enumerate(cls_ids):
        cls_name = model.names[cls_id]
        if cls_name == 'river':
            sm = masks[i].astype(np.uint8)
            full = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)
            poly = mask_to_polygon(full)
            if poly:
                river_polygons.append(poly)

    # Process detected objects
    for i, obj_id in enumerate(ids):
        cls_name = model.names[cls_ids[i]]
        x1, y1, x2, y2 = boxes[i].astype(int)
        w, h = x2 - x1, y2 - y1

        sm = masks[i].astype(np.uint8)
        full = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_area_px = np.count_nonzero(full)
        area_m2 = round(mask_area_px * (SCALE_M_PER_PX ** 2), 2)

        center_distance_m = None

        if cls_name not in ['bridge', 'vegitation inside the river', 'river']:
            if river_polygons:
                if cls_name in ['Natural-drinage', 'Manmade-drinage']:
                    obj_poly = mask_to_polygon(full)
                    if obj_poly:
                        min_poly = min(river_polygons, key=lambda r: obj_poly.distance(r))
                        p1, p2 = nearest_points(obj_poly, min_poly)
                        dist_px = p1.distance(p2)
                        center_distance_m = round(dist_px * SCALE_M_PER_PX, 2)
                        cv2.line(overlay, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 255), 2)
                        mid = ((int(p1.x + p2.x) // 2), (int(p1.y + p2.y) // 2))
                        cv2.putText(overlay, f"{center_distance_m} m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    obj_center = Point((x1 + x2) // 2, (y1 + y2) // 2)
                    min_poly = min(river_polygons, key=lambda r: r.distance(obj_center))
                    p1, p2 = nearest_points(obj_center, min_poly)
                    dist_px = p1.distance(p2)
                    center_distance_m = round(dist_px * SCALE_M_PER_PX, 2)
                    cv2.line(overlay, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 255), 2)
                    mid = ((int(p1.x + p2.x) // 2), (int(p1.y + p2.y) // 2))
                    cv2.putText(overlay, f"{center_distance_m} m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if obj_id not in seen_ids:
            seen_ids.add(obj_id)
            if cls_name == 'bridge':
                summary_totals['bridge'] += 1
            elif cls_name == 'Building':
                building_counter += 1
                summary_totals['Building'][obj_id] = building_counter
            elif cls_name in ['Natural-drinage', 'Manmade-drinage']:
                # summary_totals[cls_name]['area'] += area_m2
                summary_totals[cls_name]['count'] += 1

            rows.append({
                'id': obj_id,
                'class': cls_name,
                'frame': frame_idx,
                'time_s': time_s,
                'mask_area_m2': area_m2,
                'bridge_length_m': round(max(w, h) * SCALE_M_PER_PX, 2) if cls_name == 'bridge' else None,
                'dist_from_riverbank_m': center_distance_m,
                'dist_from_starting': distance_from_start_m,
                'summary_Building_count': len(summary_totals['Building']) if cls_name == 'Building' else None,
                'summary_Natural_drainage_count': summary_totals['Natural-drinage']['count'] if cls_name == 'Natural-drinage' else None,
                'summary_Manmade_drainage_count': summary_totals['Manmade-drinage']['count'] if cls_name == 'Manmade-drinage' else None,
                'summary_bridge_count': summary_totals['bridge'] if cls_name == 'bridge' else None
            })

        color = COLORS.get(cls_name, (255, 255, 255))
        label_txt = f"ID:{obj_id} {cls_name}"

        if cls_name == 'Building':
            label_txt += f" -{summary_totals['Building'].get(obj_id)}"
        if cls_name in ['Natural-drinage', 'Manmade-drinage']:
            contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, color, thickness=cv2.FILLED)
            label_txt += f" | Area: {area_m2} m²"
        elif cls_name == 'vegitation inside the river':
            label_txt += f" | Area: {area_m2} m²"
        elif cls_name == 'bridge':
            label_txt += f" | Length: {round(max(w, h) * SCALE_M_PER_PX, 2)} m"

        if cls_name not in ['Natural-drinage','Manmade-drinage','river']:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
        (text_w, text_h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        cv2.putText(overlay, label_txt, (center_x - text_w // 2, center_y + text_h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    vis = cv2.addWeighted(mask_overlay, 0.4, overlay, 1.0, 0)
    top_summary = (
        f"Buildings: {len(summary_totals['Building'])} | "
        f"Bridges: {summary_totals['bridge']} | "
        f"Natural-drainage: {summary_totals['Natural-drinage']['count']} | "
        f"Manmade-drainage: {summary_totals['Manmade-drinage']['count']}"
    )
    cv2.rectangle(vis, (0, 0), (W, 60), (0, 0, 128), -1)
    cv2.putText(vis, top_summary, (10, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
    writer.write(vis)
    frame_idx += 1

writer.release()
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print("✅ Video saved to:", OUTPUT_VIDEO)
print("✅ CSV saved to:", OUTPUT_CSV)

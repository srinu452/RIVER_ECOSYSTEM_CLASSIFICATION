import cv2
from ultralytics import YOLO

# Path to your trained model
MODEL_PATH   ='YOLO3/YOLO4/runs/segment/yolo9class-seg5/weights/best.pt'
# Path to input video (or 0 for webcam)
VIDEO_SOURCE = "DJI_SRT_DATA/DJI_0030.MOV"

def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Couldn't open video source {VIDEO_SOURCE}")
        return

    # VideoWriter setup
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    out   = cv2.VideoWriter("annotated_outputv6.mp4", fourcc, fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection & draw boxes
        results   = model(frame)[0]
        annotated = results.plot()

        # Write annotated frame
        out.write(annotated)

    cap.release()
    out.release()
    print("✅ Done! Saved annotated video as annotated_output.mp4")

if __name__ == "__main__":
    main()

# ___________________________________________________________________________________

##### Metric Calculation

# import cv2
# import numpy as np
# import pandas as pd
# from ultralytics import YOLO

# # CONFIGURATION SETTINGS
# VIDEO_PATH     = 'video_work/clip_06.mp4'
# OUTPUT_VIDEO   = 'annotated_output6.mp4'
# OUTPUT_CSV     = 'tracked_video_metrics4.csv'
# MODEL_PATH     = 'runs/segment/my5class2/weights/best.pt'
# SCALE_M_PER_PX = 0.05
# CONF_THRESH    = 0.25
# IOU_THRESH     = 0.45
# TRACKER_CFG    = 'bytetrack.yaml'
# DRONE_SPEED_M_S= 2

# # COLOR MAPPING for each class (BGR format)
# COLORS = {
#     'Building': (139, 0, 0),
#     'bridge': (0, 165, 255),  # Darker orange
#     'Natural-drinage': (255, 255, 255),  # whiter 
#     'Manmade-drinage': (255, 204, 229),  # Lighter pink
#     'vegitation inside the river': (144, 238, 255)  # Lighter blue
# }

# # Initialize model and video
# model = YOLO(MODEL_PATH).to('cuda')
# cap = cv2.VideoCapture(VIDEO_PATH)
# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.release()

# # Run tracking inference with YOLOv8
# streamed = model.track(
#     source=VIDEO_PATH, conf=CONF_THRESH, iou=IOU_THRESH,
#     tracker=TRACKER_CFG, stream=True, show=False, save=False
# )

# # Prepare video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

# # Tracking variables
# seen_ids = set()
# rows = []
# frame_idx = 0

# # Global totals
# summary_totals = {
#     'Building': {},
#     'bridge': 0,
#     'Natural-drinage': {'area': 0.0, 'count': 0},
#     'Manmade-drinage': {'area': 0.0, 'count': 0},
#     'vegitation inside the river': 0.0
# }

# building_counter = 0

# for res in streamed:
#     time_s = round(frame_idx / fps, 3)
#     distance_from_start_m = round(time_s * DRONE_SPEED_M_S, 2)
#     if res.masks is None or res.boxes.id is None:
#         frame_idx += 1
#         continue

#     cls_ids = res.boxes.cls.cpu().numpy().astype(int)
#     ids     = res.boxes.id.cpu().numpy().astype(int)
#     boxes   = res.boxes.xyxy.cpu().numpy()
#     masks   = res.masks.data.cpu().numpy()

#     # Original frame from video
#     vis = res.orig_img.copy()
#     # Create a duplicate for drawing on
#     overlay = vis.copy()
#     #for masking 
#     mask_overlay = np.zeros_like(vis)

#     frame_center = (W // 2, H // 2) # TO COLLECT THE FRAME CENTER VALUE

#     for i, obj_id in enumerate(ids):  #ACCORDING TO THE INDEX , IT WILL GIVE THE OBJECT ID
#         cls_name = model.names[int(cls_ids[i])]   #FROM THAT MODEL IT WILL GIVE NAMES 
#         x1, y1, x2, y2 = boxes[i].astype(int)  #FROM   THERE WHATEVER THE BOXES THAT WE HAVE  
#         w, h= x2 - x1, y2 - y1   # BASED ON THIS WE CAN COLLECT THE WIDTH AND LENGTH ACCORDING TO THE  BOX CORDINATES

#         sm = masks[i].astype(np.uint8)   # IT WILL CONSIDERED THE MASK IMAGES  AND Converts the mask values from float (0.0, 1.0) to uint8 (0 and 1), making it compatible with OpenCV functions AND You now have a binary image: white (1) for object, black (0) for background
#         full = cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST)  # esized to a fixed network size  AND  back to the original frame resolution (W, H).AND  full-sized binary mask that matches the video frame dimensions.
#         mask_area_px = np.count_nonzero(full)  # Counts how many pixels in the mask are non-zero (i.e., how many pixels are part of the object).
#         area_m2 = round(mask_area_px * (SCALE_M_PER_PX ** 2), 2)

#         length_m = None
#         left = W//2.8
#         right = W//1.6
#         object_width = int((x1+x2)//2)
#         X = 0
#         if abs(object_width-left)<abs(object_width-right):
#             X = left
#         else :
#             X = right 
#         frame_center = (int(X),(y1 + y2) // 2)

#         object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#         dx, dy = object_center[0] - frame_center[0], object_center[1] - frame_center[1]
#         dist_px = np.hypot(dx, dy)
#         center_distance_m = round(dist_px * SCALE_M_PER_PX, 2)
        
#         if cls_name not in ['bridge', 'vegitation inside the river','river']:
#              # draw line
#             cv2.line(overlay, frame_center, object_center, (0, 255, 255), 2)

#             # put distance text
#             mid_point = ((frame_center[0] + object_center[0]) // 2, (frame_center[1] + object_center[1]) // 2)
#             dist_text = f"{center_distance_m} m"
#             cv2.putText(overlay, dist_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


#         first_time = obj_id not in seen_ids
#         if first_time:
#             seen_ids.add(obj_id)

#             if cls_name == 'bridge':
#                 length_px = max(w, h)
#                 length_m = round(length_px * SCALE_M_PER_PX, 2)
#                 summary_totals['bridge'] += 1

#             elif cls_name == 'Building':
#                 building_counter += 1
#                 summary_totals['Building'][obj_id] = building_counter

#             elif cls_name in ['Natural-drinage', 'Manmade-drinage']:
#                 summary_totals[cls_name]['area'] += area_m2
#                 summary_totals[cls_name]['count'] += 1

#             elif cls_name == 'vegitation inside the river':
#                 summary_totals[cls_name] += area_m2

#             rows.append({
#                 'id': obj_id,
#                 'class': cls_name,
#                 'frame': frame_idx,
#                 'time_s': time_s,
#                 'mask_area_m2': area_m2,
#                 'bridge_length_m': length_m,
#                 'dist_from_center_m': center_distance_m,
#                 'dist_from_starting':distance_from_start_m
#             })

#         color = COLORS.get(cls_name, (255, 255, 255))
#         label_txt = f"ID:{obj_id} {cls_name}"

#         if cls_name == 'Building':
#             building_no = summary_totals['Building'].get(obj_id)
#             if building_no is not None:
#                 label_txt += f" -{building_no}"
#                 # cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
        

#         if cls_name in ['Natural-drinage', 'Manmade-drinage']:
#             contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(mask_overlay, contours, -1, COLORS.get(cls_name, (255, 255, 255)), thickness=cv2.FILLED)
#             label_txt += f" | Area: {area_m2} m²"
#         else:
#             if cls_name == 'bridge':
#                 length_m = round(max(w, h) * SCALE_M_PER_PX, 2)
#                 label_txt += f" | Length: {length_m} m"
#             elif cls_name == 'vegitation inside the river':
#                 label_txt += f" | Area: {area_m2} m²"
#            # label_txt += f" | Dist from Center: {center_distance_m} m"
#                 # cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
        
#         # Labels and text css
#         if cls_name not in ["river","Natural-drinage"]:
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
#         (text_w, text_h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
#         center_x = x1 + w // 2
#         center_y = y1 + h // 2
#         # text_color = (0, 0, 0)  # Dark gray for better readability
#         cv2.putText(overlay, label_txt, (center_x - text_w // 2, center_y + text_h // 2),
#                     cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        
#     # Blend mask with lower transparency
#     alpha = 0.40
#     vis = cv2.addWeighted(mask_overlay, alpha, overlay, 1.0, 0)

#     # Summary text at top
#     top_summary = (
#         f"Buildings: {len(summary_totals['Building'])} | "
#         f"Bridges: {summary_totals['bridge']} | "
#         f"Natural-drainage: {summary_totals['Natural-drinage']['count']} | "
#         f"Manmade-drainage: {summary_totals['Manmade-drinage']['count']}"
#     )
#     # Summary css
#     cv2.rectangle(vis, (0, 0), (W, 60), (0, 0, 128), -1)
#     cv2.putText(vis, top_summary, (10, 50),
#                 cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)

#     writer.write(vis)
#     frame_idx += 1

# writer.release()
# df = pd.DataFrame(rows)
# df.to_csv(OUTPUT_CSV, index=False)
# print("✅ Video saved to:", OUTPUT_VIDEO)
# print("✅ CSV saved to:", OUTPUT_CSV)


# # # ________________________________________________________________________________________________

# # # YOLOv8 Segmentation Inference with Area, Length & Count Summary Overlay
# # # YOLOv8 Segmentation Inference with Area, Length & Count Summary Overlay

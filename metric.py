
# Distance Calculator via longitude and latitude

# from math import radians, sin, cos, sqrt, atan2

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # Earth radius in meters

#     φ1, φ2 = radians(lat1), radians(lat2)
#     Δφ = radians(lat2 - lat1)
#     Δλ = radians(lon2 - lon1)

#     a = sin(Δφ/2)**2 + cos(φ1) * cos(φ2) * sin(Δλ/2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))

#     return R * c

# # Example usage
# d = haversine(25.34105757305762, 83.00902436380922,
#               25.341317022650504, 83.00912643426842)

# print(f"Distance: {d:.2f} meters")

# _____________________________________________________________________

# Vedio to Images conversion code 


# import cv2
# import os

# def extract_every_30th_frame(video_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"🎥 Detected FPS: {fps}")

#     frame_count = 0
#     saved_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % 120 == 0:  # Save every 30th frame
#             filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
#             cv2.imwrite(filename, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()
#     print(f"✅ Done. Saved {saved_count} frames from {frame_count} total frames.")

# # Example usage:
# extract_every_30th_frame("DJI_SRT_DATA/DJI_0007.MOV","video_chunk/images")

# ______________________________________________________________________________________________

#         # Labels miss matching settings to correct labels code:
# import os

# # Old-to-new class ID mapping
# # If your labels were using old ordering like this:
# # ['Building', 'Manmade-drinage', 'Natural-drinage', 'bridge', 'obstacles', 'potholes', 'river', 'silts', 'vegitation inside the river']
# # Then the index mapping to new:
# # ['Building', 'Manmade-drinage', 'Natural-drinage', 'bridge', 'river', 'vegitation inside the river', 'obstacles', 'potholes', 'silts']
# id_mapping = {
#     0: 0,  # Building
#     1: 1,  # Manmade-drinage
#     2: 2,  # Natural-drinage
#     3: 3,  # bridge
#     4: 6,  # obstacles → new id 6
#     5: 7,  # potholes → new id 7
#     6: 4,  # river → new id 4
#     7: 8,  # silts → new id 8
#     8: 5,  # vegitation inside river → new id 5
# }

# 📁 Change this to your dataset labels folder path
# LABEL_DIR = 'YOLO2/test/labels'  # e.g., '../train/labels'

# def remap_label_file(file_path):
#     new_lines = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             old_id = int(parts[0])
#             if old_id in id_mapping:
#                 parts[0] = str(id_mapping[old_id])
#                 new_lines.append(' '.join(parts))
#             else:
#                 print(f"Skipping invalid class ID {old_id} in {file_path}")

#     with open(file_path, 'w') as f:
#         f.write('\n'.join(new_lines))

# def process_all_labels(label_dir):
#     count = 0
#     for root, _, files in os.walk(label_dir):
#         for file in files:
#             if file.endswith('.txt'):
#                 filepath = os.path.join(root, file)
#                 remap_label_file(filepath)
#                 count += 1
#     print(f"✅ Updated {count} label files.")

# 🔧 Run it
# process_all_labels(LABEL_DIR)
# _________________________________________________________________________________________________________

# How many labels that we  have in our txt files checking code?
# import os
# from collections import Counter

# label_dir = 'YOLO2/train/labels'  # or 'valid', 'test'
# counter = Counter()

# for file in os.listdir(label_dir):
#     if file.endswith('.txt'):
#         with open(os.path.join(label_dir, file)) as f:
#             for line in f:
#                 class_id = int(line.strip().split()[0])
#                 counter[class_id] += 1

# # Map class IDs to names
# names = [
#     'Building', 'Manmade-drinage', 'Natural-drinage',
#     'bridge', 'river', 'vegitation inside the river',
#     'obstacles', 'potholes', 'silts'
# ]

# for i in range(len(names)):
#     print(f"{i}: {names[i]} -> {counter.get(i, 0)} labels")
# ____________________________________________________________________________________________

# checking for the model weights that detetcted objects
# from ultralytics import YOLO

# # Load your model checkpoint
# model_path = "runs/segment/yolo9class-finetuned2/weights/best.pt"  # <- change this
# model = YOLO(model_path)

# # Print model details
# print("Model Path:", model_path)
# print("Task Type:", model.task)

# # Get number of classes from model
# nc = model.model.nc if hasattr(model.model, 'nc') else None
# names = model.names if hasattr(model, 'names') else None

# print(f"\n✅ This model was trained with {nc} classes.")
# if names:
#     print("Class names:")
#     for i, name in names.items():
#         print(f"{i}: {name}")

# _________________________________________________________________________________________________________________________________

#  Verify each image has a matching label
# import os

# image_dir = 'YOLO2/valid/images'  # change to 'valid/images' for val set
# label_dir = 'YOLO2/valid/labels'

# missing = []
# for file in os.listdir(image_dir):
#     if file.endswith(('.jpg', '.png', '.jpeg')):
#         label_file = file.rsplit('.', 1)[0] + '.txt'
#         if not os.path.exists(os.path.join(label_dir, label_file)):
#             missing.append(file)

# print(f"Missing labels for {len(missing)} images:")
# for m in missing:
#     print(m)
# ____________________________________________________________________________________________________________________________

# count of the frames in the path

# import os

# train_path = 'train/images'  # or full path like 'C:/Users/NTIN13/Desktop/NEW_YOLO/train/images'
# image_extensions = ('.jpg', '.jpeg', '.png')

# image_count = len([f for f in os.listdir(train_path) if f.lower().endswith(image_extensions)])

# print(f"Total train images: {image_count}")
# _____________________________________________________________________________________________________

# vedio trimmings

#  splitting th vedios 
# import cv2
# import os

# # ── CONFIG ─────────────────────────────
# input_video_path = "ವೃಷಭಾವತಿ ನದಿಯ.mp4"       # Change to your video path
# output_folder    = "video_chunk/"    # Output folder
# chunk_duration   = 120                # Seconds per chunk

# # ── CREATE OUTPUT DIR ────────────────
# os.makedirs(output_folder, exist_ok=True)

# # ── READ VIDEO ───────────────────────
# cap = cv2.VideoCapture(input_video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frames_per_chunk = int(fps * chunk_duration)

# W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# chunk_idx = 0
# frame_idx = 0

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Start new chunk if needed
#     if frame_idx % frames_per_chunk == 0:
#         if out:
#             out.release()
#         chunk_filename = os.path.join(output_folder, f"clip_{chunk_idx:03d}.mp4")
#         out = cv2.VideoWriter(chunk_filename, fourcc, fps, (W, H))
#         print(f"▶️ Writing: {chunk_filename}")
#         chunk_idx += 1

#     out.write(frame)
#     frame_idx += 1

# # ── CLEANUP ───────────────────────────
# cap.release()
# if out:
#     out.release()
# print("✅ Done splitting video!")

# _______________________________________________________________________________________
#  calculoates the fps and frames and vedio resulution

# import cv2
# import os

# # Re-define video paths after reset
# video_paths = [
#     "video_work/annotated_output5.mp4"
# ]

# # Open both videos
# video_info = {}
# for path in video_paths:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         continue
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     duration = frame_count / fps
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     cap.release()

#     video_info[os.path.basename(path)] = {
#         "FPS": fps,
#         "Total Frames": frame_count,
#         "Duration (s)": duration,
#         "Resolution": f"{int(width)}x{int(height)}"
#     }

# print(video_info)



# ________________________________________________________________________________

#   frame resize code
# import cv2

# # Input and output video paths
# input_path = 'video_chunk/clip_044.mp4'                     # Your original video
# output_path = 'clip_000_3840x2160_44.mp4'          # Output upscaled video

# # Target resolution
# target_width = 3840
# target_height = 2160

# # Open the input video
# cap = cv2.VideoCapture(input_path)
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

# # Read and resize frames
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     resized_frame = cv2.resize(frame, (target_width, target_height))
#     out.write(resized_frame)

# # Release everything
# cap.release()
# out.release()
# print(f"Video successfully resized to {target_width}x{target_height} and saved as '{output_path}'")

# _________________________________________________________________________________________________________

# gsd calculation based on the srt file info 

# import re
# import srt

# def parse_srt_gsd(srt_file, sensor_width_mm=13.2, frame_width_px=1920, default_focal_length_mm=23):
#     """
#     Parses an SRT file and calculates GSD (meters/pixel) for each frame using:
#     GSD = (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px)
    
#     Returns:
#         gsd_data: dict mapping frame index to GSD value
#     """
#     with open(srt_file, 'r') as f:
#         contents = f.read()

#     subs = list(srt.parse(contents))
#     gsd_data = {}

#     # Pattern: [altitude: 310.260986] [focal_len: 280] (some may not have focal_len)
#     pattern = re.compile(r"\[.*?altitude: ([\d.]+).*?(?:focal_len ?: ?(\d+))?.*?\]")

#     for sub in subs:
#         match = pattern.search(sub.content)
#         if match:
#             altitude = float(match.group(1))
#             focal_length = float(match.group(2)) if match.group(2) else default_focal_length_mm

#             # GSD calculation
#             gsd = (sensor_width_mm * altitude) / (focal_length * frame_width_px)
#             gsd_data[int(sub.index)] = round(gsd, 5)

#     return gsd_data
# if __name__ == "__main__":
#     SRT_PATH = 'DJI_SRT_DATA/DJI_0010.SRT'
#     FRAME_WIDTH_PX = 1920
#     SENSOR_WIDTH_MM = 13.2
#     FOCAL_LENGTH_MM = 23  # Optional: if focal length isn't available in SRT

#     gsd_per_frame = parse_srt_gsd(
#         srt_file=SRT_PATH,
#         sensor_width_mm=SENSOR_WIDTH_MM,
#         frame_width_px=FRAME_WIDTH_PX,
#         default_focal_length_mm=FOCAL_LENGTH_MM
#     )

#     print("Sample GSDs (first 5 frames):")
#     for k in sorted(gsd_per_frame)[:5]:
#         print(f"Frame {k}: GSD = {gsd_per_frame[k]} m/pixel")


# _________________________________________________________________________________

# C:\Users\NTIN13\Desktop\NEW_YOLO\YOLO3\venv\Lib\site-packages\ultralytics\trackers\byte_tracker.py

# ... (lines before the update method)

#track.py to update the state of a matched track, becuase our tracker has failed due to confedence threshold or IOU threshold
# def update(self, new_track: "STrack", frame_id: int):
#     """
#     Update the state of a matched track.
#     ...
#     """
#     self.frame_id = frame_id
#     self.tracklet_len += 1

#     new_tlwh = new_track.tlwh
    
#     # Add a try-except block to handle Kalman filter errors
#     try:
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.convert_coords(new_tlwh)
#         )
#         self.state = TrackState.Tracked
#         self.is_activated = True
#     except np.linalg.LinAlgError:
#         # If the update fails due to a singular matrix, mark the track as lost.
#         # This prevents the program from crashing.
#         self.mark_lost()
#         LOGGER.warning(f"Kalman filter update failed for track {self.track_id}. Marking as lost.")
    
#     self.score = new_track.score
#     self.cls = new_track.cls
#     self.angle = new_track.angle
#     self.idx = new_track.idx

# ... (rest of the file)

# ________________________________________________
# Create a ready-to-run script that overlays coordinates + area names (from SRT) onto a video.
#!/usr/bin/env python3
# """
# add_area_from_srt.py

# Overlay `Lat: <lat>, Lon: <lon> | Area: <name>` at the bottom of a video
# using ONLY its companion SRT file (with GPS). Works well with DJI-style SRTs.

# Install (once):
#     pip install opencv-python srt geopy

# Usage:
#     python add_area_from_srt.py \
#         --video DJI_0001.mp4 \
#         --srt DJI_0001.srt \
#         --out DJI_0001_annotated.mp4

# Optional:
#     --cache geo_cache.json        # saves reverse geocode results
#     --font-scale 0.9 --thickness 2 --bar-height 0.10 --bar-alpha 0.55

# Notes:
# - Uses OpenStreetMap's Nominatim for reverse geocoding. Internet required.
# - Reverse geocoding is cached by rounded coordinates to minimize lookups.
# - The text bar is drawn at the bottom of the video with a semi‑transparent background.
# """
# import argparse
# import cv2
# import json
# import math
# import os
# import re
# from typing import List, Tuple, Optional

# import srt as srtlib
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

# # -------------------- SRT parsing --------------------
# GPS_PATTERNS = [
#     # Generic "lat, lon"
#     re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),
#     # "GPS: lat,lon"
#     re.compile(r"GPS[^0-9\-]*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),
#     # "Lat: x, Lon: y"
#     re.compile(r"Lat[^0-9\-]*(-?\d+\.\d+).*?Lon[^0-9\-]*(-?\d+\.\d+)", re.IGNORECASE),
# ]

# def parse_srt_gps(path: str) -> List[Tuple[float, float, float, float]]:
#     """Return [(start_sec, end_sec, lat, lon), ...] for entries that include GPS."""
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         content = f.read()
#     subs = list(srtlib.parse(content))

#     entries: List[Tuple[float, float, float, float]] = []
#     for sub in subs:
#         text = sub.content.replace("\n", " ")
#         lat = lon = None
#         for pat in GPS_PATTERNS:
#             m = pat.search(text)
#             if m:
#                 try:
#                     lat = float(m.group(1)); lon = float(m.group(2))
#                     break
#                 except Exception:
#                     pass
#         if lat is None or lon is None:
#             continue
#         entries.append((sub.start.total_seconds(), sub.end.total_seconds(), lat, lon))

#     if not entries:
#         raise ValueError("No GPS coordinates found in SRT. Please check the SRT format.")

#     # Merge consecutive identical coords (reduce jitter & lookups)
#     merged: List[Tuple[float, float, float, float]] = []
#     for s, e, la, lo in entries:
#         if not merged:
#             merged.append([s, e, la, lo])
#         else:
#             ps, pe, pla, plo = merged[-1]
#             if math.isclose(pla, la, abs_tol=1e-7) and math.isclose(plo, lo, abs_tol=1e-7):
#                 merged[-1][1] = e
#             else:
#                 merged.append([s, e, la, lo])
#     return [(s, e, la, lo) for s, e, la, lo in merged]

# # ---------------- Reverse geocoding with cache ----------------
# class ReverseGeocoder:
#     def __init__(self, cache_path: Optional[str] = None):
#         self.geocoder = Nominatim(user_agent="area_overlay_script")
#         # Polite rate limit; we'll cache so this isn't called per frame
#         self.reverse = RateLimiter(self.geocoder.reverse, min_delay_seconds=1.0)
#         self.cache_path = cache_path
#         self.cache = {}
#         if cache_path and os.path.exists(cache_path):
#             try:
#                 with open(cache_path, "r", encoding="utf-8") as f:
#                     self.cache = json.load(f)
#             except Exception:
#                 self.cache = {}

#     @staticmethod
#     def _key(lat: float, lon: float) -> str:
#         # Round to ~1e-4 deg (~11m) for stable caching
#         return f"{round(lat, 4):.4f},{round(lon, 4):.4f}"

#     def get_area(self, lat: float, lon: float) -> str:
#         key = self._key(lat, lon)
#         if key in self.cache:
#             return self.cache[key]
#         try:
#             loc = self.reverse((lat, lon), language="en", timeout=8)
#         except Exception:
#             loc = None
#         area = "Unknown area"
#         if loc:
#             addr = loc.raw.get("address", {})
#             # Prefer local granularity.
#             for k in ["neighbourhood","suburb","hamlet","village","locality","city_district","city","town"]:
#                 if k in addr:
#                     area = addr[k]
#                     break
#             if area == "Unknown area":
#                 for k in ["county","state_district","state"]:
#                     if k in addr:
#                         area = addr[k]
#                         break
#             if area == "Unknown area":
#                 area = loc.address.split(",")[0]
#         self.cache[key] = area
#         if self.cache_path:
#             try:
#                 with open(self.cache_path, "w", encoding="utf-8") as f:
#                     json.dump(self.cache, f, ensure_ascii=False, indent=2)
#             except Exception:
#                 pass
#         return area

# # ------------------- Rendering helpers -------------------

# def draw_bottom_bar(frame, text, bar_alpha=0.55, bar_height_ratio=0.10, pad=18,
#                     font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, font_thickness=2):
#     h, w = frame.shape[:2]
#     bar_h = max(40, int(h * bar_height_ratio))
#     y0 = h - bar_h

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
#     frame = cv2.addWeighted(overlay, bar_alpha, frame, 1 - bar_alpha, 0)

#     # Auto-wrap if text too long
#     max_width = w - 2 * pad
#     x = pad
#     y = y0 + pad + 20
#     words = text.split(" ")
#     line = ""
#     for word in words:
#         test = (line + " " + word).strip()
#         (tw, th), _ = cv2.getTextSize(test, font, font_scale, font_thickness)
#         if tw > max_width and line:
#             cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#             y += th + 10
#             line = word
#         else:
#             line = test
#     if line:
#         cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#     return frame

# # ------------------------- Main -------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Overlay Lat/Lon + Area from SRT onto video")
#     ap.add_argument("--video", required=True, help="Input video path")
#     ap.add_argument("--srt", required=True, help="Companion SRT file with GPS")
#     ap.add_argument("--out", required=True, help="Output video path")
#     ap.add_argument("--cache", default=None, help="Path to cache JSON for reverse geocoding")
#     ap.add_argument("--font-scale", type=float, default=0.9)
#     ap.add_argument("--thickness", type=int, default=2)
#     ap.add_argument("--bar-height", type=float, default=0.10)
#     ap.add_argument("--bar-alpha", type=float, default=0.55)
#     args = ap.parse_args()

#     timeline = parse_srt_gps(args.srt)  # list of (start, end, lat, lon)
#     rg = ReverseGeocoder(cache_path=args.cache)

#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {args.video}")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
#     if not out.isOpened():
#         raise RuntimeError(f"Cannot open writer for: {args.out}")

#     cur_idx = 0
#     cur_lat = cur_lon = None
#     cur_area = "Unknown area"

#     def update_active(tsec: float):
#         nonlocal cur_idx, cur_lat, cur_lon, cur_area
#         # Advance index while current entry ended before tsec
#         while cur_idx < len(timeline) and tsec > timeline[cur_idx][1]:
#             cur_idx += 1
#         if cur_idx >= len(timeline):
#             cur_idx = len(timeline) - 1
#         chosen = None
#         if 0 <= cur_idx < len(timeline):
#             s, e, la, lo = timeline[cur_idx]
#             if s <= tsec <= e:
#                 chosen = (la, lo)
#         if chosen is None:
#             # Fallback to most recent past entry
#             for i in range(len(timeline) - 1, -1, -1):
#                 s, e, la, lo = timeline[i]
#                 if s <= tsec:
#                     chosen = (la, lo)
#                     break
#         if chosen is not None:
#             la, lo = chosen
#             if (cur_lat, cur_lon) != (la, lo):
#                 cur_lat, cur_lon = la, lo
#                 cur_area = rg.get_area(cur_lat, cur_lon)

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         tsec = frame_idx / fps if fps > 0 else 0.0
#         update_active(tsec)

#         if cur_lat is not None and cur_lon is not None:
#             text = f"Lat: {cur_lat:.6f}, Lon: {cur_lon:.6f}  |  Area: {cur_area}"
#             frame = draw_bottom_bar(
#                 frame,
#                 text,
#                 bar_alpha=args.bar_alpha,
#                 bar_height_ratio=args.bar_height,
#                 font_scale=args.font_scale,
#                 font_thickness=args.thickness,
#             )

#         out.write(frame)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     print(f"Done. Wrote: {args.out}")

# if __name__ == "__main__":
#     main()
# ____________________________________________________________________________________
# another code snippet from Predict2.py:

#!/usr/bin/env python3
# """
# add_area_from_srt.py

# Overlay `Lat: <lat>, Lon: <lon> | Area: <name>` at the bottom of a video
# using ONLY its companion SRT file (with GPS). Works well with DJI-style SRTs.

# Install (once):
#     pip install opencv-python srt geopy

# Usage:
#     python add_area_from_srt.py \
#         --video DJI_0001.mp4 \
#         --srt DJI_0001.srt \
#         --out DJI_0001_annotated.mp4

# Optional:
#     --cache geo_cache.json        # saves reverse geocode results
#     --font-scale 0.9 --thickness 2 --bar-height 0.10 --bar-alpha 0.55

# Notes:
# - Uses OpenStreetMap's Nominatim for reverse geocoding. Internet required.
# - Reverse geocoding is cached by rounded coordinates to minimize lookups.
# - The text bar is drawn at the bottom of the video with a semi‑transparent background.
# """
# import argparse     #argparse is a module for parsing command-line arguments
# import cv2 # OpenCV for video processing    
# import json           # JSON for caching reverse geocoding results
# import math       # Math for numerical operations                                     
# import os   # OS for file path handling
# import re    # Regular expressions for parsing GPS coordinates
# from typing import List, Tuple, Optional    #   # Type hints for better code clarity

# import srt as srtlib  # SRT parsing library for subtitle files
# from geopy.geocoders import Nominatim # Reverse geocoder using OpenStreetMap's Nominatim
# from geopy.extra.rate_limiter import RateLimiter # Rate limiter to avoid hitting geocoding API too fast

# # -------------------- SRT parsing --------------------
# GPS_PATTERNS = [
#     # Generic "lat, lon"
#     re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),                                   
#     # "GPS: lat,lon"
#     re.compile(r"GPS[^0-9\-]*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),
#     # "Lat: x, Lon: y"
#     re.compile(r"Lat[^0-9\-]*(-?\d+\.\d+).*?Lon[^0-9\-]*(-?\d+\.\d+)", re.IGNORECASE),
# ] # List of regex patterns to match GPS coordinates in various formats

# def parse_srt_gps(path: str) -> List[Tuple[float, float, float, float]]: # 
#     """Return [(start_sec, end_sec, lat, lon), ...] for entries that include GPS.""" #
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         content = f.read()
#     subs = list(srtlib.parse(content))

#     entries: List[Tuple[float, float, float, float]] = []
#     for sub in subs:
#         text = sub.content.replace("\n", " ")
#         lat = lon = None
#         for pat in GPS_PATTERNS:
#             m = pat.search(text)
#             if m:
#                 try:
#                     lat = float(m.group(1)); lon = float(m.group(2))
#                     break
#                 except Exception:
#                     pass
#         if lat is None or lon is None:
#             continue
#         entries.append((sub.start.total_seconds(), sub.end.total_seconds(), lat, lon))

#     if not entries:
#         raise ValueError("No GPS coordinates found in SRT. Please check the SRT format.")

#     # Merge consecutive identical coords (reduce jitter & lookups)
#     merged: List[Tuple[float, float, float, float]] = []
#     for s, e, la, lo in entries:
#         if not merged:
#             merged.append([s, e, la, lo])
#         else:
#             ps, pe, pla, plo = merged[-1]
#             if math.isclose(pla, la, abs_tol=1e-7) and math.isclose(plo, lo, abs_tol=1e-7):
#                 merged[-1][1] = e
#             else:
#                 merged.append([s, e, la, lo])
#     return [(s, e, la, lo) for s, e, la, lo in merged]

# # ---------------- Reverse geocoding with cache ----------------
# class ReverseGeocoder:
#     def __init__(self, cache_path: Optional[str] = None):
#         self.geocoder = Nominatim(user_agent="area_overlay_script")
#         # Polite rate limit; we'll cache so this isn't called per frame
#         self.reverse = RateLimiter(self.geocoder.reverse, min_delay_seconds=1.0)
#         self.cache_path = cache_path
#         self.cache = {}
#         if cache_path and os.path.exists(cache_path):
#             try:
#                 with open(cache_path, "r", encoding="utf-8") as f:
#                     self.cache = json.load(f)
#             except Exception:
#                 self.cache = {}

#     @staticmethod
#     def _key(lat: float, lon: float) -> str:
#         # Round to ~1e-4 deg (~11m) for stable caching
#         return f"{round(lat, 4):.4f},{round(lon, 4):.4f}"

#     def get_area(self, lat: float, lon: float) -> str:
#         key = self._key(lat, lon)
#         if key in self.cache:
#             return self.cache[key]
#         try:
#             loc = self.reverse((lat, lon), language="en", timeout=8)
#         except Exception:
#             loc = None
#         area = "Unknown area"
#         if loc:
#             addr = loc.raw.get("address", {})
#             # Prefer local granularity.
#             for k in ["neighbourhood","suburb","hamlet","village","locality","city_district","city","town","metropolitan_area","municipality","borough"]:
#                 if k in addr:
#                     area = addr[k]
#                     break
#             if area == "Unknown area":
#                 for k in ["county","state_district","state"]:
#                     if k in addr:
#                         area = addr[k]
#                         break
#             if area == "Unknown area":
#                 area = loc.address.split(",")[0]
#         self.cache[key] = area
#         if self.cache_path:
#             try:
#                 with open(self.cache_path, "w", encoding="utf-8") as f:
#                     json.dump(self.cache, f, ensure_ascii=False, indent=2)
#             except Exception:
#                 pass
#         return area

# # ------------------- Rendering helpers -------------------

# def draw_bottom_bar(frame, text, bar_alpha=0.55, bar_height_ratio=0.10, pad=18,
#                     font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, font_thickness=2):
#     h, w = frame.shape[:2]
#     bar_h = max(40, int(h * bar_height_ratio))
#     y0 = h - bar_h

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
#     frame = cv2.addWeighted(overlay, bar_alpha, frame, 1 - bar_alpha, 0)

#     # Auto-wrap if text too long
#     max_width = w - 2 * pad
#     x = pad
#     y = y0 + pad + 20
#     words = text.split(" ")
#     line = ""
#     for word in words:
#         test = (line + " " + word).strip()
#         (tw, th), _ = cv2.getTextSize(test, font, font_scale, font_thickness)
#         if tw > max_width and line:
#             cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#             y += th + 10
#             line = word
#         else:
#             line = test
#     if line:
#         cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#     return frame

# # ------------------------- Main -------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Overlay Lat/Lon + Area from SRT onto video")
#     ap.add_argument("--video", required=True, help="Input video path")
#     ap.add_argument("--srt", required=True, help="Companion SRT file with GPS")
#     ap.add_argument("--out", required=True, help="Output video path")
#     ap.add_argument("--cache", default=None, help="Path to cache JSON for reverse geocoding")
#     ap.add_argument("--font-scale", type=float, default=0.9)
#     ap.add_argument("--thickness", type=int, default=2)
#     ap.add_argument("--bar-height", type=float, default=0.10)
#     ap.add_argument("--bar-alpha", type=float, default=0.55)
#     args = ap.parse_args()

#     timeline = parse_srt_gps(args.srt)  # list of (start, end, lat, lon)
#     rg = ReverseGeocoder(cache_path=args.cache)

#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {args.video}")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
#     if not out.isOpened():
#         raise RuntimeError(f"Cannot open writer for: {args.out}")

#     cur_idx = 0
#     cur_lat = cur_lon = None
#     cur_area = "Unknown area"

#     def update_active(tsec: float):
#         nonlocal cur_idx, cur_lat, cur_lon, cur_area
#         # Advance index while current entry ended before tsec
#         while cur_idx < len(timeline) and tsec > timeline[cur_idx][1]:
#             cur_idx += 1
#         if cur_idx >= len(timeline):
#             cur_idx = len(timeline) - 1
#         chosen = None
#         if 0 <= cur_idx < len(timeline):
#             s, e, la, lo = timeline[cur_idx]
#             if s <= tsec <= e:
#                 chosen = (la, lo)
#         if chosen is None:
#             # Fallback to most recent past entry
#             for i in range(len(timeline) - 1, -1, -1):
#                 s, e, la, lo = timeline[i]
#                 if s <= tsec:
#                     chosen = (la, lo)
#                     break
#         if chosen is not None:
#             la, lo = chosen
#             if (cur_lat, cur_lon) != (la, lo):
#                 cur_lat, cur_lon = la, lo
#                 cur_area = rg.get_area(cur_lat, cur_lon)

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         tsec = frame_idx / fps if fps > 0 else 0.0
#         update_active(tsec)

#         if cur_lat is not None and cur_lon is not None:
#             text = f"Lat: {cur_lat:.6f}, Lon: {cur_lon:.6f}  |  Area: {cur_area}"
#             frame = draw_bottom_bar(
#                 frame,
#                 text,
#                 bar_alpha=args.bar_alpha,
#                 bar_height_ratio=args.bar_height,
#                 font_scale=args.font_scale,
#                 font_thickness=args.thickness,
#             )

#         out.write(frame)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     print(f"Done. Wrote: {args.out}")

# if __name__ == "__main__":
#     main()
# ________________________________________________________________________________________
# some more ares
#!/usr/bin/env python3
# some more ares
#!/usr/bin/env python3
"""
add_area_from_srt_with_poi.py

Overlay `Lat: <lat>, Lon: <lon> | Area: <name> | Nearby: <POIs>` at the bottom of a video
using ONLY its companion SRT file (with GPS). Works well with DJI-style SRTs.

Now includes **nearby points of interest (POIs)** from OpenStreetMap via Overpass:
- Metro/rail stations
- Temples / places of worship
- Malls
- Major roads / railways

Install (once):
    pip install opencv-python srt geopy requests

Usage:
    python add_area_from_srt_with_poi.py \
        --video DJI_0010.MOV \
        --srt DJI_0010.SRT \
        --out DJI_0010_annotated.mp4 \
        --cache geo_cache.json \
        --poi-cache poi_cache.json \
        --poi-radius 800 \
        --poi-limit 2

Notes:
- Uses OpenStreetMap's Nominatim for reverse geocoding **and** Overpass API for POIs. Internet required.
- Reverse geocoding & POIs are cached by rounded coordinates to minimize lookups.
- The bottom bar auto-wraps long text.
"""
import argparse
import cv2
import json
import math
import os
import re
from typing import List, Tuple, Optional, Dict, Any

import srt as srtlib
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests

# -------------------- SRT parsing --------------------
GPS_PATTERNS = [
    # Generic "lat, lon"
    re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),
    # "GPS: lat,lon"
    re.compile(r"GPS[^0-9\-]*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", re.IGNORECASE),
    # "Lat: x, Lon: y"
    re.compile(r"Lat[^0-9\-]*(-?\d+\.\d+).*?Lon[^0-9\-]*(-?\d+\.\d+)", re.IGNORECASE),
]

def parse_srt_gps(path: str) -> List[Tuple[float, float, float, float]]:
    """Return [(start_sec, end_sec, lat, lon), ...] for entries that include GPS."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    subs = list(srtlib.parse(content))

    entries: List[Tuple[float, float, float, float]] = []
    for sub in subs:
        text = sub.content.replace("\n", " ")
        lat = lon = None
        for pat in GPS_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    lat = float(m.group(1)); lon = float(m.group(2))
                    break
                except Exception:
                    pass
        if lat is None or lon is None:
            continue
        entries.append((sub.start.total_seconds(), sub.end.total_seconds(), lat, lon))

    if not entries:
        raise ValueError("No GPS coordinates found in SRT. Please check the SRT format.")

    # Merge consecutive identical coords (reduce jitter & lookups)
    merged: List[Tuple[float, float, float, float]] = []
    for s, e, la, lo in entries:
        if not merged:
            merged.append([s, e, la, lo])
        else:
            ps, pe, pla, plo = merged[-1]
            if math.isclose(pla, la, abs_tol=1e-7) and math.isclose(plo, lo, abs_tol=1e-7):
                merged[-1][1] = e
            else:
                merged.append([s, e, la, lo])
    return [(s, e, la, lo) for s, e, la, lo in merged]

# ---------------- Reverse geocoding with cache ----------------
class ReverseGeocoder:
    def __init__(self, cache_path: Optional[str] = None):
        self.geocoder = Nominatim(user_agent="area_overlay_script")
        self.reverse = RateLimiter(self.geocoder.reverse, min_delay_seconds=1.0)
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    @staticmethod
    def _key(lat: float, lon: float) -> str:
        # Round to ~1e-4 deg (~11m) for stable caching
        return f"{round(lat, 4):.4f},{round(lon, 4):.4f}"

    def get_area(self, lat: float, lon: float) -> str:
        key = self._key(lat, lon)
        if key in self.cache:
            return self.cache[key]
        try:
            loc = self.reverse((lat, lon), language="en", timeout=8)
        except Exception:
            loc = None
        area = "Unknown area"
        if loc:
            addr = loc.raw.get("address", {})
            for k in ["neighbourhood","suburb","hamlet","village","locality","city_district","city","town"]:
                if k in addr:
                    area = addr[k]
                    break
            if area == "Unknown area":
                for k in ["county","state_district","state"]:
                    if k in addr:
                        area = addr[k]
                        break
            if area == "Unknown area":
                area = loc.address.split(",")[0]
        self.cache[key] = area
        if self.cache_path:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return area

# -------------------- POIs via Overpass --------------------
class POIFinder:
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, radius_m: int = 800, per_cat_limit: int = 2, cache_path: Optional[str] = None):
        self.radius_m = radius_m
        self.per_cat_limit = max(1, per_cat_limit)
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    @staticmethod
    def _key(lat: float, lon: float, r: int) -> str:
        return f"{round(lat, 4):.4f},{round(lon, 4):.4f},{r}"

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def query(self, lat: float, lon: float) -> Dict[str, List[Dict[str, Any]]]:
        key = self._key(lat, lon, self.radius_m)
        if key in self.cache:
            return self.cache[key]

        r = self.radius_m
        q = f"""
[out:json][timeout:25];
(
  node(around:{r},{lat},{lon})[railway=station];
  node(around:{r},{lat},{lon})[station=subway];
  node(around:{r},{lat},{lon})[public_transport=station];
  node(around:{r},{lat},{lon})[amenity=place_of_worship];
  node(around:{r},{lat},{lon})[shop=mall];
  way(around:{r},{lat},{lon})[highway~"^(motorway|trunk|primary|secondary)$"];
  way(around:{r},{lat},{lon})[railway=rail];
);
out center 60;
"""
        try:
            resp = requests.post(self.OVERPASS_URL, data={"data": q}, timeout=30)
            data = resp.json()
        except Exception:
            data = {"elements": []}

        metro, temples, malls, roads, rails = [], [], [], [], []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("ref") or "(unnamed)"
            if el.get("type") == "way":
                c = el.get("center", {})
                lat2, lon2 = c.get("lat"), c.get("lon")
            else:
                lat2, lon2 = el.get("lat"), el.get("lon")
            if lat2 is None or lon2 is None:
                continue
            dist_km = self._haversine_km(lat, lon, lat2, lon2)

            if tags.get("railway") == "station" or tags.get("station") == "subway" or tags.get("subway") == "yes" or tags.get("public_transport") == "station":
                metro.append((dist_km, name))
            elif tags.get("amenity") == "place_of_worship":
                temples.append((dist_km, name))
            elif tags.get("shop") == "mall":
                malls.append((dist_km, name))
            elif tags.get("railway") == "rail":
                rails.append((dist_km, name))
            elif tags.get("highway") in {"motorway","trunk","primary","secondary"}:
                roads.append((dist_km, name))

        def topk(lst):
            lst.sort(key=lambda x: x[0])
            return [{"name": n, "dist_km": round(d, 2)} for d, n in lst[: self.per_cat_limit]]

        result = {
            "metro": topk(metro),
            "temples": topk(temples),
            "malls": topk(malls),
            "roads": topk(roads),
            "railways": topk(rails),
        }
        self.cache[key] = result
        self._save_cache()
        return result

# ------------------- Rendering helpers -------------------

def draw_bottom_bar(frame, text, bar_alpha=0.55, bar_height_ratio=0.10, pad=18,
                    font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, font_thickness=2):
    h, w = frame.shape[:2]
    bar_h = max(56, int(h * bar_height_ratio))
    y0 = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, bar_alpha, frame, 1 - bar_alpha, 0)

    # Auto-wrap if text too long
    max_width = w - 2 * pad
    x = pad
    y = y0 + pad + 20
    words = text.split(" ")
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        (tw, th), _ = cv2.getTextSize(test, font, font_scale, font_thickness)
        if tw > max_width and line:
            cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y += th + 10
            line = word
        else:
            line = test
    if line:
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return frame

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay Lat/Lon + Area + Nearby POIs from SRT onto video")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--srt", required=True, help="Companion SRT file with GPS")
    ap.add_argument("--out", required=True, help="Output video path")
    ap.add_argument("--cache", default=None, help="Path to cache JSON for reverse geocoding")
    ap.add_argument("--poi-cache", default=None, help="Path to cache JSON for POIs (Overpass)")
    ap.add_argument("--poi-radius", type=int, default=800, help="POI search radius (meters)")
    ap.add_argument("--poi-limit", type=int, default=2, help="Max POIs per category")
    ap.add_argument("--font-scale", type=float, default=0.9)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--bar-height", type=float, default=0.11)
    ap.add_argument("--bar-alpha", type=float, default=0.55)
    args = ap.parse_args()

    timeline = parse_srt_gps(args.srt)  # list of (start, end, lat, lon)
    rg = ReverseGeocoder(cache_path=args.cache)
    pf = POIFinder(radius_m=args.poi_radius, per_cat_limit=args.poi_limit, cache_path=args.poi_cache)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open writer for: {args.out}")

    cur_idx = 0
    cur_lat = cur_lon = None
    cur_area = "Unknown area"
    cur_pois: Dict[str, List[Dict[str, Any]]] = {}

    def format_pois(pois: Dict[str, List[Dict[str, Any]]]) -> str:
        parts = []
        if pois.get("metro"):
            parts.append("Metro: " + ", ".join(f"{p['name']}({p['dist_km']}km)" for p in pois["metro"]))
        if pois.get("temples"):
            parts.append("Temples: " + ", ".join(f"{p['name']}({p['dist_km']}km)" for p in pois["temples"]))
        if pois.get("malls"):
            parts.append("Malls: " + ", ".join(f"{p['name']}({p['dist_km']}km)" for p in pois["malls"]))
        if pois.get("railways"):
            parts.append("Railway: " + ", ".join(f"{p['name']}({p['dist_km']}km)" for p in pois["railways"]))
        if pois.get("roads"):
            parts.append("Roads: " + ", ".join(f"{p['name']}({p['dist_km']}km)" for p in pois["roads"]))
        return " | ".join(parts)

    def update_active(tsec: float):
        nonlocal cur_idx, cur_lat, cur_lon, cur_area, cur_pois
        while cur_idx < len(timeline) and tsec > timeline[cur_idx][1]:
            cur_idx += 1
        if cur_idx >= len(timeline):
            cur_idx = len(timeline) - 1
        chosen = None
        if 0 <= cur_idx < len(timeline):
            s, e, la, lo = timeline[cur_idx]
            if s <= tsec <= e:
                chosen = (la, lo)
        if chosen is None:
            for i in range(len(timeline) - 1, -1, -1):
                s, e, la, lo = timeline[i]
                if s <= tsec:
                    chosen = (la, lo)
                    break
        if chosen is not None:
            la, lo = chosen
            if (cur_lat, cur_lon) != (la, lo):
                cur_lat, cur_lon = la, lo
                cur_area = rg.get_area(cur_lat, cur_lon)
                cur_pois = pf.query(cur_lat, cur_lon)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tsec = frame_idx / fps if fps > 0 else 0.0
        update_active(tsec)

        if cur_lat is not None and cur_lon is not None:
            poi_text = format_pois(cur_pois)
            base = f"Lat: {cur_lat:.6f}, Lon: {cur_lon:.6f}  |  Area: {cur_area}"
            full_text = base + (f"  |  Nearby: {poi_text}" if poi_text else "")
            frame = draw_bottom_bar(
                frame,
                full_text,
                bar_alpha=args.bar_alpha,
                bar_height_ratio=args.bar_height,
                font_scale=args.font_scale,
                font_thickness=args.thickness,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. Wrote: {args.out}")

if __name__ == "__main__":
    main()

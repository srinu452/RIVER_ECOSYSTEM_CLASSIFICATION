
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
# d = haversine(12.909025834791612, 77.4866514509254,
#               12.90818989296757, 77.48600295775486)

# print(f"Distance: {d:.2f} meters")


#  splitting th vedios 
# import cv2
# import os

# # ── CONFIG ─────────────────────────────
# input_video_path = "Vrishabhavathi River_Video.mp4"       # Change to your video path
# output_folder    = "video_chunks"    # Output folder
# chunk_duration   = 60                # Seconds per chunk

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


# ____________________________________________________________________________________________________________

# Speed calculator
# import cv2

# # Path to your video
# video_path = 'video_work/clip_05.mp4'

# # Open video file
# cap = cv2.VideoCapture(video_path)

# # Get FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"✅ FPS of the video: {fps}")

# cap.release()

# _________________________________________________________________________________________

# Speed change
# import cv2

# Paths
# input_video = 'video_chunks/clip_016.mp4'         # Replace with your actual video path
# output_video = 'output_50fps.mp4' # Output file
# target_fps = 50                   # Desired FPS

# # Open the input video
# cap = cv2.VideoCapture(input_video)
# original_fps = cap.get(cv2.CAP_PROP_FPS)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the output video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

# print(f"Original FPS: {original_fps} ➜ Converting to {target_fps} FPS...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     out.write(frame)

# cap.release()
# out.release()
# print("✅ Conversion complete. Video saved as:", output_video)
# ____________________________________________________________________________________

# video's image conversion
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

#         if frame_count % 30 == 0:  # Save every 30th frame
#             filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
#             cv2.imwrite(filename, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()
#     print(f"✅ Done. Saved {saved_count} frames from {frame_count} total frames.")

# # Example usage:
# extract_every_30th_frame("video_chunks/clip_006.mp4","video_chunks/images/")

# ________________________________________________________________________________________________

#  calculoates the fps and frames and vedio resulution

import cv2
import os

# Re-define video paths after reset
video_paths = [
    "annotated_output3.mp4",
    "clip_000_3840x2160.mp4"
]

# Open both videos
video_info = {}
for path in video_paths:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    video_info[os.path.basename(path)] = {
        "FPS": fps,
        "Total Frames": frame_count,
        "Duration (s)": duration,
        "Resolution": f"{int(width)}x{int(height)}"
    }

print(video_info)
# ____________________________________________________________________________________________
# #   frame resize code
import cv2

# Input and output video paths
input_path = 'video_chunks/clip_007.mp4'                     # Your original video
output_path = 'clip_000_3840x2160.mp4'          # Output upscaled video

# Target resolution
target_width = 3840
target_height = 2160

# Open the input video
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

# Read and resize frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (target_width, target_height))
    out.write(resized_frame)

# Release everything
cap.release()
out.release()
print(f"Video successfully resized to {target_width}x{target_height} and saved as '{output_path}'")
# ___________________________________________________________________________________________________________________________________



# import os
# from pathlib import Path

# # Folder with your label .txt files
# LABELS_DIR = "labels/"
# os.makedirs(LABELS_DIR, exist_ok=True)

# # Class ID remapping
# # Old ID : New ID
# remap = {
#     3: 4,  # bridge
#     4: 5,  # vegitation inside the river
#     5: 3   # River
# }

# # Path to your extracted labels
# input_dir = Path("NEW_YOLO3/test/labels")  # <-- change this

# for txt_file in input_dir.rglob("*.txt"):
#     new_lines = []
#     with open(txt_file, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts: continue
#             cls = int(parts[0])
#             new_cls = remap.get(cls, cls)
#             new_lines.append(" ".join([str(new_cls)] + parts[1:]))

#     # Save corrected label
#     with open(LABELS_DIR + "/" + txt_file.name, "w") as f:
#         f.write("\n".join(new_lines))

# print("✅ Label class IDs fixed and saved to:", LABELS_DIR)

# River Ecosystem Monitoring System

## Overview
The **River Ecosystem Monitoring System** is an AI/ML-powered solution designed to analyze river surroundings using drone-captured videos, orthoimages, and GPS metadata.  
The system detects, segments, and classifies key features such as:
- Buildings
- Manmade & Natural Drainage Lines
- Bridges
- Vegetation inside rivers
- Obstacles, Potholes, and Silts

This project enables **real-time environmental monitoring**, **infrastructure planning**, and **decision-making** by visualizing results with geo-referenced outputs on **Google Earth Pro** and **GeoServer**.

---

## Key Features
- **YOLOv8 Segmentation & Detection**  
  - Single model for bounding boxes and masks.
  - High-accuracy detection of river-based features.

- **Geospatial Integration**  
  - Utilizes **GPS metadata** and **SRT files** to produce geo-referenced results.
  - Supports visualization through **KML/GeoJSON** overlays.

- **Metric Calculations**
  - Area measurement (e.g., silt deposition, vegetation spread)
  - Length of drainage lines and bridges
  - Object counts (e.g., number of buildings or obstacles)

- **Visualization**
  - Outputs visualized in Google Earth Pro and GeoServer.
  - Annotated videos display bounding boxes and segmentation masks.

---

## Tech Stack
- **Programming Language:** Python
- **AI/ML Frameworks:** YOLOv8, PyTorch
- **Geospatial Tools:** Shapely, Fiona, Rasterio, GeoPandas
- **Visualization:** Google Earth Pro, GeoServer
- **Database:** PostgreSQL/PostGIS
- **Frontend:** Flask (for user uploads and result viewing)

---

## Pipeline
1. **Data Collection**
   - Drone videos, orthoimages, and SRT files (GPS metadata).
   
2. **Preprocessing**
   - Video frame extraction, coordinate alignment, and masking.

3. **Object Detection & Segmentation**
   - YOLOv8 segmentation model trained on river ecosystem features.

4. **Metric Extraction**
   - Area, length, and object counts computed from detections.

5. **Visualization**
   - Results displayed on annotated videos and geo-referenced maps.

---

## Applications
- **Environmental Monitoring**
- **River Restoration Projects**
- **Urban and Rural Planning**
- **Disaster Management & Flood Risk Assessment**

---

## Future Enhancements
- Integration of **LiDAR-based elevation data** for 3D analysis.
- Real-time anomaly detection (e.g., illegal construction near rivers).
- Cloud deployment for large-scale monitoring.

---

## Author
**Srinivasulu**  
AI/ML Engineer | Nakshatech Pvt Ltd  
Expert in Computer Vision, Geospatial AI, and Deep Learning.

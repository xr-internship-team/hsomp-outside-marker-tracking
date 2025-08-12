# HSOMP Outside Marker Tracking

This repository is part of the **HSOMP (Holographic Stabilization with Outside Marker Positioning)** project.  
It is responsible for **tracking a marker (QR code) with an external camera**, estimating the HoloLens 2 position and rotation, and sending this data to the HoloLens via **UDP**.  
This solution improves hologram stability in situations where **SLAM** and **IMU** sensors of the HoloLens fail or produce inaccurate results.



## 📌 Problem
When the HoloLens 2 is used in moving platforms (e.g., vehicles, tanks), **IMU sensor drift** can occur.  
If SLAM algorithms also fail, holograms begin to drift and lose alignment.  
This module solves the problem by using an **external camera** to read a marker placed on the HoloLens and determine its exact position and rotation.

## 🚀 Features
- **AprilTag (marker) detection** from an external camera feed  
- **3D position & rotation** estimation  
- **Adaptive Kalman filter** for smoothing measurements  
- **UDP data transmission** to the HoloLens  

## 📦 Technologies
- **Python**
- **OpenCV**
- **Adaptive Kalman Filter**
- **UDP Networking**
- **AprilTag Tracking**

## 🔧 Installation
```bash
git clone https://github.com/xr-internship-team/hsomp-outside-marker-tracking.git
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## 📷 Camera Calibration

Before running the marker tracking script, you need to calibrate your camera and generate a `calib_params.npz` file.

### Steps:

1. **Print a checkerboard pattern** (e.g., 9x6 inner corners, each square 2.5cm).
2. **Capture multiple images** of the checkerboard from different angles and distances. Save them in a folder, e.g., `calib_images/`.
3. **Run the calibration script:**
   ```bash
   python calibCamera.py
   ```
   This will process all images in `calib_images/`, perform camera calibration, and save the result as `calib_params.npz`.

4. **Place `calib_params.npz` in your project directory.**

> The marker tracking script will automatically load this file for camera parameters.

## ▶ Usage
1. Connect and set up your external camera.  
2. Place an AprilTag (marker) on the HoloLens 2 headset.  
3. Run the tracking script:
```bash
python markerTracking.py
```
(CSV logging can be enabled by setting `ENABLE_CSV_LOG = True` in the script.)

4. The system will:
   - Detect the marker  
   - Estimate position and rotation  
   - Apply Kalman filtering  
   - Send results via UDP to the HoloLens  

## 🔗 Related Repository
For the Unity + MRTK application that runs on HoloLens 2 and receives the tracking data sent from this system, please see the related repository:
[HSOMP Holographic Visualizer](https://github.com/xr-internship-team/hsomp-holographic-visualizer)


## 📜 License
This project is licensed under the terms specified in the repository.
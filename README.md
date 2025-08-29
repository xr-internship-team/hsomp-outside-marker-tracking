HSOMP Outside FaceMesh Tracking

This repository is part of the HSOMP (Hologram Stability on Moving Platform) project.
It is responsible for tracking facial landmarks (MediaPipe FaceMesh) with an external camera, estimating the HoloLens 2 position and rotation, and sending this data to the HoloLens via UDP.
This solution improves hologram stability in situations where SLAM and IMU sensors of the HoloLens fail or produce inaccurate results.

📌 Problem

When the HoloLens 2 is used on moving platforms (e.g., vehicles, tanks), IMU sensor drift may occur.
If SLAM algorithms also fail, holograms begin to drift and lose alignment.
This module solves the problem by using MediaPipe FaceMesh through an external camera to detect facial landmarks and determine the headset’s exact position and rotation.

<p align="center"> <img src="assets/HLwithFaceMesh.jpeg" alt="Hololens 2 with FaceMesh" width="400"/> <br> <em>Microsoft Hololens 2 tracked with external camera using FaceMesh.</em> </p>
🚀 Features

MediaPipe FaceMesh facial landmark detection

3D position & rotation estimation

Adaptive Kalman filter for smoothing measurements

UDP data transmission to the HoloLens

📦 Technologies

Python

OpenCV

MediaPipe FaceMesh

Adaptive Kalman Filter

UDP Networking

🔧 Installation
git clone https://github.com/xr-internship-team/hsomp-outside-facemesh-tracking.git


Install dependencies:

pip install -r requirements.txt

📷 Camera Calibration

As with the marker-based version, this system also requires camera calibration. The calibration parameters are stored in a calib_params.npz file.

Steps:

Print a checkerboard pattern (e.g., 9x6 inner corners, each square 2.5cm).

Capture multiple images of the checkerboard from different angles and distances, and store them in a folder, e.g., calib_images/.

Run the calibration script:

python calibCamera.py


This will process all images in calib_images/, perform camera calibration, and save the result as calib_params.npz.

Place calib_params.npz in your project directory.

The tracking script will automatically load this file for camera parameters.

▶ Usage

Connect and set up your external camera.

Position yourself so that your face (with or without glasses) is visible to the external camera while wearing HoloLens 2.

Run the tracking script:

python facemeshTracking.py


(CSV logging can be enabled by setting ENABLE_CSV_LOG = True in the script.)

The system will:

Detect FaceMesh landmarks

Estimate position and rotation

Apply Kalman filtering

Send results via UDP to the HoloLens

<p align="center"> <img src="assets/video_facemesh_with_glasses.gif" alt="facemesh_with_glasses" width="600"/> <br> <em>FaceMesh landmark detection and pose estimation with glasses.</em> </p> <p align="center"> <img src="assets/video_facemesh_without_glasses.gif" alt="facemesh_without_glasses" width="600"/> <br> <em>FaceMesh landmark detection and pose estimation without glasses.</em> </p>
🔗 Related Repository

For the Unity + MRTK application that runs on HoloLens 2 and receives the tracking data sent from this system, please see the related repository:
HSOMP Holographic Visualizer

📜 License

This project is licensed under the terms specified in the repository.
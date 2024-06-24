# Vehicle Counting and Speed Estimation using YOLO and SORT

This project demonstrates vehicle counting and speed estimation using YOLO (You Only Look Once) for object detection and SORT (Simple Online and Realtime Tracking) for tracking the detected vehicles in a video stream. The aim is to automate the process of monitoring traffic flow, which is essential for effective traffic management and planning.

## Insights

### Motivation
Traffic congestion is a common issue in urban areas, leading to increased travel time, fuel consumption, and pollution. Manual traffic monitoring is labor-intensive and prone to errors. An automated system that can accurately count vehicles and estimate their speeds in real-time can significantly enhance traffic management and reduce congestion.

### Approach
This project leverages state-of-the-art computer vision techniques to detect and track vehicles in real-time. The combination of YOLO for object detection and SORT for object tracking provides a robust solution for vehicle counting and speed estimation.

### YOLO (You Only Look Once)
YOLO is an object detection algorithm known for its speed and accuracy. It divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell. YOLO is capable of detecting multiple objects in a single frame, making it suitable for real-time applications.

### SORT (Simple Online and Realtime Tracking)
SORT is an efficient tracking algorithm that associates detections from YOLO across frames to maintain object identities. It uses a combination of the Kalman filter for predicting object positions and the Hungarian algorithm for data association. SORT is known for its simplicity and high performance in real-time tracking scenarios.

### Speed Estimation
In addition to counting vehicles, the system estimates their speeds by analyzing the displacement of the bounding boxes between frames. This is crucial for traffic management, as it helps in understanding traffic flow dynamics and identifying potential speeding violations.

### Practical Applications
- **Traffic Monitoring:** Automated counting and speed estimation of vehicles can help traffic authorities monitor congestion in real-time and take necessary actions.
- **Urban Planning:** Data collected from this system can be used for urban planning, such as optimizing traffic light timings, planning new roads, and managing traffic during events.
- **Law Enforcement:** Speed estimation can assist in identifying and penalizing speeding vehicles, improving road safety.

Here is the link for breakdown and casestudy for this project [Vehicle Counting Blog](https://github.com/shahrul-amin/Blog-Vehicle-Counting-Using-Yolo-And-Sort)

## Table of Contents
- Installation
- Usage
- References

## Installation

### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required packages.
4. Download the YOLOv3 weights and configuration files by running:
   ```bash
   bash download_weights

## Usage

### Running the Project
1. Place the input video file in the `input` directory. Ensure the file is named `input_video.mp4` or update the script with the correct file name.
2. Run the vehicle counting script using `python vehicle_counting.py`.
3. The output video with the counted vehicles and estimated speeds will be saved in the `output` directory as `output_video.avi`.

### Script Breakdown
The main script performs the following steps:
- Imports necessary packages.
- Initializes the SORT tracker and necessary variables.
- Loads the YOLO model and initializes the video stream.
- Processes each frame: reads frames, performs object detection using YOLO, tracks objects using SORT, counts vehicles crossing a predefined line, and estimates their speeds.
- Saves the processed frames to an output video file.

## References
- [YOLO Object Detection with OpenCV](https://pjreddie.com/darknet/yolo/)
- [SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)
- [Vehicle Counting using Python and YOLO](https://github.com/bamwani/vehicle-counting-using-python-yolo)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) by Joseph Redmon and Ali Farhadi
- [Multiple Object Tracking using SORT](https://www.luffca.com/2023/04/multiple-object-tracking-sort/)
- [Car Counting and Speed Estimation](https://github.com/bamwani/car-counting-and-speed-estimation-yolo-sort-python)

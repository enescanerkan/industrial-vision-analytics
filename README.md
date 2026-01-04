# Machine Detection and Cycle Analysis System

This project is a computer vision system designed for industrial monitoring. It performs real-time machine detection, object tracking, and zone-based counting to facilitate production tracking and cycle time analysis on video footage.

## Features

- Deep learning-based machine detection using YOLO
- Robust object tracking with ByteTrack
- Customizable zone definitions for specific areas of interest
- Zone-based counting and occupancy monitoring
- Real-time visualization of detection and tracking results

## Demo Videos

### Zone Definition Process
The video below demonstrates the zone selection interface where regions of interest are marked using bounding boxes to define worker monitoring areas:

https://github.com/user-attachments/assets/c84a040c-7c2a-4476-8494-1f06bf321511

### System in Action
Watch the complete system detecting and tracking workers in real-time with zone-based analytics:

https://github.com/user-attachments/assets/37142b9f-5c17-4576-8c19-380aee78bf67

## Requirements

The following dependencies are required to run the project:

- numpy>=1.21.0
- opencv-python>=4.8.0
- torch>=2.0.0
- torchvision>=0.15.0
- ultralytics>=8.0.0
- supervision>=0.18.0

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/enescanerkan/industrial-vision-analytics.git
```

2. Navigate to the project directory:
```bash
   cd industrial-vision-analytics/machine-detection
```

3. Install the required packages:
```bash
   pip install -r requirements.txt
```

4. Ensure your trained YOLO model file is placed in the `models` directory.

## Configuration

Configuration parameters can be modified in `src/config.py`:

- **BASE_DIR**: Root directory of the project
- **MODEL_PATH**: Path to the YOLO model file
- **VIDEO_PATH**: Path to the input video file for analysis
- **ZONES_PATH**: Path to the JSON file storing zone coordinates
- **CONFIDENCE_THRESHOLD**: Minimum confidence score for detections
- **IOU_THRESHOLD**: Intersection over Union threshold for tracking

## Usage

### 1. Zone Definition

Before running the analysis, define the regions of interest using the selector tool:
```bash
python zones/zone_selector.py
```

**Controls:**
- **Left Click & Drag**: Draw a rectangular zone
- **Release Mouse**: Complete the selection
- **'s'**: Save the defined zones to the JSON file
- **'r'**: Remove the last drawn zone
- **'q'**: Quit the application

### 2. Detection and Analysis

To execute the main detection, tracking, and counting system:
```bash
python src/main.py
```

## Project Structure
```
machine-detection/
├── src/
│   ├── config.py        # Configuration settings
│   ├── main.py          # Main execution script
│   ├── detect.py        # Object detection logic
│   └── zone_counter.py  # Logic for zone counting and analysis
├── zones/
│   ├── zone_selector.py # GUI tool for defining zones
│   └── zones.json       # Stored zone coordinates
├── models/
│   ├── best_wc.pt       # Trained YOLO model weights
│   └── bytetrack.yaml   # Tracker configuration
└── requirements.txt
```

## Important Notes

- The system automatically detects and utilizes CUDA-enabled GPUs for accelerated processing.
- Zone definitions are resolution-independent and will scale automatically if the video resolution changes.
- The integration of ByteTrack ensures that machines are tracked consistently across frames, preventing duplicate counts and enabling accurate cycle time estimation.

# Face tracking

## Installation
```bash
pip3 install -r requirements.txt
```

## Usage
```bash
python3 run.py
```

## Current configuration
1. Camera: 8 FPS 
2. Detector: Haar Cascade
3. Tracker: CSRT Tracker
3. Classifier: MobileNetv2 (120ms/image on Pi)
4. Redetect: 15 frames -> redetect
5. IOU threshold: 0.9
6. Extract face: whenever new faces are detected



## Experiences
1. Run 3 process
2. Use all 4 CPUs on Raspberry, 70%/CPU on average
3. Use 200MB of RAM (Pi has 1GB RAM)
4. Recent MobileNetv2 classifier takes about 120ms/image on Pi
5. Tracking task: 160-200 ms
6. Redetect and update: 1 - 1.5 s
7. The whole system: 4 FPS on Pi



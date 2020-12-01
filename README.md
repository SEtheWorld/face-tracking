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
1. Camera: 30FPS
2. Detector: Haar Cascade
3. Classifier: MobileNetv2 (120ms/image on Pi)
4. Redetect: 20 frames -> redetect
5. IOU threshold: 0.7
6. Extract face: 5 frames -> extract



## Experiences
1. Run 5 threads parallel
2. Use all 4 CPUs on Raspberry, 70%/CPU on average
3. Use 120MB of RAM (Pi has 1GB RAM)
4. Recent MobileNetv2 classifier takes about 120ms/image on Pi


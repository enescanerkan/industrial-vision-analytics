import os

class Config:
    BASE_DIR = r"C:\Users\Monster\Desktop\machine-detection"
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_bwc.pt')
    VIDEO_PATH = os.path.join(BASE_DIR, 'test_videos', 'wc_1.mp4')
    ZONES_PATH = os.path.join(BASE_DIR, 'zones', 'zones.json')
    TRACKER_CONFIG = os.path.join(BASE_DIR, 'models', 'bytetrack.yaml')
    
    # Tracking parametreleri
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.5
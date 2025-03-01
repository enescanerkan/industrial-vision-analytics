import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import json
import torch
from zone_counter import ZoneCounter
from config import Config
from cycle_time_analyzer import CycleTimeAnalyzer

class MachineDetector:
    def __init__(self, model_path, video_path):
        # GPU kontrolü
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Model yükleme
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)
        
        # Video kaynağı
        self.video_path = video_path
        
        # zones.json yolunu belirle
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.zones_path = os.path.join(self.project_dir, 'zones.json')
        
        # zones.json'dan bölgeleri yükle
        self.zones = self._load_zones()
        self.zone_counter = ZoneCounter(self.zones)
        self.cycle_analyzer = CycleTimeAnalyzer(self.zones)

    def _load_zones(self):
        if os.path.exists(Config.ZONES_PATH):
            with open(Config.ZONES_PATH, 'r') as f:
                data = json.load(f)
                
            # Video frame boyutlarını al
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                current_height, current_width = frame.shape[:2]
                saved_height = data["frame_size"]["height"]
                saved_width = data["frame_size"]["width"]
                
                # Boyutlar farklıysa ölçeklendirme yap
                if current_height != saved_height or current_width != saved_width:
                    scale_x = current_width / saved_width
                    scale_y = current_height / saved_height
                    
                    scaled_zones = {}
                    for zone_name, zone_info in data["zones"].items():
                        coords = zone_info["coords"]
                        scaled_coords = [
                            int(coords[0] * scale_x),  # x1
                            int(coords[1] * scale_y),  # y1
                            int(coords[2] * scale_x),  # x2
                            int(coords[3] * scale_y)   # y2
                        ]
                        scaled_zones[zone_name] = {
                            "coords": scaled_coords,
                            "count": zone_info["count"]
                        }
                    return scaled_zones
                
                return data["zones"]
            
        # Varsayılan zone'lar
        return {
            "zone1": {"coords": [100, 100, 300, 300], "count": 0},
            "zone2": {"coords": [400, 100, 600, 300], "count": 0},
            "zone3": {"coords": [700, 100, 900, 300], "count": 0}
        }

    def _format_detections(self, results):
        # CUDA hatası için güncellendi
        boxes = results.boxes.xyxy.to('cpu').numpy()  # Direkt CPU'ya taşı
        scores = results.boxes.conf.to('cpu').numpy()
        class_ids = results.boxes.cls.to('cpu').numpy().astype(int)
        
        return sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )

    def _draw_results(self, frame, tracks):
        # Tespit kutularını çiz - ince çizgi (thickness=1) ve mavi renk
        for box, track_id in zip(tracks.xyxy, tracks.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            # Bounding box - ince mavi çizgi (BGR: 255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # ID yazısı - küçük punto ve mavi renk
            # Font scale 0.5 (daha küçük yazı)
            cv2.putText(frame, 
                       f"ID: {track_id}", 
                       (x1, y1-5),  # Biraz daha yukarı
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4,  # Font boyutu küçültüldü
                       (255, 0, 0),  # Mavi renk (BGR)
                       1)  # Yazı kalınlığı

        # Bölgeleri çiz
        for zone_name, zone_info in self.zones.items():
            x1, y1, x2, y2 = zone_info["coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Zone çizgileri aynı kalınlıkta
            cv2.putText(frame, 
                       f"{zone_name}: {zone_info['count']}", 
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       (0, 0, 255), 
                       2)

    def _draw_cycle_times(self, frame, current_cycles):
        """Frame üzerine cycle time bilgilerini çiz"""
        # Her zone için ayrı y-offset takibi
        zone_y_offsets = {zone_name: 0 for zone_name in self.zones.keys()}
        
        for track_id, zones in current_cycles.items():
            for zone_name, time_in_zone in zones.items():
                try:
                    if zone_name in self.zones:
                        zone_coords = self.zones[zone_name]["coords"]
                        x1, y1 = zone_coords[0], zone_coords[1]
                        
                        # Y pozisyonunu hesapla (20 piksel aralıklarla)
                        text_y = y1 - 25 - (zone_y_offsets[zone_name] * 15)
                        
                        # Metni hazırla ve çiz
                        text = f"ID:{track_id} {time_in_zone:.1f}s"
                        cv2.putText(
                            frame,
                            text,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,  # Font boyutu küçültüldü
                            (0, 255, 0),
                            1    # Çizgi kalınlığı azaltıldı
                        )
                        
                        # Y-offset'i güncelle
                        zone_y_offsets[zone_name] += 1
                except Exception as e:
                    print(f"Cycle time çiziminde hata: track_id={track_id}, zone={zone_name}, hata={str(e)}")
                    continue
    
    def process_frame(self, frame):
        # Model ile tespit ve tracking
        with torch.no_grad():
            results = self.model.track(
                frame,
                conf=Config.CONFIDENCE_THRESHOLD,
                iou=Config.IOU_THRESHOLD,
                tracker=Config.TRACKER_CONFIG,
                persist=True,
                device=self.device
            )[0]
        
        # Sonuçları CPU'ya taşı
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Detections oluştur
            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=class_ids,
                tracker_id=track_ids
            )
            
            # Bölge sayımlarını güncelle
            self.zone_counter.update(detections)
            
            # Cycle time analizi
            self.cycle_analyzer.update(detections)
            current_cycles = self.cycle_analyzer.get_current_cycle_times()
            
            # Görselleştirme
            self._draw_results(frame, detections)
            self._draw_cycle_times(frame, current_cycles)
            
            # Her 1000 frame'de bir istatistikleri kaydet
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
                
            if self.frame_count % 1000 == 0:
                self.cycle_analyzer.save_statistics('cycle_time_stats.json')
        
        return frame
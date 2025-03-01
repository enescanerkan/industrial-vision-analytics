import numpy as np
import supervision as sv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import time

@dataclass
class TrackInfo:
    last_seen: float
    last_position: Tuple[float, float]
    in_zones: Set[str]
    completed_zones: Set[str]

class ZoneCounter:
    def __init__(self, zones, max_disappeared_time=1.0, max_distance=50):
        self.zones = zones
        self.track_history = {}  # track_id -> TrackInfo
        self.zone_counts = defaultdict(int)
        self.max_disappeared_time = max_disappeared_time  # saniye
        self.max_distance = max_distance  # piksel
        self.disappeared_tracks = {}  # {track_id: (last_position, last_seen_time)}
        
    def _calculate_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _is_in_zone(self, center: Tuple[float, float], zone_coords: List[int]) -> bool:
        x, y = center
        x1, y1, x2, y2 = zone_coords
        return x1 < x < x2 and y1 < y < y2

    def _handle_disappeared_tracks(self, current_time: float, detections):
        # Kaybolan track'leri kontrol et
        current_track_ids = set(detections.tracker_id)
        
        # Yeni tespit edilen track'ler için en yakın kaybolan track'i bul
        for i, track_id in enumerate(detections.tracker_id):
            if track_id not in self.track_history:
                center = self._calculate_center(detections.xyxy[i])
                
                # En yakın kaybolan track'i bul
                closest_old_id = None
                min_distance = float('inf')
                
                for old_id, (old_center, last_seen) in list(self.disappeared_tracks.items()):
                    if current_time - last_seen > self.max_disappeared_time:
                        continue
                        
                    distance = np.sqrt((center[0] - old_center[0])**2 + 
                                     (center[1] - old_center[1])**2)
                    
                    if distance < self.max_distance and distance < min_distance:
                        min_distance = distance
                        closest_old_id = old_id
                
                # Eğer yakın bir track bulunduysa, ID'yi geri yükle
                if closest_old_id is not None:
                    self.track_history[track_id] = self.track_history.pop(closest_old_id)
                    del self.disappeared_tracks[closest_old_id]

        # Mevcut frame'de görünmeyen track'leri kaydet
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                track_info = self.track_history[track_id]
                self.disappeared_tracks[track_id] = (track_info.last_position, current_time)

    def update(self, detections: sv.Detections):
        current_time = time.time()
        
        # Kaybolan track'leri kontrol et
        self._handle_disappeared_tracks(current_time, detections)
        
        # Her tespit için zone kontrolü
        for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            center = self._calculate_center(bbox)
            
            # Track geçmişi yoksa oluştur
            if track_id not in self.track_history:
                self.track_history[track_id] = TrackInfo(
                    last_seen=current_time,
                    last_position=center,
                    in_zones=set(),
                    completed_zones=set()
                )
            
            track_info = self.track_history[track_id]
            track_info.last_seen = current_time
            track_info.last_position = center
            
            # Her zone için kontrol
            for zone_name, zone_info in self.zones.items():
                is_in_zone = self._is_in_zone(center, zone_info["coords"])
                
                if is_in_zone:
                    # Zone'a yeni giriş
                    if zone_name not in track_info.in_zones:
                        track_info.in_zones.add(zone_name)
                else:
                    # Zone'dan çıkış
                    if zone_name in track_info.in_zones:
                        track_info.in_zones.remove(zone_name)
                        if zone_name not in track_info.completed_zones:
                            track_info.completed_zones.add(zone_name)
                            self.zones[zone_name]["count"] += 1
                            print(f"ID {track_id} completed {zone_name}. New count: {self.zones[zone_name]['count']}")

        # Uzun süre görünmeyen track'leri temizle
        for track_id in list(self.disappeared_tracks.keys()):
            if current_time - self.disappeared_tracks[track_id][1] > self.max_disappeared_time:
                if track_id in self.track_history:
                    del self.track_history[track_id]
                del self.disappeared_tracks[track_id]

    def get_counts(self):
        return {zone: info["count"] for zone, info in self.zones.items()}

    def get_active_tracks(self):
        return {track_id: info.in_zones 
                for track_id, info in self.track_history.items()}
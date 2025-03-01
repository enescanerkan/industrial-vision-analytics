from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import time
import json
import numpy as np
from collections import defaultdict
import traceback  # Hata detayı için ekledik
from datetime import datetime

@dataclass
class ObjectCycleData:
    entry_time: float
    exit_time: float = None
    zone_name: str = None
    cycle_complete: bool = False

class CycleTimeAnalyzer:
    def __init__(self, zones):
        self.zones = zones
        # Her zone için ayrı bir dictionary tutuyoruz
        self.zone_objects = {zone_name: {} for zone_name in zones.keys()}  # {zone_name: {track_id: ObjectCycleData}}
        self.completed_cycles = defaultdict(list)  # {zone_name: [completed_cycles]}
        self.cycle_stats = {zone_name: [] for zone_name in zones.keys()}  # Her bölge için ayrı istatistik
        
    def _calculate_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _is_in_zone(self, center: Tuple[float, float], zone_coords: List[int]) -> bool:
        x, y = center
        x1, y1, x2, y2 = zone_coords
        return x1 < x < x2 and y1 < y < y2

    def _convert_to_native_types(self, obj):
        """NumPy tiplerini native Python tiplerine dönüştür"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self._convert_to_native_types(key): self._convert_to_native_types(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        return obj
    
    def update(self, detections):
        try:
            current_time = time.time()
            
            # Eğer detections None ise veya tracker_id yoksa, işlemi atla
            if detections is None or not hasattr(detections, 'tracker_id') or len(detections.tracker_id) == 0:
                return
            
            current_track_ids = set(map(int, detections.tracker_id))  # track_id'leri int'e çevir
            
            # Her zone için aktif boxları takip et
            active_boxes = {zone_name: set() for zone_name in self.zones.keys()}
            
            # Mevcut frame'deki tespitleri işle
            for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                try:
                    track_id = int(track_id)  # NumPy int64'ü normal int'e çevir
                    center = self._calculate_center(bbox)
                    
                    # Her bölge için kontrol
                    for zone_name, zone_info in self.zones.items():
                        is_in_zone = self._is_in_zone(center, zone_info["coords"])
                        
                        if is_in_zone:
                            active_boxes[zone_name].add(track_id)
                            
                            # Nesne bölgeye yeni girdiyse
                            if track_id not in self.zone_objects[zone_name]:
                                self.zone_objects[zone_name][track_id] = ObjectCycleData(
                                    entry_time=current_time,
                                    zone_name=zone_name
                                )
                                print(f"Box ID {track_id} entered {zone_name}")
                except Exception as box_error:
                    print(f"Box işlenirken hata: track_id={track_id}, hata={str(box_error)}")
                    continue
            
            # Her zone için çıkan veya kaybolan nesneleri kontrol et
            for zone_name in self.zones.keys():
                try:
                    current_zone_tracks = set(self.zone_objects[zone_name].keys())
                    
                    # Zone'dan çıkan nesneleri bul
                    exited_tracks = current_zone_tracks - active_boxes[zone_name]
                    
                    for track_id in exited_tracks:
                        cycle_data = self.zone_objects[zone_name][track_id]
                        if not cycle_data.cycle_complete:
                            cycle_data.exit_time = current_time
                            cycle_data.cycle_complete = True
                            cycle_time = cycle_data.exit_time - cycle_data.entry_time
                            
                            # İstatistikleri güncelle
                            self.cycle_stats[zone_name].append(cycle_time)
                            
                            # Tamamlanan döngüyü kaydet
                            completed_cycle = {
                                'track_id': track_id,
                                'entry_time': float(cycle_data.entry_time),
                                'exit_time': float(cycle_data.exit_time),
                                'cycle_time': float(cycle_time)
                            }
                            self.completed_cycles[zone_name].append(completed_cycle)
                            print(f"Box ID {track_id} exited {zone_name} after {cycle_time:.2f} seconds")
                            
                            # Tamamlanan cycle'ı temizle
                            del self.zone_objects[zone_name][track_id]
                except Exception as zone_error:
                    print(f"Zone işlenirken hata: zone={zone_name}, hata={str(zone_error)}")
                    continue
                
        except Exception as e:
            print(f"Cycle time analizi sırasında hata: {str(e)}")
            print(traceback.format_exc())
    
    def get_current_cycle_times(self):
        """Aktif nesnelerin anlık cycle time'larını döndür"""
        current_time = time.time()
        current_cycles = {}
        
        # Her track_id için bir dictionary oluştur
        for zone_name, tracks in self.zone_objects.items():
            for track_id, cycle_data in tracks.items():
                if not cycle_data.cycle_complete:
                    if track_id not in current_cycles:
                        current_cycles[track_id] = {}
                    current_time_in_zone = current_time - cycle_data.entry_time
                    current_cycles[track_id][zone_name] = current_time_in_zone
        
        return current_cycles
    
    def get_zone_statistics(self):
        """Her bölge için istatistikleri hesapla"""
        stats = {}
        for zone_name in self.zones.keys():
            cycle_times = self.cycle_stats[zone_name]
            if cycle_times:
                stats[zone_name] = {
                    'min_time': float(min(cycle_times)),
                    'max_time': float(max(cycle_times)),
                    'avg_time': float(sum(cycle_times) / len(cycle_times)),
                    'total_objects': len(cycle_times),
                    'current_objects': len([t for t in self.zone_objects[zone_name].values() 
                                         if not t.cycle_complete])
                }
            else:
                stats[zone_name] = {
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'avg_time': 0.0,
                    'total_objects': 0,
                    'current_objects': 0
                }
        return stats
    
    def _format_timestamp(self, timestamp):
        """Unix timestamp'i okunabilir formata çevir"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

    def save_statistics(self, output_path):
        """İstatistikleri JSON dosyasına kaydet"""
        stats = {
            'zone_statistics': self.get_zone_statistics(),
            'completed_cycles': {
                zone_name: [{
                    'track_id': cycle['track_id'],
                    'entry_time': self._format_timestamp(cycle['entry_time']),
                    'exit_time': self._format_timestamp(cycle['exit_time']),
                    'cycle_time': cycle['cycle_time'],
                    'cycle_time_formatted': f"{cycle['cycle_time']:.2f} saniye"
                } for cycle in cycles]
                for zone_name, cycles in self.completed_cycles.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4) 
import cv2
import json
import os
import sys

# Proje kök dizinini Python path'ine ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import Config  # Config'i import ediyoruz

class ZoneSelector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.zones = {}
        self.current_zone = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.zone_count = 1
        self.temp_frame = None  # Geçici frame'i saklayacak
        self.frame_size = None  # Frame boyutlarını saklamak için

    def normalize_coordinates(self, start_point, end_point):
        """Koordinatları normalize eder (sol üst, sağ alt formatına çevirir)"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # x koordinatlarını sırala
        left_x = min(x1, x2)
        right_x = max(x1, x2)
        
        # y koordinatlarını sırala
        top_y = min(y1, y2)
        bottom_y = max(y1, y2)
        
        return [left_x, top_y, right_x, bottom_y]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)  # Başlangıçta end_point'i de set et
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_frame = self.original_frame.copy()  # Orijinal frame'i kopyala
            self.end_point = (x, y)
            
            # Mevcut zone'ları çiz
            self.draw_existing_zones(self.temp_frame)
            
            # Geçici zone'u çiz
            cv2.rectangle(self.temp_frame, 
                        self.start_point, 
                        self.end_point, 
                        (255, 0, 0), 2)
            
            cv2.imshow("Zone Selector", self.temp_frame)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Koordinatları normalize et
            coords = self.normalize_coordinates(self.start_point, self.end_point)
            
            # Minimum boyut kontrolü
            if abs(coords[2] - coords[0]) > 20 and abs(coords[3] - coords[1]) > 20:
                zone_name = f"zone{self.zone_count}"
                self.zones[zone_name] = {
                    "coords": coords,
                    "count": 0
                }
                self.zone_count += 1
                print(f"\nZone eklendi: {zone_name}")
                print(f"Koordinatlar: {coords}")
            else:
                print("\nZone çok küçük! Lütfen daha büyük bir alan seçin.")

    def draw_existing_zones(self, frame):
        for zone_name, zone_info in self.zones.items():
            coords = zone_info["coords"]
            cv2.rectangle(frame, 
                        (coords[0], coords[1]), 
                        (coords[2], coords[3]), 
                        (0, 0, 255), 2)
            cv2.putText(frame, zone_name, 
                      (coords[0], coords[1]-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                      (0, 0, 255), 2)

    def select_zones(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Video açılamadı!")
            return

        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı!")
            return

        # Frame boyutlarını kaydet
        self.frame_size = frame.shape[:2]  # (height, width)
        self.original_frame = frame.copy()
        self.temp_frame = self.original_frame.copy()

        cv2.namedWindow("Zone Selector")
        cv2.setMouseCallback("Zone Selector", self.mouse_callback)

        print("\nKullanım:")
        print("1. Sol tık ile çizmeye başlayın")
        print("2. Sürükleyerek zone'u belirleyin")
        print("3. Bırakın")
        print("4. 'q' tuşu ile çıkın")
        print("5. 's' tuşu ile kaydedin")
        print("6. 'r' tuşu ile son zone'u silin")

        while True:
            if not self.drawing:
                display_frame = self.original_frame.copy()
                self.draw_existing_zones(display_frame)
                cv2.imshow("Zone Selector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_zones()
                print("\nZone'lar kaydedildi!")
                break
            elif key == ord('r') and self.zones:  # Son zone'u silme
                last_zone = f"zone{self.zone_count-1}"
                if last_zone in self.zones:
                    del self.zones[last_zone]
                    self.zone_count -= 1
                    print(f"\n{last_zone} silindi")

        cap.release()
        cv2.destroyAllWindows()

    def save_zones(self):
        zones_dir = os.path.join(os.path.dirname(self.video_path), '..', 'zones')
        os.makedirs(zones_dir, exist_ok=True)
        zones_path = os.path.join(zones_dir, 'zones.json')
        
        # Frame boyutlarıyla birlikte kaydet
        data = {
            "frame_size": {
                "height": self.frame_size[0],
                "width": self.frame_size[1]
            },
            "zones": self.zones
        }
        
        with open(zones_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("\nZone koordinatları:")
        print(f"Frame boyutları: {self.frame_size}")
        print("self.zones = {")
        for zone_name, zone_info in self.zones.items():
            print(f'    "{zone_name}": {{"coords": {zone_info["coords"]}, "count": 0}},')
        print("}")

def main():
    selector = ZoneSelector(Config.VIDEO_PATH)
    selector.select_zones()

if __name__ == "__main__":
    main()
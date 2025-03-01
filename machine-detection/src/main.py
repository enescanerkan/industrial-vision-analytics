import cv2
from detect import MachineDetector
from config import Config
import traceback  # Hata detayı için ekledik

def main():
    detector = MachineDetector(
        model_path=Config.MODEL_PATH,
        video_path=Config.VIDEO_PATH
    )
    
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    
    if not cap.isOpened():
        print("Hata: Video açilamadi!")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video bitti.")
                break
                
            processed_frame = detector.process_frame(frame)
            
            cv2.imshow('Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
        print("Hata detayı:")
        print(traceback.format_exc())  # Hata stack trace'ini yazdır
    finally:
        # İstatistikleri kaydet
        try:
            detector.cycle_analyzer.save_statistics('cycle_time_stats.json')
        except Exception as save_error:
            print(f"İstatistikler kaydedilirken hata oluştu: {str(save_error)}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
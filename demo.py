from ultralytics import YOLO
import cv2
import numpy as np

def run_inference():
    # 1. Charger le modèle
    model = YOLO("best.pt")

    source_path = "video.mp4"
    output_path = "scorpions_detection_output.mp4"

   
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialiser le VideoWriter (Codec 'mp4v' est standard pour le .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    window_name = "Detection: Ighirdm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720)


    results = model.predict(
        source=source_path, 
        conf=0.3, 
        device=0, 
        stream=True
    )

   
    BRIGHT_GREEN = (0, 255, 0)  # BGR
    LABEL_TEXT = "Ighirdm"

    for r in results:
        
        img = r.orig_img.copy()
        
        for box in r.boxes:
          
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            
            cv2.rectangle(img, (x1, y1), (x2, y2), BRIGHT_GREEN, 3)

         
            label = f"{LABEL_TEXT} {conf:.2f}"
            
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(img, (x1 - 5, y1 - th - 20), (x1 + tw, y1), BRIGHT_GREEN, -1)
            

            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        out.write(img)

        cv2.imshow(window_name, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée sous : {output_path}")

if __name__ == '__main__':
    run_inference()
from ultralytics import YOLO

def train_model():
    # 1. Load the model
    model = YOLO("yolo26n.pt") 
    model.train(
        data="data.yaml", 
        epochs=100,      
        imgsz=640,       
        device=0,
        batch=16
    )

if __name__ == '__main__':
    train_model()
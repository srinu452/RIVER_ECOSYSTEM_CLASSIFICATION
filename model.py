from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
   

# Load your 6-class model
    model = YOLO("runs/segment/yolo9class-finetuned2/weights/best.pt")

# Train it on 9-class dataset (automatically replaces head with correct output shape)
    model.train(
        data="YOLO2/data_seg2.yaml",  # <-- has nc=9 and new class names
        epochs=50,
        imgsz=960,
        batch=4,
        lr0=1e-4,
        warmup_epochs=3,
        optimizer="Adam",
        name="yolo9class-finetuned",
        device=0,
        task="segment"
    )


if __name__ == '__main__':
    freeze_support()  # Only necessary on Windows
    main()



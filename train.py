from ultralytics import YOLO


model = YOLO("yolov8n.pt")

results = model.train(
    data=r"C:\Users\mauli\Downloads\Obstacle detection.v1i.yolov8\data.yaml",
    epochs=30,
    imgsz=640,
    freeze=10,       
    batch=16,
    name="blind_assist_finetuned",
    patience=5,
    device=0          #
)

print("\n" + "="*50)
print("Training Complete!")
print("Tera model yahan save hua hai:")
print("runs/detect/blind_assist_finetuned/weights/best.pt")
print("="*50)

from ultralytics import YOLO

# Pretrained model load karo (Transfer Learning base)
model = YOLO("yolov8n.pt")

# Fine-tuning on custom obstacle detection dataset
results = model.train(
    data=r"C:\Users\mauli\Downloads\Obstacle detection.v1i.yolov8\data.yaml",
    epochs=5,
    imgsz=640,
    freeze=10,        # Early 10 layers freeze = proper Transfer Learning
    batch=16,
    name="blind_assist_finetuned",
    patience=5,
    device=0          # GPU use karega agar available ho, warna CPU
)

print("\n" + "="*50)
print("Training Complete!")
print("Tera model yahan save hua hai:")
print("runs/detect/blind_assist_finetuned/weights/best.pt")
print("="*50)
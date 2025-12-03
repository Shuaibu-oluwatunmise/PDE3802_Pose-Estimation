import torch
from ultralytics import YOLO
from pathlib import Path


def main():
    # === CONFIGURATION ===
    DATA_YAML = Path("place_holder")    # Path to your dataset YAML
    MODEL_NAME = "yolov8n.pt"            # Model to train (Nano version)
    EPOCHS = 50                          # Total training epochs
    IMG_SIZE = 640                       # Image size for training
    BATCH_SIZE = 32                      # Batch size per iteration
    PROJECT_NAME = "runs/detect"         # Directory to save results
    RUN_NAME = "yolov8n_detect_V1"       # Subfolder name for this run
    PATIENCE = 10

    # === DEVICE SELECTION ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training YOLOv8n for {EPOCHS} epochs on device: {device.upper()}")

    # === FIX WINDOWS MULTIPROCESSING ===
    torch.multiprocessing.set_sharing_strategy("file_system")

    # === VERIFY DATA FILE ===
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"❌ Dataset config not found at: {DATA_YAML}")

    # === INITIALIZE MODEL ===
    model = YOLO(MODEL_NAME)

    # === TRAIN ===
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        workers=12,
        verbose=True,
        amp=True,       
        patience=PATIENCE, 
    )

    print(f"\n✅ Training complete! Check results under '{PROJECT_NAME}/{RUN_NAME}'")


if __name__ == "__main__":
    main()
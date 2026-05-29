import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# =========================
# CONFIG (CHANGE THIS)
# =========================
video_path = "model/vi3.mp4"
model_path = "model/model.pth"
classes = ["defect", "good"]

TARGET_FPS = 0.6   #  CHANGE THIS VALUE (1, 2, 5, 10, etc.)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print(f"🚀 Running at {TARGET_FPS} FPS inference (press Q to quit)")

# =========================
# CONTROL VARIABLES
# =========================
target_delay = 1 / TARGET_FPS
last_inference_time = 0

prev_time = time.time()
frame_id = 0

# store last prediction so it stays visible
last_label = "..."
last_conf = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("✅ Video ended")
        break

    frame_id += 1

    # always show video
    display_frame = cv2.resize(frame, (800, 600))

    current_time = time.time()

    # =========================
    # RUN MODEL ONLY AT TARGET FPS
    # =========================
    if current_time - last_inference_time >= target_delay:
        last_inference_time = current_time

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)

            confidence, pred = torch.max(probs, 1)

        last_label = classes[pred.item()]
        last_conf = confidence.item() * 100

        # terminal log
        print(f"Frame {frame_id:04d} | {last_label} | {last_conf:.2f}%")

    # =========================
    # REAL VIDEO FPS (DISPLAY SPEED)
    # =========================
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # =========================
    # DISPLAY
    # =========================
    color = (0, 255, 0) if last_label == "good" else (0, 0, 255)

    cv2.putText(
        display_frame,
        f"{last_label} {last_conf:.2f}% | Display FPS: {fps:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("3D Print Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
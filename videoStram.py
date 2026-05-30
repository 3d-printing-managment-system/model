import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------- CONFIG --------
video_path = "model/v2.mp4"
model_path = "model/model.pth"
classes = ["defect", "good"]

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- MODEL --------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- VIDEO --------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("✅ Video ended")
        break

    frame_id += 1

    # Resize frame for display
    display_frame = cv2.resize(frame, (800, 600))

    # Prepare image for model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    confidence_value = confidence.item() * 100

    # Print prediction to console
    print(
        f"Frame {frame_id:04d} | Prediction: {label} | Confidence: {confidence_value:.2f}%"
    )

    # Draw prediction on video
    color = (0, 255, 0) if label == "good" else (0, 0, 255)

    cv2.putText(
        display_frame,
        f"{label} ({confidence_value:.2f}%)",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
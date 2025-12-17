import cv2
import torch
import albumentations as A
import time
import sys

from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler

# Fix Windows Unicode issue
sys.stdout.reconfigure(encoding='utf-8')

# Logger
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# Transforms
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = DETR(num_classes=3)
model.load_pretrained("checkpoints/99_model.pt")
model.to(device)
model.eval()

CLASSES = get_classes()
COLORS = get_colors()

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

# 🔴 DETECTION SPEED (SLOW)
DETECT_EVERY = 8   # ← SLOW detection
frame_id = 0

last_bboxes = []
fps_start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Run detection every 8 frames
    if frame_id % DETECT_EVERY == 0:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            transformed = transforms(image=frame_rgb)
            image = transformed["image"].unsqueeze(0).to(device)

            inference_start = time.time()
            result = model(image)
            inference_time = (time.time() - inference_start) * 1000

        probabilities = result["pred_logits"].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)

        keep_mask = max_probs > 0.5
        batch_idx, query_idx = torch.where(keep_mask)

        h, w, _ = frame.shape
        bboxes = rescale_bboxes(
            result["pred_boxes"][batch_idx, query_idx],
            (w, h)
        )

        last_bboxes = []

        for bclass, bprob, bbox in zip(
                max_classes[batch_idx, query_idx],
                max_probs[batch_idx, query_idx],
                bboxes):

            cid = int(bclass.item())
            conf = float(bprob.item())
            x1, y1, x2, y2 = map(int, bbox.tolist())

            last_bboxes.append((cid, conf, x1, y1, x2, y2))

        # Safe logging
        if last_bboxes:
            try:
                detection_handler.log_detections(
                    [{"class": CLASSES[cid], "confidence": conf}
                     for cid, conf, *_ in last_bboxes],
                    frame_id=frame_id
                )
            except UnicodeEncodeError:
                pass

        detection_handler.log_inference_time(
            inference_time, 1000 / inference_time
        )

    # Draw last detections (smooth)
    for cid, conf, x1, y1, x2, y2 in last_bboxes:
        label = f"{CLASSES[cid]} - {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      COLORS[cid], 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 10, y1),
            COLORS[cid],
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
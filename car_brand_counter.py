#--------------------------------------------------------------
#KOUKOSIAS ATHANASIOS 2025-2026 UTH
# Results at:https://github.com/4l0pix/Advanced-Machine-Learning-Techniques/blob/trafic-cam-tracker/
#--------------------------------------------------------------

import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import cv2
    import torch
    from PIL import Image
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
    from tqdm.auto import tqdm
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt") from exc


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
#paths
DATA_DIR = "grouped-cars/"
VIDEO_PATH = "samples-videos/traffic2.mp4"
OUTPUT_VIDEO = "outputs/annotated.mp4"
CKPT_PATH = "models/brand_classifier.pth"
TRAIN_STATE_PATH = "models/training_state.pth"
DETECTOR = "yolov8n.pt"

#training
EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4

#video
FRAME_SKIP = 1
YOLO_CONF = 0.4
MIN_CROP = 50
CLASSES = [2]
ROI = None
COLORS = [(0, 200, 80), (0, 140, 255), (220, 60, 60), (180, 0, 255), (255, 220, 0), (0, 200, 200)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_video(path):
    #local-video-only
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def make_model(num_classes):
    #mobilenet-brand-head
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        print("pretrained MobileNet weights unavailable; using random weights", flush=True)
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def save_training_state(path, epoch, best_acc, brands, model, opt):
    #resume-checkpoint
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "brands": brands,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }, path)


def train_classifier():
    #data-augmentation
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.ImageFolder(Path(DATA_DIR) / "train", transform=train_tf)
    test_set = datasets.ImageFolder(Path(DATA_DIR) / "test", transform=val_tf)
    if train_set.classes != test_set.classes:
        raise ValueError("Train/test brand folders do not match.")
    if len(train_set.classes) < 2:
        raise ValueError(f"{DATA_DIR} must contain at least two brand folders.")

    #loaders
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = make_model(len(train_set.classes)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    start_epoch = 1
    best_acc = 0.0

    #resume-if-available
    if Path(TRAIN_STATE_PATH).exists():
        state = torch.load(TRAIN_STATE_PATH, map_location=DEVICE)
        if state["brands"] == train_set.classes:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_acc = state["best_acc"]
            print(f"resuming training from epoch {start_epoch}/{EPOCHS}", flush=True)

    print(f"device: {DEVICE}", flush=True)
    print(f"dataset: {len(train_set)} train / {len(test_set)} test | brands: {train_set.classes}", flush=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        #train-one-epoch
        model.train()
        total_loss = 0.0
        bar = tqdm(train_dl, desc=f"epoch {epoch:02d}/{EPOCHS}", unit="batch")
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * images.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        #test-one-epoch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"epoch {epoch:02d}/{EPOCHS} loss={total_loss / len(train_set):.4f} test_acc={acc:.3f}", flush=True)
        if acc >= best_acc:
            best_acc = acc
            Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"brands": train_set.classes, "model": model.state_dict()}, CKPT_PATH)
        save_training_state(TRAIN_STATE_PATH, epoch, best_acc, train_set.classes, model, opt)

    print(f"saved classifier: {CKPT_PATH} best_test_acc={best_acc:.3f}", flush=True)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model"])
    model.eval()
    return model, train_set.classes


def load_classifier():
    #resume-or-load-best
    if Path(TRAIN_STATE_PATH).exists():
        state = torch.load(TRAIN_STATE_PATH, map_location=DEVICE)
        if state.get("epoch", 0) < EPOCHS:
            return train_classifier()
    if Path(CKPT_PATH).exists():
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model = make_model(len(ckpt["brands"])).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"loaded classifier: {CKPT_PATH}", flush=True)
        return model, ckpt["brands"]
    return train_classifier()


def in_roi(box, roi):
    #area-filter
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def classify_crop(model, brands, tfm, crop):
    #brand-probability
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = tfm(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    conf, idx = probs.max(0)
    return brands[idx.item()], float(conf)


def best_vote(scores):
    #stable-track-brand
    brand, score = max(scores.items(), key=lambda item: item[1])
    return brand, score / sum(scores.values())


def draw_label(frame, text, x, y, color):
    #box-label
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y = max(0, y - h - 8)
    cv2.rectangle(frame, (x, y), (x + w + 6, y + h + 8), color, -1)
    cv2.putText(frame, text, (x + 3, y + h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def process_video(model, brands):
    #open-video
    video = prepare_video(VIDEO_PATH)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")

    yolo = YOLO(DETECTOR)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    #crop-transform
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    colors = {brand: COLORS[i % len(COLORS)] for i, brand in enumerate(brands)}
    votes = defaultdict(lambda: defaultdict(float))
    frame_no = processed = 0
    frame_skip = max(1, FRAME_SKIP)

    #detect-track-annotate
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        annotated = frame.copy()

        if ROI:
            cv2.rectangle(annotated, tuple(ROI[:2]), tuple(ROI[2:]), (255, 255, 255), 2)
        if frame_no % frame_skip == 0:
            processed += 1
            result = yolo.track(frame, persist=True, conf=YOLO_CONF, classes=CLASSES, tracker="bytetrack.yaml", verbose=False)[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP or not in_roi((x1, y1, x2, y2), ROI):
                        continue
                    brand, conf = classify_crop(model, brands, tfm, frame[y1:y2, x1:x2])
                    votes[int(track_id)][brand] += conf
                    brand, stable_prob = best_vote(votes[int(track_id)])
                    color = colors[brand]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    draw_label(annotated, f"#{int(track_id)} {brand} {stable_prob:.0%}", x1, y1, color)

        counts = Counter(best_vote(score)[0] for score in votes.values())
        #live-count-panel
        cv2.rectangle(annotated, (5, 5), (250, 28 + 22 * len(brands)), (0, 0, 0), -1)
        for i, brand in enumerate(brands):
            cv2.putText(annotated, f"{brand}: {counts[brand]}", (12, 27 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[brand], 2)
        writer.write(annotated)
        if processed and processed % 30 == 0:
            print(f"frame {frame_no}/{total_frames} counts={dict(counts)}", flush=True)

    cap.release()
    writer.release()

    #save-results
    counts = Counter(best_vote(score)[0] for score in votes.values())
    data = {
        "total_unique_cars": sum(counts.values()),
        "brands": dict(counts),
        "tracks": {str(k): best_vote(v)[0] for k, v in votes.items()},
    }
    json_path = str(Path(OUTPUT_VIDEO).with_suffix(".json"))
    Path(json_path).write_text(json.dumps(data, indent=2))
    print(f"annotated video: {OUTPUT_VIDEO}", flush=True)
    print(f"counts json: {json_path}", flush=True)
    print(json.dumps(data["brands"], indent=2), flush=True)


def main():
    #validate-inputs
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f"Grouped dataset not found: {DATA_DIR}")
    if not (Path(DATA_DIR) / "train").is_dir() or not (Path(DATA_DIR) / "test").is_dir():
        raise FileNotFoundError(f"{DATA_DIR} must contain train/ and test/ folders. Run group_by_brand.py first.")
    model, brands = load_classifier()
    process_video(model, brands)


if __name__ == "__main__":
    main()

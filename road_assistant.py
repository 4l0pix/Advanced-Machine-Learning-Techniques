#!/usr/bin/env python3

#imports
import argparse
import csv
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


#gtsrb id to label map
GTSRB_LABELS = {
    0: "speed_limit_20",
    1: "speed_limit_30",
    2: "speed_limit_50",
    3: "speed_limit_60",
    4: "speed_limit_70",
    5: "speed_limit_80",
    6: "end_speed_limit_80",
    7: "speed_limit_100",
    8: "speed_limit_120",
    9: "no_passing",
    10: "no_passing_trucks",
    11: "priority_road",
    12: "priority_next_intersection",
    13: "yield",
    14: "stop",
    15: "no_vehicles",
    16: "trucks_prohibited",
    17: "no_entry",
    18: "general_danger",
    19: "curve_left",
    20: "curve_right",
    21: "double_curve",
    22: "bumpy_road",
    23: "slippery_road",
    24: "road_narrows_right",
    25: "road_work",
    26: "traffic_signals",
    27: "pedestrians_crossing",
    28: "children_crossing",
    29: "bicycles_crossing",
    30: "ice_snow",
    31: "animal_crossing",
    32: "end_all_restrictions",
    33: "turn_right_ahead",
    34: "turn_left_ahead",
    35: "ahead_only",
    36: "go_straight_or_right",
    37: "go_straight_or_left",
    38: "keep_right",
    39: "keep_left",
    40: "roundabout",
    41: "end_no_passing",
    42: "end_no_passing_trucks",
    43: "not_traffic_sign",
}

#labels ignored so false detections do not become signs
NON_SIGN_LABELS = {
    "person",
    "pedestrian",
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "bus",
    "truck",
    "van",
    "train",
    "motorcycle",
    "bicycle",
    "bike",
    "traffic light",
    "window",
    "door",
    "building",
    "advertisement",
    "billboard",
    "logo",
    "license plate",
    "plate",
}

#standard transform for sign crops
SIGN_TF = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
    ]
)


#select gpu when available
def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


#find first file matching patterns
def first_match(root, patterns):
    root = Path(root)
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


#convert sign label to driver instruction
def message_for_sign(label, urgency="normal"):
    label = str(label).lower().replace(" ", "_")
    if label == "not_traffic_sign":
        return None
    speed = re.search(r"speed[_-]?limit[_-]?(\d+)", label)
    if speed:
        return f"Speed limit is {speed.group(1)} km/h. Please adjust your speed."
    if label in {"stop", "stop_sign"}:
        if urgency == "immediate":
            return "STOP sign close. Brake and prepare to halt."
        return "Stop sign ahead. Prepare to halt."
    if label in {"yield", "give_way"}:
        return "Yield ahead. Slow down and give way."
    if label == "no_entry":
        return "Wrong way / no entry. Do not enter."
    if label in {"general_danger", "danger", "warning"}:
        return "Potential risks ahead. Please pay close attention."
    if label == "animal_crossing":
        return "Animal crossing zone. Watch the road edges."
    if label in {"pedestrians_crossing", "pedestrian_crossing"}:
        return "Pedestrian crossing ahead. Slow down and scan the road."
    if label == "children_crossing":
        return "Children crossing area. Slow down and pay close attention."
    if label == "road_work":
        return "Road works ahead. Reduce speed and watch for workers."
    if label == "traffic_signals":
        return "Traffic signals ahead. Be ready to stop."
    if label == "slippery_road":
        return "Slippery road warning. Reduce speed and avoid sudden braking."
    if label in {"bumpy_road", "road_narrows_right", "curve_left", "curve_right", "double_curve"}:
        return "Road hazard ahead. Reduce speed and pay close attention."
    if label == "bicycles_crossing":
        return "Bicycle crossing ahead. Slow down and watch both sides."
    if label in {"no_passing", "no_passing_trucks"}:
        return "No passing zone. Stay in lane."
    if label in {"keep_right", "keep_left", "ahead_only", "turn_right_ahead", "turn_left_ahead", "roundabout"}:
        return f"Mandatory direction sign: {label.replace('_', ' ')}. Follow lane guidance."
    return f"Traffic sign detected: {label.replace('_', ' ')}. Please pay attention."


#reject non traffic labels from custom detectors
def is_probable_sign_label(label):
    normalized = str(label).lower().strip().replace("-", " ").replace("_", " ")
    return normalized not in NON_SIGN_LABELS


#load gtsrb images from csv or folders
class GTSRBDataset(Dataset):
    def __init__(self, root, split="Train", image_size=64, train=True):
        self.root = Path(root)
        self.items = self._discover_items(split)
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size + 8, image_size + 8)),
                    transforms.RandomCrop((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
                ]
            )
        else:
            self.tf = SIGN_TF

    def _discover_items(self, split):
        rows = []
        csv_candidates = list(self.root.rglob(f"{split}.csv")) + list(self.root.rglob("Train.csv"))
        for csv_path in csv_candidates[:1]:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    if "Path" in row and "ClassId" in row:
                        img_path = (csv_path.parent / row["Path"]).resolve()
                        if img_path.exists():
                            rows.append((img_path, int(row["ClassId"])))
            if rows:
                return rows
        split_dirs = [self.root / split, self.root / split.lower(), self.root]
        exts = {".png", ".jpg", ".jpeg", ".ppm"}
        for base in split_dirs:
            if not base.exists():
                continue
            for class_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
                if class_dir.name.isdigit():
                    class_id = int(class_dir.name)
                    for img_path in class_dir.rglob("*"):
                        if img_path.suffix.lower() in exts:
                            rows.append((img_path, class_id))
            if rows:
                return rows
        return rows

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


#build background crops for the not traffic sign class
class NegativeCropDataset(Dataset):
    def __init__(self, cityscapes_root, count=12000, image_size=64, train=True):
        self.root = Path(cityscapes_root)
        self.count = count
        self.image_size = image_size
        self.label = 43
        self.paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            self.paths.extend(self.root.rglob(pattern))
        self.paths = [p for p in self.paths if "label" not in p.name.lower() and "mask" not in p.name.lower()]
        if not self.paths:
            print("warning: no Cityscapes/background images found, negative class will be empty")
        jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03)
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                jitter if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
            ]
        )

    def __len__(self):
        return self.count if self.paths else 0

    def __getitem__(self, idx):
        path = random.choice(self.paths)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        low = max(32, min(w, h) // 12)
        high = max(40, min(w, h) // 3)
        low, high = min(low, high), max(low, high)
        crop_size = min(random.randint(low, high), w, h)
        x1 = random.randint(0, max(0, w - crop_size))
        y1 = random.randint(0, max(0, h - crop_size))
        crop = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        return self.tf(crop), self.label


#compact cnn used to classify gtsrb sign crops
class SmallGTSRBCNN(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


#combine gtsrb signs and cityscapes negative crops
def make_classifier_loaders(gtsrb_root, cityscapes_root, val_fraction=0.15, batch_size=128, negative_count=12000):
    sign_dataset = GTSRBDataset(gtsrb_root, split="Train", train=True)
    if len(sign_dataset) == 0:
        raise FileNotFoundError(f"no GTSRB images found under {gtsrb_root}")
    val_count = max(1, int(len(sign_dataset) * val_fraction))
    train_count = len(sign_dataset) - val_count
    generator = torch.Generator().manual_seed(42)
    train_signs, val_signs = torch.utils.data.random_split(sign_dataset, [train_count, val_count], generator=generator)
    val_base = GTSRBDataset(gtsrb_root, split="Train", train=False)
    val_signs = torch.utils.data.Subset(val_base, val_signs.indices)
    neg_train = NegativeCropDataset(cityscapes_root, count=negative_count, train=True)
    neg_val = NegativeCropDataset(cityscapes_root, count=max(1000, negative_count // 5), train=False)
    train_dataset = torch.utils.data.ConcatDataset([train_signs, neg_train])
    val_dataset = torch.utils.data.ConcatDataset([val_signs, neg_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"training samples: {len(train_dataset)} validation samples: {len(val_dataset)} classes: 44")
    return train_loader, val_loader


@torch.no_grad()
#measure validation accuracy and negative class accuracy
def evaluate_classifier(model, loader, device):
    model.eval()
    correct, total = 0, 0
    negative_correct, negative_total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        neg_mask = y == 43
        if neg_mask.any():
            negative_correct += (pred[neg_mask] == y[neg_mask]).sum().item()
            negative_total += neg_mask.sum().item()
    return correct / max(1, total), negative_correct / max(1, negative_total)


#train the 44 class sign classifier and save best checkpoint
def train_gtsrb_classifier(gtsrb_root, cityscapes_root, output_path, epochs=20, batch_size=128, negative_count=12000):
    device = device_name()
    train_loader, val_loader = make_classifier_loaders(gtsrb_root, cityscapes_root, batch_size=batch_size, negative_count=negative_count)
    model = SmallGTSRBCNN(num_classes=44).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_val = 0.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        val_acc, neg_acc = evaluate_classifier(model, val_loader, device)
        train_acc = correct / max(1, total)
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / max(1, total):.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} neg_acc={neg_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "num_classes": 44, "labels": GTSRB_LABELS}, output_path)
            print(f"saved best checkpoint: {output_path}")
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


#load new or legacy gtsrb classifier checkpoint
def load_gtsrb_classifier(path):
    path = Path(path)
    if not path.exists():
        return None
    device = device_name()
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        num_classes = int(checkpoint.get("num_classes", 44))
        state = checkpoint["model_state"]
    else:
        num_classes = 43
        state = checkpoint
    model = SmallGTSRBCNN(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


#wrap cityscapes segformer road and sidewalk segmentation
class RoadSegmenter:
    def __init__(self, model_name_or_path):
        device = device_name()
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.road_ids = [i for i, name in labels.items() if name == "road"]
        self.sidewalk_ids = [i for i, name in labels.items() if name == "sidewalk"]
        if not self.road_ids:
            raise ValueError("segmentation model does not expose a road class")

    @torch.no_grad()
    def predict_masks(self, frame_bgr):
        device = device_name()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(device)
        logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        road = np.isin(pred, self.road_ids)
        sidewalk = np.isin(pred, self.sidewalk_ids) if self.sidewalk_ids else np.zeros_like(road, dtype=bool)
        return road, sidewalk


#estimate driving corridor width at frame height
def driving_corridor_bounds(y, h, w, horizon=0.45, top_width=0.18, bottom_width=0.78):
    horizon_y = h * horizon
    if y <= horizon_y:
        width_ratio = top_width
    else:
        t = min(1.0, (y - horizon_y) / max(1.0, h - horizon_y))
        width_ratio = top_width + t * (bottom_width - top_width)
    half = 0.5 * width_ratio * w
    return int(w / 2 - half), int(w / 2 + half)


#decide if person base is on road and not on the sidewalk
def is_pedestrian_dangerous(box, road_mask, sidewalk_mask, frame_shape, args):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    box_h = max(1, y2 - y1)
    if box_h / h < args.min_person_height_ratio:
        return False, 0.0, 0.0
    base_y1 = max(y1, int(y2 - 0.12 * box_h))
    base_x1 = int(x1 + 0.20 * (x2 - x1))
    base_x2 = int(x2 - 0.20 * (x2 - x1))
    road_region = road_mask[base_y1:y2 + 1, base_x1:base_x2 + 1]
    sidewalk_region = sidewalk_mask[base_y1:y2 + 1, base_x1:base_x2 + 1]
    road_overlap = float(road_region.mean()) if road_region.size else 0.0
    sidewalk_overlap = float(sidewalk_region.mean()) if sidewalk_region.size else 0.0
    foot_x = (x1 + x2) / 2
    left, right = driving_corridor_bounds(y2, h, w, args.driving_corridor_horizon, args.driving_corridor_top_width, args.driving_corridor_bottom_width)
    in_corridor = left <= foot_x <= right
    corridor_ok = True if not args.ped_alert_require_driving_corridor else in_corridor
    clearly_sidewalk = sidewalk_overlap >= args.sidewalk_overlap_safe_threshold and sidewalk_overlap >= road_overlap
    on_street = road_overlap >= args.road_overlap_threshold and road_overlap > sidewalk_overlap
    return bool(on_street and not clearly_sidewalk and corridor_ok), road_overlap, sidewalk_overlap


#find possible traffic sign regions from red blue yellow colors
def sign_candidates_by_color(frame_bgr, args):
    min_area = int(frame_bgr.shape[0] * frame_bgr.shape[1] * args.sign_proposal_min_area_ratio)
    max_area = int(frame_bgr.shape[0] * frame_bgr.shape[1] * args.sign_proposal_max_area_ratio)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 50), (135, 255, 255))
    yellow = cv2.inRange(hsv, (15, 60, 80), (40, 255, 255))
    mask = red1 | red2 | blue | yellow
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = frame_bgr.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(1, bh)
        bottom_ratio = (y + bh) / max(1, h)
        fill_ratio = area / max(1, bw * bh)
        if 0.45 <= aspect <= 1.85 and 0.08 <= fill_ratio <= 0.92 and bottom_ratio <= args.sign_proposal_max_bottom_ratio:
            pad = int(0.12 * max(bw, bh))
            boxes.append((max(0, x - pad), max(0, y - pad), min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)))
    return boxes


@torch.no_grad()
#classify a candidate sign crop and reject background
def classify_sign_crop(frame_bgr, box, classifier, args):
    if classifier is None:
        return None, 0.0, 0.0
    device = device_name()
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None, 0.0, 0.0
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = SIGN_TF(pil).unsqueeze(0).to(device)
    prob = torch.softmax(classifier(x), dim=1)[0]
    top2 = torch.topk(prob, k=2)
    conf = float(top2.values[0].item())
    margin = float((top2.values[0] - top2.values[1]).item())
    cls = int(top2.indices[0].item())
    if conf < args.sign_annotation_min_prob or margin < args.sign_annotation_min_margin:
        return None, conf, margin
    label = GTSRB_LABELS.get(cls, "not_traffic_sign")
    if label == "not_traffic_sign":
        return None, conf, margin
    return label, conf, margin


#keep alerts stable across multiple frames
class AlertFilter:
    def __init__(self, confirm_frames=3):
        self.confirm_frames = confirm_frames
        self.counts = defaultdict(int)

    def update(self, events):
        active = {event["key"] for event in events}
        for key in list(self.counts):
            if key not in active:
                self.counts[key] = 0
        confirmed = []
        for event in events:
            self.counts[event["key"]] += 1
            if self.counts[event["key"]] >= self.confirm_frames:
                confirmed.append(event)
        return confirmed


#orchestrate detection segmentation logic and drawing
class DriverAssistant:
    def __init__(self, args, sign_classifier=None):
        self.args = args
        self.sign_classifier = sign_classifier
        self.yolo = YOLO(args.yolo_model)
        self.sign_yolo = YOLO(args.custom_sign_weights) if args.custom_sign_weights else None
        self.segmenter = RoadSegmenter(args.segformer_model)
        self.alert_filter = AlertFilter(args.alert_confirm_frames)

#draw a single box and label on frame
    def draw_box(self, frame, box, label, color):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

#draw confirmed driver warnings at the top
    def overlay_messages(self, frame, messages):
        if not messages:
            return
        h, w = frame.shape[:2]
        unique = []
        seen = set()
        for event in sorted(messages, key=lambda e: e.get("priority", 5)):
            if event["message"] not in seen:
                seen.add(event["message"])
                unique.append(event["message"])
        panel_h = min(h - 20, 24 + 34 * len(unique))
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (w - 12, 12 + panel_h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        for i, msg in enumerate(unique):
            y = 44 + i * 34
            color = (40, 40, 255) if msg.startswith("CRITICAL") or msg.startswith("STOP") else (0, 255, 255)
            cv2.putText(frame, msg[:110], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

#detect people and coco stop signs
    def detect_people_and_stop_signs(self, frame, road_mask, sidewalk_mask):
        events = []
        result = self.yolo.predict(frame, conf=self.args.confidence, verbose=False)[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id)).lower()
            xyxy = box.xyxy[0].detach().cpu().numpy()
            conf = float(box.conf.item())
            if label == "person":
                dangerous, road_overlap, sidewalk_overlap = is_pedestrian_dangerous(xyxy, road_mask, sidewalk_mask, frame.shape, self.args)
                if dangerous:
                    events.append({"key": "pedestrian_on_road", "message": "CRITICAL: Pedestrian on road. Brake and pay close attention!", "priority": 0, "box": xyxy})
                    self.draw_box(frame, xyxy, f"person on street {conf:.2f} road={road_overlap:.2f}", (0, 0, 255))
                else:
                    self.draw_box(frame, xyxy, f"person no alert {conf:.2f} road={road_overlap:.2f} side={sidewalk_overlap:.2f}", (80, 180, 80))
            elif label == "stop sign":
                box_h = (xyxy[3] - xyxy[1]) / frame.shape[0]
                urgency = "immediate" if box_h > 0.10 else "normal"
                msg = message_for_sign("stop", urgency=urgency)
                events.append({"key": "stop_sign", "message": msg, "priority": 1, "box": xyxy})
                self.draw_box(frame, xyxy, f"stop sign {conf:.2f} | {msg[:48]}", (0, 255, 255))
        return events

#detect signs from custom yolo or color candidates
    def detect_custom_signs(self, frame):
        events = []
        if self.sign_yolo is not None:
            result = self.sign_yolo.predict(frame, conf=self.args.confidence, verbose=False)[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                if not is_probable_sign_label(label):
                    continue
                xyxy = box.xyxy[0].detach().cpu().numpy()
                conf = float(box.conf.item())
                msg = message_for_sign(label)
                if msg is None:
                    continue
                events.append({"key": f"sign_{label}", "message": msg, "priority": 2, "box": xyxy})
                self.draw_box(frame, xyxy, f"{label} {conf:.2f} | {msg[:48]}", (255, 200, 0))
            return events
        if self.sign_classifier is None or not self.args.use_color_sign_proposals:
            return events
        candidates = sign_candidates_by_color(frame, self.args)
        if self.args.debug_sign_proposals:
            for candidate in candidates:
                self.draw_box(frame, candidate, "sign candidate", (160, 160, 160))
        for xyxy in candidates:
            label, conf, margin = classify_sign_crop(frame, xyxy, self.sign_classifier, self.args)
            if label is None:
                continue
            msg = message_for_sign(label)
            if msg is None:
                continue
            self.draw_box(frame, xyxy, f"{label} {conf:.2f} | {msg[:48]}", (255, 200, 0))
            if conf >= self.args.sign_alert_min_prob and margin >= self.args.sign_alert_min_margin:
                events.append({"key": f"sign_{label}", "message": msg, "priority": 2, "box": xyxy})
        return events

#run all models and draw one annotated frame
    def process_frame(self, frame):
        road_mask, sidewalk_mask = self.segmenter.predict_masks(frame)
        if self.args.show_road_overlay:
            road_overlay = np.zeros_like(frame)
            road_overlay[road_mask] = (70, 70, 70)
            frame = cv2.addWeighted(frame, 0.88, road_overlay, 0.12, 0)
        events = []
        events.extend(self.detect_people_and_stop_signs(frame, road_mask, sidewalk_mask))
        events.extend(self.detect_custom_signs(frame))
        confirmed = self.alert_filter.update(events)
        self.overlay_messages(frame, confirmed)
        return frame, confirmed


#protect against overwriting the input video
def safe_output_path(video_path, output_path):
    video_path = Path(video_path)
    output_path = Path(output_path)
    if output_path == video_path:
        output_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


#process video and write annotated mp4
def process_video(args, sign_classifier):
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_path = safe_output_path(video_path, args.output)
    assistant = DriverAssistant(args, sign_classifier=sign_classifier)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames
    if args.max_video_seconds is not None:
        max_frames = min(max_frames, int(args.max_video_seconds * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps / args.frame_stride, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video writer: {output_path}")
    print(f"writing annotated video to: {output_path}")
    start = time.time()
    processed = 0
    frame_idx = 0
    try:
        while cap.isOpened() and frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                annotated, alerts = assistant.process_frame(frame)
                writer.write(annotated)
                processed += 1
                if processed % 25 == 0:
                    elapsed = time.time() - start
                    active = [a["message"] for a in alerts]
                    print(f"processed {processed} frames {processed / max(1e-6, elapsed):.2f} fps active alerts: {active}")
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    print(f"done: {output_path}")
    return output_path


#parse command line configuration
def parse_args():
    parser = argparse.ArgumentParser(description="annotate road video with pedestrians, street signs, and driver instructions")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="annotated_output.mp4")
    parser.add_argument("--gtsrb-root", default=None)
    parser.add_argument("--cityscapes-root", default=None)
    parser.add_argument("--classifier-path", default="gtsrb_classifier.pt")
    parser.add_argument("--train-classifier", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negative-count", type=int, default=12000)
    parser.add_argument("--custom-sign-weights", default=None)
    parser.add_argument("--segformer-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--alert-confirm-frames", type=int, default=3)
    parser.add_argument("--max-video-seconds", type=float, default=None)
    parser.add_argument("--use-color-sign-proposals", action="store_true")
    parser.add_argument("--sign-annotation-min-prob", type=float, default=0.75)
    parser.add_argument("--sign-annotation-min-margin", type=float, default=0.08)
    parser.add_argument("--sign-alert-min-prob", type=float, default=0.95)
    parser.add_argument("--sign-alert-min-margin", type=float, default=0.45)
    parser.add_argument("--sign-proposal-max-bottom-ratio", type=float, default=0.85)
    parser.add_argument("--sign-proposal-min-area-ratio", type=float, default=0.00008)
    parser.add_argument("--sign-proposal-max-area-ratio", type=float, default=0.045)
    parser.add_argument("--debug-sign-proposals", action="store_true")
    parser.add_argument("--road-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--sidewalk-overlap-safe-threshold", type=float, default=0.20)
    parser.add_argument("--ped-alert-require-driving-corridor", action="store_true")
    parser.add_argument("--min-person-height-ratio", type=float, default=0.07)
    parser.add_argument("--driving-corridor-bottom-width", type=float, default=0.78)
    parser.add_argument("--driving-corridor-top-width", type=float, default=0.18)
    parser.add_argument("--driving-corridor-horizon", type=float, default=0.45)
    parser.add_argument("--show-road-overlay", action="store_true")
    return parser.parse_args()


#load or train classifier and start inference
def main():
    args = parse_args()
    print(f"using device: {device_name()}")
    sign_classifier = None
    classifier_path = Path(args.classifier_path)
    if args.train_classifier:
        if args.gtsrb_root is None:
            raise ValueError("--gtsrb-root is required when --train-classifier is used")
        if args.cityscapes_root is None:
            raise ValueError("--cityscapes-root is required when --train-classifier is used")
        sign_classifier = train_gtsrb_classifier(args.gtsrb_root, args.cityscapes_root, classifier_path, args.epochs, args.batch_size, args.negative_count)
    else:
        sign_classifier = load_gtsrb_classifier(classifier_path)
        print(f"classifier loaded: {sign_classifier is not None}")
    process_video(args, sign_classifier)


if __name__ == "__main__":
    main()

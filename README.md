# Car Brand Counter

Detect cars in a local `.mp4` video, classify each tracked car by brand, and save an annotated output video with bounding boxes, track ids, brand probabilities, and live brand counts.

The project uses two scripts:

```text
group_by_brand.py      # prepares train/test data once
car_brand_counter.py   # trains/loads the model and processes the video
```

## Data Layout

The main script expects the dataset to already be grouped into train/test folders:

```text
grouped-cars/
  train/
    Toyota/
    Honda/
    BMW/
  test/
    Toyota/
    Honda/
    BMW/
```

Create this layout from the original raw dataset with:

```bash
python3 group_by_brand.py 60k-car-dataset --dst grouped-cars
```

`group_by_brand.py` uses:

```python
TRAIN_RATIO = 0.80
SEED = 42
```

So each brand is split into 80% train and 20% test in a reproducible way.

## Environment

Install dependencies directly:

```bash
pip install -r requirements.txt
```

Or create the Python 3.12 CUDA-ready virtual environment:

```bash
bash setup_venv.sh
source .venv/bin/activate
```

Check CUDA driver visibility with:

```bash
nvidia-smi
```

If `nvidia-smi` fails, PyTorch can still install, but CUDA training will not work until the NVIDIA driver is fixed.

## Run

Edit the constants at the top of `car_brand_counter.py`:

```python
DATA_DIR = "grouped-cars/"
VIDEO_PATH = "traffic.mp4"
OUTPUT_VIDEO = "outputs/annotated.mp4"
EPOCHS = 8
BATCH_SIZE = 32
```

Then run:

```bash
python3 car_brand_counter.py
```

There are no command-line arguments in the main script now. The script only uses local video files; it does not download videos from the internet.

## Outputs

The main script writes:

```text
outputs/annotated.mp4
outputs/annotated.json
models/brand_classifier.pth
models/training_state.pth
```

`outputs/annotated.mp4` contains the original video with:

```text
bounding boxes around detected cars
track ids
predicted brand labels
brand probability per tracked car
live counts per brand
```

`outputs/annotated.json` contains:

```json
{
  "total_unique_cars": 0,
  "brands": {},
  "tracks": {}
}
```

## Checkpoints

Two checkpoints are saved:

```text
models/brand_classifier.pth
models/training_state.pth
```

`brand_classifier.pth` stores the best model by test accuracy. This is the model used later for video processing.

`training_state.pth` stores the last completed epoch, model weights, optimizer state, brand names, and best accuracy. If training crashes or is interrupted before `EPOCHS`, rerun:

```bash
python3 car_brand_counter.py
```

Training resumes from the next epoch instead of starting from zero.

## Architecture

The application has two learning components:

```text
YOLOv8 + ByteTrack        -> car detection and tracking
MobileNetV3-Small         -> brand classification from car crops
```

The full pipeline is:

```text
local mp4 video
  -> read frame with OpenCV
  -> detect cars with YOLOv8
  -> assign stable track ids with ByteTrack
  -> crop each detected car
  -> classify crop with MobileNetV3-Small
  -> accumulate brand confidence per track
  -> draw box, id, brand, probability
  -> write annotated frame to output video
  -> export final brand counts to JSON
```

### Dataset Preparation

`group_by_brand.py` reads raw car folders or files and extracts the brand from the first token in the name:

```text
Toyota_Corolla_2017  -> Toyota
Honda Civic 2018     -> Honda
BMW_3Series_2020     -> BMW
```

It groups items per brand and splits each brand independently into train/test. Splitting per brand matters because it keeps the train and test sets balanced across brands.

### Brand Classifier

The classifier is `MobileNetV3-Small` from `torchvision.models`.

The original ImageNet classification head is replaced with:

```python
nn.Linear(model.classifier[-1].in_features, number_of_brands)
```

This gives one softmax output per brand folder in:

```text
grouped-cars/train/<brand>
```

Training uses:

```text
CrossEntropyLoss
AdamW optimizer
ImageNet normalization
random resized crop
horizontal flip
color jitter
tqdm progress bars
```

The test transform is deterministic:

```text
resize
center crop
normalize
```

This keeps test accuracy stable between runs.

### Detection And Tracking

The video side uses Ultralytics YOLO:

```python
yolo.track(..., tracker="bytetrack.yaml")
```

YOLO detects objects in each frame. The script filters detections using:

```python
CLASSES = [2]
```

In COCO labels, class `2` is `car`. To include buses and trucks, edit:

```python
CLASSES = [2, 5, 7]
```

ByteTrack gives each detected car a stable `track_id`. That prevents counting the same car again on every frame.

### Brand Voting

Every time a tracked car appears, its crop is classified by the MobileNet model. The script does not trust just one frame. Instead, it accumulates confidence scores per track:

```text
track 3:
  Toyota += 0.72
  Toyota += 0.81
  Honda  += 0.20
```

The winning brand is the brand with the largest total confidence for that track. The displayed probability is:

```text
winning brand score / total score for that track
```

This makes the result more stable when one crop is blurry or partially occluded.

### Counting Logic

The final count is based on unique ByteTrack ids, not raw detections. If the same car appears for 100 frames, it is still counted once.

The JSON output maps each track id to its resolved brand:

```json
{
  "tracks": {
    "1": "Toyota",
    "2": "BMW"
  }
}
```

Then the script counts how many unique tracks belong to each brand.

### Area Filtering

By default:

```python
ROI = None
```

This means every detected car is counted. To count cars only inside a region, set:

```python
ROI = [100, 150, 900, 700]
```

The script checks whether the center of the bounding box is inside this rectangle.

### Annotation

For each tracked car, the output frame receives:

```text
colored bounding box
#track_id
brand name
stable brand probability
```

A live panel in the top-left corner shows current unique-car counts per brand.

## Important Constants

Edit these in `car_brand_counter.py`:

```python
DATA_DIR = "grouped-cars/"
VIDEO_PATH = "traffic.mp4"
OUTPUT_VIDEO = "outputs/annotated.mp4"
CKPT_PATH = "models/brand_classifier.pth"
TRAIN_STATE_PATH = "models/training_state.pth"
DETECTOR = "yolov8n.pt"
EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4
FRAME_SKIP = 1
YOLO_CONF = 0.4
MIN_CROP = 50
CLASSES = [2]
ROI = None
```

## Notes

The first run can take time because it trains the brand classifier.

After training finishes, future runs load `models/brand_classifier.pth` and go directly to video processing.

If you change the dataset brands, delete old checkpoints before retraining:

```bash
rm models/brand_classifier.pth models/training_state.pth
```

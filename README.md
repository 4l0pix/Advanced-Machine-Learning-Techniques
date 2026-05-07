# Car Brand Counter

Detect cars in a video, classify each tracked car by brand, and save an annotated output video with boxes, track numbers, brand probabilities, and live counts.

## Data

First group the image dataset by brand. The app expects this layout:

```text
grouped-cars/train/Toyota/...
grouped-cars/train/Honda/...
grouped-cars/test/Toyota/...
grouped-cars/test/Honda/...
```

If your original dataset is still raw model folders/files, run this once. It creates an 80/20 train/test split:

```bash
python group_by_brand.py path/to/raw_dataset --dst grouped-cars
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

For a Python 3.12 CUDA-ready local venv, use:

```bash
bash setup_venv.sh
source .venv/bin/activate
```

If CUDA is not detected after setup, check the NVIDIA driver first:

```bash
nvidia-smi
```

Train the classifier if needed, then annotate one video:

```bash
python car_brand_counter.py
```

If training is interrupted, rerun the same command. It resumes from `models/training_state.pth`.

Edit these constants at the top of `car_brand_counter.py` when your paths differ:

```python
DATA_DIR = "grouped-cars/"
VIDEO_PATH = "traffic.mp4"
EPOCHS = 8
```

Outputs:

```text
outputs/annotated.mp4
outputs/annotated.json
```

Use `ROI` and `CLASSES` constants in `car_brand_counter.py` if you want an area filter or more vehicle classes.

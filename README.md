# Signature & Stamp Detection API

**In short:** an API that checks whether an uploaded certificate (image or PDF) has been signed and stamped. I built it for a university application portal, where students upload internship certificates.

- **The headline:** my first model scored 0.88, but that number was a lie. The same certificates had ended up in both the training and test data, so the model was being tested on images it had already seen.
- **The fix:** I found 103 near-duplicate pairs using image fingerprints (perceptual hashing), rebuilt the split so no copy crosses between training and testing, and retrained from scratch.
- **The honest result:** on a held-out test set it has never seen, the model scores **0.79 mAP@50**. On the validation set it finds **93% of all marks** if you ignore the labels; most of its mistakes are calling a signature a "signature over a stamp", not missing marks.
- **Built on my own data:** 555 images, about 350 of them real certificates I photographed and labelled by hand.

![Example prediction on a certificate template](reports/example_prediction.jpg)

*Example on a certificate template (placeholder names, no personal data). Real certificates are not shown.*

**Built with:** Python, YOLO26 (Ultralytics), FastAPI, PyMuPDF, OpenCV, Roboflow, Docker

---

## Why I built this

A university portal needed to check that uploaded internship certificates were actually signed and stamped before a person reviewed them. Doing that by eye for hundreds of documents is slow and boring, so I trained a model to spot three things:

- **Signatures**
- **Stamps**
- **"Mix"**: a signature written over a stamp

## The story: my first score was too good to be true

### What went wrong

My first model (YOLOv8s) scored **0.88 mAP@50** on the validation set. That looked great, until I checked the data.

While building the dataset, I'd often saved several versions of the same certificate: recoloured, greyscaled, brightened, or with the signature removed. These were uploaded as separate images and then split randomly into train, validation and test. So the same certificate could end up in both training and testing.

That's like giving students the exam questions the day before. The score looks great, but it doesn't tell you how they'd do on new questions.

### How I found it

I wrote `tools/check_leakage.py`, which gives every image a fingerprint (pHash and dHash) and compares every image in one split against every image in the others. Two images that look nearly the same get nearly the same fingerprint.

It found **103 near-duplicate pairs** across the splits. Roughly 15–30% of the validation images had a copy in the training set.

### How I fixed it

With `tools/make_clean_split.py`:

1. **Group the copies.** Any images whose fingerprints differ by 8 bits or fewer go in the same group (430 groups; the biggest has 11 images).
2. **Split by group, not by image.** Whole groups go into train, validation or test (70/20/10), so all versions of a certificate stay together. I also checked the class counts were balanced.
3. **Check again.** Re-running the leakage check found **zero** duplicates across splits.
4. **Retrain from scratch.** I started from the standard pretrained model, not my old one, because the old model had already seen copies of the new test images.

`reports/split_report.csv` lists every image, where it was before, where it is now, and which group it belongs to.

### Before and after

| Model | Data | mAP@50 |
|---|---|---|
| YOLOv8s (first version) | leaky validation | 0.88 (inflated) |
| YOLO26s | clean validation | 0.91 |
| YOLO26s | **clean test** | **0.79** |

One honest caveat: the old and new rows aren't a perfect comparison, because I changed both the data split and the model. A fair comparison would retrain YOLOv8s on the clean split (it's on the to-do list).

## Results

### How to read the numbers

- **Precision:** when the model says "that's a stamp", how often it's right.
- **Recall:** of all the real stamps, how many the model found.
- **mAP@50:** a single score combining both, counting a detection as correct if its box overlaps the real one by at least half. 1.0 is perfect.

### Test set: 55 images, 61 marks, never used for any decision

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| **All** | **0.84** | **0.77** | **0.79** |
| Mix | 0.67 | 0.82 | 0.76 |
| Signature | 0.87 | 0.79 | 0.88 |
| Stamp | 0.99 | 0.71 | 0.74 |

### What the mistakes look like

On the validation set (122 marks), the model missed **only 1 mark entirely**. Almost all of its errors were picking the wrong label, mostly calling a plain signature a "mix" (18 of 66 signatures).

That makes sense: where a signature ends and "signature over a stamp" begins is partly a judgement call, even for the person labelling. And for the portal's real question, "is this certificate signed and stamped?", a mix-up between those two labels matters much less than a missed mark.

If you ignore the labels and just ask "did it find a mark?", recall on validation is **0.93**.

A few more things worth knowing:
- **Signature boxes are the least precise.** Signatures don't have sharp edges, so where exactly the box should go varies.
- **The test set is small.** One stamp is worth about 6 recall points, so differences under about 5 points are just noise.
- **Confidence threshold:** the API keeps detections with confidence 0.3 or higher, which gave the best balance of precision and recall on validation. Results barely change anywhere between 0.25 and 0.7.

Full numbers and the confusion matrix are in [`reports/evaluation.md`](reports/evaluation.md).

![F1 curve](yolo_model/yolo26s_clean/BoxF1_curve.png)
![Confusion matrix](yolo_model/yolo26s_clean/confusion_matrix.png)

## The data

| Split | Images | Signatures | Stamps | Mix |
|---|---|---|---|---|
| Train | 389 | 221 | 105 | 84 |
| Validation | 111 | 66 | 31 | 25 |
| Test | 55 | 33 | 17 | 11 |

- About 350 real internship certificates I photographed, plus about 205 certificate templates for variety in layout.
- 147 images (26%) have no marks at all. Many are copies with the signature removed. These teach the model not to fire on an empty signature line.
- Labelled by hand in Roboflow, exported at 640×640.
- **The dataset is private.** Real certificates contain names, signatures and company stamps, which are personal data under GDPR, so I can't publish them.

## Training

| Setting | Value |
|---|---|
| Starting model | YOLO26s, pretrained on COCO |
| Image size | 640 |
| Epochs | 150 (stops early after 50 with no improvement); best was epoch 111 |
| Batch size | 16 |
| Hardware | Google Colab, Tesla T4 GPU |

Speed: about 16 ms per image on a T4. Full settings are in `yolo_model/yolo26s_clean/args.yaml`.

## Run it

### With Docker (easiest)

```bash
docker build -t sign-stamp-api .
docker run -p 8000:8000 sign-stamp-api
```

Then open http://localhost:8000/docs.

### Without Docker

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/docs.

### Settings

| Variable | Default | What it does |
|---|---|---|
| `MODEL_PATH` | `yolo_model/yolo26s_clean/weights/best.pt` | Which model weights to load |
| `CONF_THRESHOLD` | `0.3` | Ignore detections below this confidence |

With Docker, pass them with `-e`, for example `docker run -p 8000:8000 -e CONF_THRESHOLD=0.5 sign-stamp-api`.

## Using the API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Check the API is running and which model is loaded |
| POST | `/detect` | Upload an image or PDF (form field `file`); get detections for every page |

By default `/detect` also returns the page with boxes drawn on it. Add `include_image=false` if you only need the detections; the response shrinks from about 200 KB to under 1 KB per page.

```bash
curl -X POST http://127.0.0.1:8000/detect -F "file=@certificate.pdf"
curl -X POST "http://127.0.0.1:8000/detect?include_image=false" -F "file=@certificate.pdf"
```

The response has one entry per page:

```json
{
  "page1": {
    "detections": [
      {"bbox": [410, 432, 161, 65], "label": "signature", "confidence": 0.91}
    ],
    "image_with_annotations": "data:image/jpeg;base64,..."
  }
}
```

`bbox` is `[x, y, width, height]` in pixels.

## Limits and responsible use

- **It's a helper, not the final check.** It misses about 7% of marks, so it should flag documents for a person to review, not reject them automatically.
- **One source of data.** It was built for one university portal. Other document styles, photos taken at steep angles, or faded stamps haven't been tested.
- **Small test set** (61 marks), so expect results to vary by a few points on new data.
- **Personal data.** Uploaded documents contain personal data. The API deletes the upload and every image it makes from it after each request, even if something fails. Any real deployment should still follow GDPR rules on keeping data.
- **Known issue:** PDF pages are rendered at 72 dpi, which may be too low for very faint signatures.

## Reproduce it

```bash
pip install pillow imagehash
python tools/check_leakage.py path/to/dataset                         # find duplicates across splits
python tools/make_clean_split.py path/to/dataset path/to/clean_split  # build the leak-free split
yolo detect train data=path/to/clean_split/data.yaml model=yolo26s.pt epochs=150 patience=50 imgsz=640 seed=0
yolo detect val model=runs/detect/train/weights/best.pt data=path/to/clean_split/data.yaml split=test
```

The leakage tools work on any image dataset in YOLO format, not just this one.

## What's in the repo

```
app.py                  the API (/health, /detect)
functions.py            PDF to images, running the model, drawing boxes
Dockerfile              run the API in one command
requirements.txt
yolo_model/
  yolo26s_clean/        model weights, training curves, confusion matrix, settings
tools/
  check_leakage.py      find near-duplicate images across splits
  make_clean_split.py   rebuild the split so duplicates stay together
reports/
  evaluation.md         every number, the threshold choice, and the error analysis
  example_prediction.jpg
  split_report.csv      old vs new split for every image
```

## What's next

- Retrain YOLOv8s on the clean split for a fair before/after comparison.
- Train on 25/50/75/100% of the data to see whether more data would help.
- Collect more real "mix" examples and hard cases (angled photos, faded stamps).
- Render PDF pages at a higher resolution and measure the difference on faint signatures.

## License

The code here is MIT (see `LICENSE`). The model is trained and run with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which has its own licence (AGPL-3.0 at the time of writing), so the model weights and any deployment follow those terms.

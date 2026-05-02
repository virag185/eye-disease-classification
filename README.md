# RetinaForge Studio

RetinaForge Studio is a local web application for classifying retinal images from the provided `eye_diseases_classification.zip` dataset.

It includes:

- Upload-based retinal image analysis
- Selectable dataset samples by disease class
- Four-class classification: cataract, diabetic retinopathy, glaucoma, normal
- Ranked confidence scores
- Image quality metrics for brightness, contrast, and sharpness
- Regional scan map visualization
- In-session analysis history
- Exportable JSON report for the current case

The generated model asset uses handcrafted visual features and a compact nearest-neighbor bank built from the dataset. The app runs fully in the browser after the local server starts.

The displayed prediction percentage is a model similarity/confidence score, not medical accuracy. Built-in dataset samples can produce very high confidence because they are close to the reference image bank. The current validation score is shown separately in the app.

## Run

From this folder:

```powershell
& "C:\Users\91797\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" app_server.py
```

Then open:

```text
http://localhost:8080/app/
```

## Rebuild The Model

```powershell
& "C:\Users\91797\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\build_model.py
```

The generated browser model is written to `app/assets/model.js`.

## Backend Endpoints

- `GET /api/health`
- `POST /api/analyze` with form field `image`
- `POST /api/analyze-sample` with JSON body `{ "path": "dataset/dataset/cataract/0_left.jpg" }`

## Note

This is for education and hackathon demonstration. It is not a medical device or a substitute for clinical diagnosis.

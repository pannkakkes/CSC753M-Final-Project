# Multimodal Emotion Recognition on IEMOCAP

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Get the dataset (IEMOCAP)

1. Fill out the release form at https://sail.usc.edu/iemocap/iemocap_release.htm
2. Once approved, download `IEMOCAP_full_release`
3. Place it at:

   ```
   <repo_root>/datasets/IEMOCAP_full_release/
       Session1/
       Session2/
       Session3/
       Session4/
       Session5/
   ```

## 3. Get the auxiliary model assets

```
<repo_root>/experiments/detector.tflite
<repo_root>/experiments/face_landmarker.task
```

Download from Google's MediaPipe model index (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).

## 4. Run order

```bash
python extract_features.py
python train_baselines.py
python train_gcn_baselines.py
python train_cross_attention.py
python missing_modality_stress_test.py
python evaluate_significance.py
python visualize_results.py
```

## 5. Where outputs land

```
cached_features/          per-utterance text/audio feature .npy files + df_all.csv index
cached_visual_features/   per-utterance visual feature .npy files
checkpoints/              fitted scalers, speaker-norm stats, and trained fusion model weights
results/                  per-model CSVs, significance tables, misclassification summary
figures/                  confusion matrices and t-SNE figures
```
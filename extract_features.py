# Extracts and caches text, audio, and visual features for every IEMOCAP utterance, and writes the resulting utterance-level index to `df_all.csv` for downstream training scripts to load. See `extract_features.py` for details.

# Mediapipe must be imported before torch, otherwise the GPU backend will fail to initialize
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Standard library
from pathlib import Path
import re
import os
import time
import random
import warnings
from collections import defaultdict

# Numerics
import numpy as np
import pandas as pd

# Audio / video
import cv2
import librosa
import torchaudio
import timm
from PIL import Image
import torchvision.transforms as T

# PyTorch
import torch
import torch.nn as nn

# Transformers
from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Scikit-learn
from sklearn.preprocessing import LabelEncoder

# Progress bar
from tqdm import tqdm


# Reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Records the time taken for each stage of feature extraction, so we can report it in the paper.
class ExperimentTimer:
    def __init__(self):
        self.timings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def record(self, session, modality, stage, duration):
        self.timings[session][modality][stage].append(duration)

    def get_summary(self):
        summary = []
        for sess, modalities in self.timings.items():
            for mod, stages in modalities.items():
                for stage, durations in stages.items():
                    avg = np.mean(durations)
                    total = np.sum(durations)
                    summary.append({
                        "Session": sess, "Modality": mod, "Stage": stage,
                        "Avg_Sec": f"{avg:.4f}", "Total_Sec": f"{total:.2f}"
                    })
        return pd.DataFrame(summary)


timer = ExperimentTimer()


# Paths

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "IEMOCAP_full_release"
FEATURE_DIR = ROOT / "cached_features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TEXT_FEATURE_DIR = FEATURE_DIR / "text"
TEXT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_FEATURE_DIR = ROOT / "cached_visual_features"
VISUAL_FEATURE_DIR.mkdir(exist_ok=True)

AUDIO_DIM = 768
AUDIO_METHOD = "wav2vec2"

set_seed(42)

_wav2vec2_processor = None
_wav2vec2_model = None


def get_wav2vec2_model(device):
    global _wav2vec2_processor, _wav2vec2_model
    if _wav2vec2_model is None:
        model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp"
        try:
            _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_id)
            _wav2vec2_model = Wav2Vec2Model.from_pretrained(model_id).to(device)
            print(f"Loaded emotion-tuned wav2vec2: {model_id}")
        except Exception as e1:
            print(f"Warning: {e1}. Falling back to facebook/wav2vec2-base")
            _wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            _wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        _wav2vec2_model.eval()
    return _wav2vec2_processor, _wav2vec2_model


# Labels

def load_emotions(eval_path):
    id_to_emotion = {}
    with open(eval_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("["):
                parts = line.split("\t")
                if len(parts) >= 3:
                    utt_id = parts[1]
                    raw_label = parts[2]
                    if raw_label in ["ang", "hap", "sad", "neu"]:
                        id_to_emotion[utt_id.strip()] = raw_label.strip()
                    if raw_label in ["exc"]:
                        # Excitement merged into happiness (see paper §4)
                        id_to_emotion[utt_id.strip()] = "hap"
    return id_to_emotion

# Ses04F_impro04_F000 -> Ses04F
def get_speaker_from_utterance(utterance_id):
    sess_match = re.match(r'Ses(\d+)', utterance_id)
    if not sess_match:
        return None
    sess_num = sess_match.group(1)

    parts = utterance_id.split("_")
    if len(parts) >= 3 and parts[-1][0] in ("F", "M"):
        return f"Ses{sess_num}{parts[-1][0]}"

    if "_F" in utterance_id:
        return f"Ses{sess_num}F"
    elif "_M" in utterance_id:
        return f"Ses{sess_num}M"
    return None


# Audio features

def extract_audio_features(file_path, method="wav2vec2", n_mfcc=13):
    try:
        if method == "wav2vec2":
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor, model = get_wav2vec2_model(device)

            with torch.no_grad():
                inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(device)
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state       # (1, T, 768)
                features = hidden.mean(dim=1).squeeze().cpu().numpy()  # (768,)
            return features

        if method == "mfcc":
            y, sr = librosa.load(file_path, sr=None, mono=True)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            comprehensive_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            mean = np.mean(comprehensive_mfccs, axis=1)
            std = np.std(comprehensive_mfccs, axis=1)
            max_val = np.max(comprehensive_mfccs, axis=1)
            aggregated_features = np.concatenate([mean, std, max_val])
            pitch, _ = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitch)
            pitch_std = np.std(pitch)
            energy = np.mean(librosa.feature.rms(y=y))
            extra_features = np.array([pitch_mean, pitch_std, energy])
            return np.concatenate([aggregated_features, extra_features])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Visual features

# Face detector — used to crop the face from the video frame, before passing it to the landmarker and CNN.
_detector_model_path = str(ROOT / "experiments" / "detector.tflite")
_detector_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=_detector_model_path),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.1,  # low threshold: video is often low-resolution
)
face_detector_instance = vision.FaceDetector.create_from_options(_detector_options)

# Face landmarker — used to extract 478 landmarks and 52 blendshape scores from the cropped face.
_landmarker_path = str(ROOT / "experiments" / "face_landmarker.task")
_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=_landmarker_path),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.1,
    min_face_presence_confidence=0.1,
    min_tracking_confidence=0.1,
    output_face_blendshapes=True,
)
face_landmarker_instance = vision.FaceLandmarker.create_from_options(_landmarker_options)
print("Face landmarker initialized (478 landmarks + 52 blendshapes)")

CNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EfficientNet-B0
_cnn_model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0,
).to(CNN_DEVICE)
_cnn_model.eval()

# Attempt to load AffectNet-tuned weights (from the HSEmotion repo) for the CNN backbone. If not found, fall back to ImageNet weights.
_affectnet_path = os.path.expanduser("~/.hsemotion/enet_b0_8_best_afew.pt")
_affectnet_loaded = False
try:
    _raw = torch.load(_affectnet_path, weights_only=False, map_location="cpu")
    src_sd = _raw.state_dict()
    tgt_sd = _cnn_model.state_dict()
    matched = {k: v for k, v in src_sd.items()
               if k in tgt_sd and tgt_sd[k].shape == v.shape}
    tgt_sd.update(matched)
    _cnn_model.load_state_dict(tgt_sd, strict=False)
    _affectnet_loaded = True
    print(f"AffectNet backbone: {len(matched)}/{len(tgt_sd)} layers transferred")
except Exception as e:
    print(f"AffectNet transfer skipped ({e})")
    print("Using ImageNet pretrained weights")

_cnn_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with torch.no_grad():
    _dummy_out = _cnn_model(torch.zeros(1, 3, 224, 224).to(CNN_DEVICE))
CNN_EMBED_DIM = _dummy_out.shape[1]

print(f"AffectNet loaded : {_affectnet_loaded}")
print(f"CNN_EMBED_DIM    : {CNN_EMBED_DIM}   (expect 1280)")
print(f"Device           : {CNN_DEVICE}")

BLENDSHAPE_DIM = 52
EMBED_DIM_TOTAL = CNN_EMBED_DIM + BLENDSHAPE_DIM   
VISUAL_DIM = EMBED_DIM_TOTAL * 2             
print(f"VISUAL_DIM       : {VISUAL_DIM}  (CNN + blendshapes, temporal mean+std)")

# Upscales small ROIs to 640×640 for better detection, applies CLAHE and unsharp masking, and returns the preprocessed ROI.
def preprocess_for_detection(roi):
    h, w = roi.shape[:2]
    if w < 320:
        roi = cv2.resize(roi, (640, 640), interpolation=cv2.INTER_CUBIC)
    else:
        roi = cv2.resize(roi, (640, 640), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(roi, (0, 0), 3)
    roi = cv2.addWeighted(roi, 1.5, blur, -0.5, 0)
    return roi

# Returns a 52-dimensional blendshape vector for the given Mediapipe image, or a zero vector if no face was detected.
def extract_blendshapes(mp_image):
    result = face_landmarker_instance.detect(mp_image)
    if not result.face_blendshapes or len(result.face_blendshapes) == 0:
        return np.zeros(BLENDSHAPE_DIM, dtype=np.float32)

    scores = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
    if scores.shape[0] != BLENDSHAPE_DIM:
        fixed = np.zeros(BLENDSHAPE_DIM, dtype=np.float32)
        n = min(BLENDSHAPE_DIM, scores.shape[0])
        fixed[:n] = scores[:n]
        return fixed
    return scores

# Crops the speaker's face from the video frame, runs it through the landmarker and CNN, and returns a 1332-dimensional feature vector (1280 from EfficientNet-B0 + 52 blendshapes). Returns None if no face was detected or if the CNN fails to produce a valid embedding.
def extract_face_embedding_cnn(frame, speaker_side):
    if frame is None:
        return None

    h, w, _ = frame.shape
    roi = frame[:, :w // 2] if speaker_side == "left" else frame[:, w // 2:]
    roi = preprocess_for_detection(roi)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
    result = face_detector_instance.detect(mp_image)

    if not result.detections:
        return None

    best = max(result.detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bbox = best.bounding_box
    pad_x = int(bbox.width * 0.25)
    pad_y = int(bbox.height * 0.25)
    x1 = max(0, int(bbox.origin_x) - pad_x)
    y1 = max(0, int(bbox.origin_y) - pad_y)
    x2 = min(640, int(bbox.origin_x + bbox.width) + pad_x)
    y2 = min(640, int(bbox.origin_y + bbox.height) + pad_y)
    face_crop = roi[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    roi_final = cv2.resize(face_crop, (224, 224))

    # Cross-validate the detector's crop with the landmarker, to avoid false positives.
    crop_rgb = cv2.cvtColor(roi_final, cv2.COLOR_BGR2RGB)
    crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    landmark_result = face_landmarker_instance.detect(crop_image)

    if not landmark_result.face_landmarks or len(landmark_result.face_landmarks) == 0:
        return None  # detector found a "face" the landmarker doesn't agree with

    blendshape_vec = extract_blendshapes(crop_image)

    try:
        img_rgb = cv2.cvtColor(roi_final, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = _cnn_transform(img_pil).unsqueeze(0).to(CNN_DEVICE)

        with torch.no_grad():
            emb = _cnn_model(tensor).squeeze().cpu().numpy().astype(np.float32)

        if emb.shape[0] != CNN_EMBED_DIM:
            return None

        return np.concatenate([emb, blendshape_vec]).astype(np.float32)

    except Exception:
        return None


# Video handling

def get_video_path(session_path, dialogue_name):
    video_dir = session_path / "dialog" / "avi" / "DivX"
    video_file = video_dir / f"{dialogue_name}.avi"
    if video_file.exists() and str(video_file)[0] != ".":
        return str(video_file)
    return None


def get_frame_indices(start_time, end_time, fps, num_frames=10):
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    if end_frame <= start_frame:
        return [start_frame] * num_frames
    frame_ids = np.linspace(start_frame, end_frame, num_frames).astype(int)
    return frame_ids.tolist()

# Returns the left speaker's ID (e.g., "Ses04F") from the video filename, or None if it can't be determined. If session_num is provided, it will be used to construct the ID.
def get_left_speaker(video_path, session_num=None):
    name = os.path.basename(video_path)
    gender = "F" if "F" in name else ("M" if "M" in name else None)
    if gender is None:
        return None
    if session_num is not None:
        return f"Ses{session_num:02d}{gender}"
    return gender


def extract_frames(start_time, end_time, cap, fps, num_frames=10):
    frame_ids = get_frame_indices(start_time, end_time, fps, num_frames)
    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    return frames


# Per-dialogue processing

def process_dialogue(session_path, dialogue_name, audio_method="wav2vec2"):
    transcript_path = session_path / "dialog" / "transcriptions" / f"{dialogue_name}.txt"
    eval_path = session_path / "dialog" / "EmoEvaluation" / f"{dialogue_name}.txt"
    audio_dir = session_path / "sentences" / "wav" / dialogue_name

    visual_zero_count = 0
    visual_total_count = 0

    full_video_path = get_video_path(session_path, dialogue_name)
    if full_video_path is None:
        cap, fps, left_speaker = None, None, None
    else:
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        session_match = re.match(r'Ses(\d+)', dialogue_name)
        session_num_local = int(session_match.group(1)) if session_match else None
        left_speaker = get_left_speaker(full_video_path, session_num=session_num_local)

    # Parse the transcript to extract utterance IDs, start/end times, and text.
    utterances = []
    pattern = r"(\S+)\s+\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]:\s*(.*)"
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                utterances.append({
                    "utterance_id": match.group(1),
                    "start_time": float(match.group(2)),
                    "end_time": float(match.group(3)),
                    "text": match.group(4).strip(),
                })

    df = pd.DataFrame(utterances)
    if df.empty or "utterance_id" not in df.columns:
        print(f"No utterances parsed from {transcript_path.name} — skipping.")
        return pd.DataFrame(columns=[
            "utterance_id", "start_time", "end_time", "text", "video_path",
            "speaker", "speaker_session", "emotion", "audio_feature_path",
            "text_feature_path", "visual_feature_path", "visual_detection_rate",
        ])

    df["video_path"] = full_video_path
    df["speaker"] = df["utterance_id"].apply(get_speaker_from_utterance)
    df["speaker_session"] = df["utterance_id"].str.extract(r'(Ses\d+)')[0]

    id_to_emotion = load_emotions(eval_path)
    df["emotion"] = df["utterance_id"].map(id_to_emotion)

    feature_paths, text_feat_paths = [], []
    visual_feature_paths, detection_rates = [], []

    for row in df.itertuples():
        utt_id = row.utterance_id
        start_time, end_time = row.start_time, row.end_time

        # Audio
        audio_file = os.path.join(audio_dir, f"{utt_id}.wav")
        t0 = time.perf_counter()
        audio_save_path = FEATURE_DIR / f"{utt_id}_audio_{audio_method}.npy"
        if audio_save_path.exists():
            timer.record(dialogue_name[:5], "audio", "load_cache", time.perf_counter() - t0)
        else:
            audio_feat = extract_audio_features(audio_file, method=audio_method)
            if audio_feat is None:
                audio_dim = 768 if audio_method == "wav2vec2" else AUDIO_DIM
                audio_feat = np.zeros(audio_dim)
            np.save(audio_save_path, audio_feat)
            timer.record(dialogue_name[:5], "audio", "extraction", time.perf_counter() - t0)
        feature_paths.append(str(audio_save_path))

        # Text
        text_save_path = TEXT_FEATURE_DIR / f"{utt_id}_text.npy"
        text_feat_paths.append(str(text_save_path))

        # Visual
        t1 = time.perf_counter()
        visual_save_path = VISUAL_FEATURE_DIR / f"{utt_id}_visual.npy"
        speaker = row.speaker
        visual_total_count += 1

        if visual_save_path.exists():
            timer.record(dialogue_name[:5], "visual", "load_cache", time.perf_counter() - t1)
            visual_feature_paths.append(str(visual_save_path))
            detection_rates.append(None)  # cached — rate unknown
            continue

        if cap is None:
            visual_feat = np.zeros(VISUAL_DIM)
            visual_zero_count += 1
            np.save(visual_save_path, visual_feat)
            visual_feature_paths.append(str(visual_save_path))
            detection_rates.append(None)
            continue

        side = "left" if speaker == left_speaker else "right"
        NUM_CANDIDATES, NUM_KEEP = 30, 10
        raw_frames = extract_frames(start_time, end_time, cap, fps, NUM_CANDIDATES)

        candidate_embeddings = [extract_face_embedding_cnn(f, side) for f in raw_frames]
        valid_candidates = [v for v in candidate_embeddings if v is not None]
        detection_rate = len(valid_candidates) / max(len(raw_frames), 1)
        detection_rates.append(detection_rate)

        if len(valid_candidates) < 3:
            # Too few valid frames — fallback to zero vector
            visual_feat = np.zeros(VISUAL_DIM)
            visual_zero_count += 1
        else:
            if len(valid_candidates) < NUM_KEEP:
                pad = [valid_candidates[-1]] * (NUM_KEEP - len(valid_candidates))
                temporal_vectors = np.array(valid_candidates + pad)
            else:
                idx = np.linspace(0, len(valid_candidates) - 1, NUM_KEEP, dtype=int)
                temporal_vectors = np.array([valid_candidates[i] for i in idx])

            mean_vec = np.mean(temporal_vectors, axis=0)
            std_vec = np.std(temporal_vectors, axis=0)
            visual_feat = np.concatenate([mean_vec, std_vec])

        np.save(visual_save_path, visual_feat)
        visual_feature_paths.append(str(visual_save_path))
        timer.record(dialogue_name[:5], "visual", "extraction", time.perf_counter() - t1)

    df["audio_feature_path"] = feature_paths
    df["text_feature_path"] = text_feat_paths
    df["visual_feature_path"] = visual_feature_paths
    df["visual_detection_rate"] = detection_rates
    if cap is not None:
        cap.release()

    if visual_total_count > 0:
        coverage = (visual_total_count - visual_zero_count) / visual_total_count * 100
        print(f"[{dialogue_name}] Visual coverage: {coverage:.1f}% "
              f"({visual_total_count - visual_zero_count}/{visual_total_count} valid)")

    return df


# Text embeddings

def generate_and_cache_bert(df, model_name="roberta-base", batch_size=32):
    mask = [not Path(p).exists() for p in df["text_feature_path"]]
    missing_indices = df.index[mask].tolist()

    if not missing_indices:
        print("All text embeddings found in cache.")
        return

    print(f"Generating text embeddings with {model_name} "
          f"for {len(missing_indices)} utterances...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts_to_process = df.loc[missing_indices, "text"].tolist()
    paths_to_save = df.loc[missing_indices, "text_feature_path"].tolist()

    for i in range(0, len(texts_to_process), batch_size):
        batch_texts = texts_to_process[i:i + batch_size]
        batch_paths = paths_to_save[i:i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over the token embeddings, taking attention mask into account
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = (
                (token_embeddings * mask_expanded).sum(dim=1)
                / mask_expanded.sum(dim=1).clamp(min=1e-9)
            ).cpu().numpy()

        for emb, path in zip(embeddings, batch_paths):
            np.save(path, emb)

    print("Text embeddings saved.")


# Main

def main():
    # Collect all dialogues across sessions 1-5
    tasks = []
    for session_num in range(1, 6):
        s_path = DATASET_ROOT / f"Session{session_num}"
        if not s_path.exists():
            continue
        t_dir = s_path / "dialog" / "transcriptions"
        for file in t_dir.glob("*.txt"):
            if not file.stem.startswith("."):
                tasks.append((s_path, file.stem))

    # Extract features for each dialogue and build the utterance-level index
    print(f"Processing {len(tasks)} dialogues...")
    all_dfs = []
    for session_path, dialogue_name in tqdm(tasks):
        try:
            df_dialogue = process_dialogue(session_path, dialogue_name, audio_method=AUDIO_METHOD)
            if not df_dialogue.empty:
                all_dfs.append(df_dialogue)
        except Exception as e:
            print(f"Skipping {dialogue_name} due to error: {e}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.dropna(
        subset=["audio_feature_path", "text_feature_path", "visual_feature_path", "emotion"]
    )
    df_all = df_all.reset_index(drop=True)

    print("Total samples:", len(df_all))
    print(df_all["emotion"].value_counts())

    # Text embeddings
    generate_and_cache_bert(df_all, model_name="roberta-base")

    # Integer-encode the emotion labels for downstream training scripts.
    le = LabelEncoder()
    df_all["label"] = le.fit_transform(df_all["emotion"])
    print(dict(zip(le.classes_, le.transform(le.classes_))))

    # Any missing visual feature files (e.g., due to failed detection) are filled with zero vectors, so that the downstream training scripts can still load the utterance index without crashing.
    missing_paths = [p for p in df_all["visual_feature_path"] if not Path(p).exists()]
    for p in missing_paths:
        np.save(p, np.zeros(VISUAL_DIM))
    if missing_paths:
        print(f"Filled {len(missing_paths)} missing visual feature files with zeros.")

    # Sanity check: report the percentage of utterances with valid visual features. If >30% are missing, print a warning.
    X_video = np.vstack([np.load(p) for p in df_all["visual_feature_path"]])
    zero_visual_count = np.sum(np.all(X_video == 0, axis=1))
    visual_coverage = (len(X_video) - zero_visual_count) / len(X_video) * 100
    print(f"Valid visual features: {len(X_video) - zero_visual_count}/{len(X_video)} "
          f"({visual_coverage:.1f}%)")
    if 100 - visual_coverage > 30:
        print("WARNING: >30% visual feature coverage loss.")

    # Persist the utterance-level index to CSV for downstream training scripts to load.
    df_all.to_csv(FEATURE_DIR / "df_all.csv", index=False)
    print(f"Saved utterance index to {FEATURE_DIR / 'df_all.csv'}")


if __name__ == "__main__":
    main()
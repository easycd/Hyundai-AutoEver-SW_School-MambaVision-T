"""Train and test MambaVision-T from a Streamlit UI."""
import argparse
from datetime import datetime
from io import BytesIO
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import time

import streamlit as st
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision import transforms

from hand_data import CLASSES, EXTENSIONS, collect, metrics
from result_report import roc_auc
from pr_charts import render_pr_charts
from video_inference import render_video_ui
from train_mambavision import HandDataset

register_heif_opener()
ROOT = Path(__file__).resolve().parent
LABELS = {0: "정상", 1: "반칙"}


def to_wsl_path(value):
    value = value.strip().strip('"')
    if re.match(r"^[A-Za-z]:[\\/]", value):
        win = PureWindowsPath(value)
        return Path("/mnt") / win.drive[0].lower() / Path(*win.parts[1:])
    return Path(value).expanduser().resolve()


def choose_windows_folder():
    powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not powershell.exists():
        return None
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$dialog.Description = 'Select dataset folder'; "
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $dialog.SelectedPath }"
    )
    result = subprocess.run([str(powershell), "-NoProfile", "-Command", script],
                            text=True, capture_output=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() or None


def dataset_status(root):
    try:
        train = collect(root, "train")
        test = collect(root, "test")
    except (FileNotFoundError, ValueError) as error:
        return None, str(error)
    return {
        "train 정상": sum(row["label"] == 0 for row in train),
        "train 반칙": sum(row["label"] == 1 for row in train),
        "test 정상": sum(row["label"] == 0 for row in test),
        "test 반칙": sum(row["label"] == 1 for row in test),
    }, None


def checkpoints():
    runs = ROOT / "runs"
    if not runs.exists():
        return []
    return sorted(runs.glob("**/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)


def active_training_processes():
    """Detect running trainers, including ones started before locking was added."""
    result = subprocess.run(["pgrep", "-af", "train_mambavision.py"],
                            text=True, capture_output=True)
    return [line for line in result.stdout.splitlines()
            if "python" in line and "train_mambavision.py" in line]


@st.cache_resource(show_spinner="학습된 MambaVision-T를 불러오는 중입니다...")
def load_model(checkpoint_path, modified_time):
    del modified_time
    from mambavision import create_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_to_idx") != CLASSES:
        raise ValueError("체크포인트의 라벨 구성이 정상=0, 반칙=1과 다릅니다.")
    cache = Path.home() / ".cache" / "hand_mambavision"
    cache.mkdir(parents=True, exist_ok=True)
    with torch.serialization.safe_globals([argparse.Namespace]):
        model = create_model("mamba_vision_T", pretrained=False,
                             model_path=str(cache / "mambavision_tiny_1k.pth.tar"))
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model.num_classes = 2
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device, checkpoint


def open_image(source):
    with Image.open(source) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def preprocess(image, checkpoint):
    size = int(checkpoint["image_size"])
    padded = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC,
                          color=(124, 116, 104))
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    return pipeline(padded).unsqueeze(0)


def predict(model, device, checkpoint, image, threshold=0.5):
    tensor = preprocess(image, checkpoint).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    predicted = int(probabilities[1] > threshold)
    return predicted, probabilities


def test_files(root, label):
    folder = root / label / "test"
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in EXTENSIONS)


def test_signature(root):
    parts = []
    for row in collect(root, "test"):
        path = Path(root) / row["path"]
        stat = path.stat()
        parts.append(f"{row['path']}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner="test 이미지 전체를 판정하는 중입니다...")
def evaluate_all_test(checkpoint_path, checkpoint_mtime, data_path, signature):
    del signature
    model, device, checkpoint = load_model(checkpoint_path, checkpoint_mtime)
    rows = collect(data_path, "test")
    dataset = HandDataset(data_path, rows, checkpoint["image_size"],
                          checkpoint["mean"], checkpoint["std"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                         num_workers=4, pin_memory=True,
                                         persistent_workers=True)
    results = []
    offset = 0
    forward_seconds = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if offset == 0:
                for _ in range(3):
                    model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - started
            probabilities = logits.softmax(dim=1).cpu()
            for index in range(len(labels)):
                row = rows[offset + index]
                violation_probability = float(probabilities[index, 1])
                results.append({
                    "path": row["path"],
                    "file": Path(row["path"]).name,
                    "actual": int(row["label"]),
                    "predicted": int(violation_probability > 0.5),
                    "violation_probability": violation_probability,
                })
            offset += len(labels)
    scores = metrics([r["actual"] for r in results], [r["predicted"] for r in results])
    scores["roc_auc"] = roc_auc([r["actual"] for r in results],
                                [r["violation_probability"] for r in results])
    scores["inference_ms_per_image"] = forward_seconds * 1000 / len(results)
    return {"results": results, "scores": scores,
            "params": sum(p.numel() for p in model.parameters())}


st.set_page_config(page_title="MambaVision 손 이탈 판독", page_icon="✋", layout="wide")
st.title("MambaVision 손 이탈 학습 및 판독")
st.caption("데이터셋 폴더 선택 → MambaVision-T 학습 → test 이미지 판정")

if "data_dir" not in st.session_state:
    st.session_state.data_dir = str(ROOT / "hand_img")
if "run_name" not in st.session_state:
    st.session_state.run_name = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")

st.subheader("1. 데이터셋 폴더")
folder_col, path_col = st.columns([1, 4])
with folder_col:
    if st.button("폴더 선택", width="stretch"):
        selected = choose_windows_folder()
        if selected:
            st.session_state.data_dir = selected
with path_col:
    st.text_input("데이터셋 경로", key="data_dir",
                  help=r"예: C:\Project\Model\hand_img 또는 /mnt/c/Project/Model/hand_img")

try:
    data_root = to_wsl_path(st.session_state.data_dir)
    counts, dataset_error = dataset_status(data_root)
except Exception as error:
    data_root, counts, dataset_error = None, None, str(error)

if dataset_error:
    st.error("선택한 폴더에서 정상/반칙 train·test 구조를 찾지 못했습니다.")
    st.code("선택한 폴더/\n  정상/train/\n  정상/test/\n  반칙/train/\n  반칙/test/")
    st.caption(dataset_error)
else:
    columns = st.columns(4)
    for column, (name, count) in zip(columns, counts.items()):
        column.metric(name, f"{count}장")

train_tab, predict_tab, video_tab = st.tabs(["2. 모델 학습", "3. test 이미지 판정", "4. 영상 판독"])

with train_tab:
    st.write("ImageNet 사전학습 MambaVision-T를 정상·반칙 2개 클래스로 미세조정합니다.")
    setting_columns = st.columns(3)
    with setting_columns[0]:
        epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30, step=1)
    with setting_columns[1]:
        batch_size = st.selectbox("Batch size", [2, 4, 8], index=2)
    with setting_columns[2]:
        st.text_input("실험 이름", key="run_name")

    valid_run_name = bool(re.fullmatch(r"[A-Za-z0-9_.-]+", st.session_state.run_name))
    output_dir = ROOT / "runs" / st.session_state.run_name
    active_trainings = active_training_processes()
    if active_trainings:
        st.warning("현재 다른 학습이 실행 중입니다. 완료되거나 중단될 때까지 새 학습을 시작할 수 없습니다.")
    if not valid_run_name:
        st.error("실험 이름에는 영문, 숫자, 마침표, 밑줄, 하이픈만 사용할 수 있습니다.")
    elif output_dir.exists() and any(output_dir.iterdir()):
        st.warning("같은 이름의 결과 폴더가 이미 있습니다. 새 실험 이름을 입력하세요.")

    can_train = (counts is not None and valid_run_name and not active_trainings
                 and not (output_dir.exists() and any(output_dir.iterdir())))
    if st.button("학습 시작", type="primary", disabled=not can_train, width="stretch"):
        command = [
            sys.executable, "-u", str(ROOT / "train_mambavision.py"),
            "--data-dir", str(data_root), "--epochs", str(int(epochs)),
            "--batch-size", str(batch_size), "--output", str(output_dir),
        ]
        status = st.status("학습 준비 중...", expanded=True)
        log_area = st.empty()
        log_lines = []
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-18:]), language="text")
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if return_code == 0 and (output_dir / "best.pt").exists():
            status.update(label="학습 완료", state="complete", expanded=False)
            st.session_state.latest_checkpoint = str(output_dir / "best.pt")
            st.success("학습이 완료됐습니다. ‘test 이미지 판정’ 탭에서 결과를 확인하세요.")
        else:
            status.update(label="학습 실패", state="error", expanded=True)
            st.error("로그의 마지막 오류를 확인하세요.")

with predict_tab:
    available = checkpoints()
    if not available:
        st.info("먼저 ‘모델 학습’ 탭에서 학습을 완료하세요.")
    elif counts is None:
        st.info("유효한 데이터셋 폴더를 선택하세요.")
    else:
        preferred = st.session_state.get("latest_checkpoint")
        default_index = next((i for i, path in enumerate(available) if str(path) == preferred), 0)
        selected_checkpoint = st.selectbox(
            "사용할 학습 모델", available, index=default_index,
            format_func=lambda path: str(path.relative_to(ROOT)),
        )
        try:
            model, device, checkpoint = load_model(
                str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns
            )
        except Exception as error:
            st.error(f"모델을 불러오지 못했습니다: {error}")
            st.stop()

        signature = test_signature(data_root)
        try:
            evaluation = evaluate_all_test(
                str(selected_checkpoint), selected_checkpoint.stat().st_mtime_ns,
                str(data_root), signature,
            )
        except Exception as error:
            st.error(f"test 전체 판정 중 오류가 발생했습니다: {error}")
            st.stop()

        threshold = st.slider(
            "반칙 판정 기준값 (Threshold)", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01, format="%.2f", key="decision_threshold",
            help="반칙 확률이 기준값보다 크면 반칙, 같거나 작으면 정상입니다.",
        )
        st.caption(f"반칙 확률이 {threshold * 100:.0f}%를 초과하면 반칙으로 판정합니다. "
                   "기준을 낮추면 더 많은 사진을 반칙으로, 높이면 더 적은 사진을 반칙으로 판정합니다.")
        st.caption("기준값 변경은 재학습 없이 적용됩니다. 최종 기준은 validation 데이터로 정하는 것이 좋습니다.")
        all_results = [dict(r, predicted=int(r["violation_probability"] > threshold))
                       for r in evaluation["results"]]
        scores = dict(evaluation["scores"])
        scores.update(metrics([r["actual"] for r in all_results],
                              [r["predicted"] for r in all_results]))
        training_summary = selected_checkpoint.parent / "training_summary.json"
        train_seconds = None
        if training_summary.exists():
            train_seconds = json.loads(training_summary.read_text(encoding="utf-8")).get("train_seconds")
        st.subheader("모델 성능 · test 전체 기준")
        st.caption(f"실험: {selected_checkpoint.parent.name} · test {len(all_results)}장 · 반칙을 양성으로 계산")
        performance = {
            "모델": "MambaVision-T",
            "Threshold": f"{threshold:.2f}",
            "Accuracy": f"{scores['accuracy']:.4f}",
            "Precision": f"{scores['violation_precision']:.4f}",
            "Recall": f"{scores['violation_recall']:.4f}",
            "F1-score": f"{scores['violation_f1']:.4f}",
            "ROC-AUC": "N/A" if scores["roc_auc"] is None else f"{scores['roc_auc']:.4f}",
            "Train Time (s)": "기록 없음" if train_seconds is None else f"{train_seconds:.2f}",
            "Inference Time (ms/장)": f"{scores['inference_ms_per_image']:.2f}",
            "Params": f"{evaluation['params']:,}",
        }
        st.dataframe([performance], hide_index=True, width="stretch")
        tn, fp = scores["confusion_matrix"][0]
        fn, tp = scores["confusion_matrix"][1]
        st.dataframe([{"모델": "MambaVision-T", "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                       "주요 실패 유형": "오분류 없음" if fp + fn == 0 else "아래 오분류 이미지에서 확인"}],
                     hide_index=True, width="stretch")
        st.caption("점수는 0~1. FN은 반칙을 정상으로 놓친 수, FP는 정상을 반칙으로 판정한 수입니다. "
                   "실패 원인은 이미지 확인이 필요합니다.")
        st.caption("ROC-AUC는 확률의 구분 능력을 나타내므로 기준값을 바꿔도 변하지 않습니다.")
        st.caption("학습 시간은 검증·데이터 로딩을 포함한 전체 학습 시간입니다. "
                   "추론 시간은 3회 워밍업 후 GPU forward만 측정한 배치 크기 8의 장당 평균이며 전처리를 제외합니다.")
        table_text = "\t".join(performance) + "\n" + "\t".join(performance.values())
        table_text += f"\n\n모델\tTP\tTN\tFP\tFN\nMambaVision-T\t{tp}\t{tn}\t{fp}\t{fn}\n"
        table_text += "\n평가: test 전체 / 양성: 반칙 / 점수: 0~1 / 추론: batch 8, forward만, 워밍업 제외\n"
        table_text += f"판정 기준: 반칙 확률 > {threshold:.2f}이면 반칙, 같거나 작으면 정상\n"
        st.download_button("성능 표 TXT 다운로드", data=table_text.encode("utf-8-sig"),
                           file_name=f"{selected_checkpoint.parent.name}_test_performance.txt", mime="text/plain")

        render_pr_charts(all_results, threshold, scores)

        st.subheader("test 폴더 전체 판정 요약")
        normal_results = [result for result in all_results if result["actual"] == 0]
        violation_results = [result for result in all_results if result["actual"] == 1]
        summary = [
            {
                "실제 폴더": "정상 폴더",
                "전체": len(normal_results),
                "정상 판정": sum(result["predicted"] == 0 for result in normal_results),
                "반칙 판정": sum(result["predicted"] == 1 for result in normal_results),
            },
            {
                "실제 폴더": "반칙 폴더",
                "전체": len(violation_results),
                "정상 판정": sum(result["predicted"] == 0 for result in violation_results),
                "반칙 판정": sum(result["predicted"] == 1 for result in violation_results),
            },
        ]
        st.dataframe(summary, hide_index=True, width="stretch")

        input_mode = st.radio("이미지 입력", ["test 이미지 전체 보기", "다른 이미지 업로드"], horizontal=True)
        image = actual_label = source_name = predicted = probabilities = None
        if input_mode == "test 이미지 전체 보기":
            test_rows = collect(data_root, "test")
            preview_dataset = HandDataset(
                data_root, test_rows, checkpoint["image_size"], checkpoint["mean"], checkpoint["std"]
            )
            st.caption("모델에 입력된 축소 이미지를 전부 표시합니다. 각 이미지 아래에서 판정 결과를 확인하세요.")
            for label in (0, 1):
                grouped = [(i, r) for i, r in enumerate(all_results) if r["actual"] == label]
                st.subheader(f"{LABELS[label]} 폴더 · {len(grouped)}장")
                for start in range(0, len(grouped), 4):
                    columns = st.columns(4)
                    for column, (index, result) in zip(columns, grouped[start:start + 4]):
                        with column:
                            with st.container(border=True):
                                st.image(preview_dataset.prepared_image(index), width="stretch")
                                st.caption(result["file"])
                                if result["predicted"] == 1:
                                    st.error("판정: 반칙")
                                else:
                                    st.success("판정: 정상")
                                st.write(f"실제: {LABELS[label]}")
                                st.write(f"반칙 확률: {result['violation_probability'] * 100:.2f}%")
                                if result["predicted"] == label:
                                    st.caption("예측 일치")
                                else:
                                    failure = "FN · 반칙을 놓침" if label == 1 else "FP · 정상 오탐"
                                    st.warning(f"오분류 ({failure})")
        else:
            uploaded = st.file_uploader(
                "판정할 이미지", type=["jpg", "jpeg", "png", "heic", "heif", "webp", "bmp"]
            )
            if uploaded:
                image = open_image(BytesIO(uploaded.getvalue()))
                source_name = uploaded.name
                with st.spinner("판정 중..."):
                    predicted, probabilities = predict(model, device, checkpoint, image, threshold)

        if image is not None:
            image_column, result_column = st.columns([1.35, 1])
            with image_column:
                st.image(image, caption=source_name, width="stretch")
            with result_column:
                if predicted == 1:
                    st.error("판정 결과: 반칙")
                else:
                    st.success("판정 결과: 정상")
                st.metric("반칙 확률", f"{probabilities[1] * 100:.2f}%")
                st.progress(float(probabilities[1]),
                            text=f"정상 {probabilities[0] * 100:.2f}% · 반칙 {probabilities[1] * 100:.2f}%")
                if actual_label is not None:
                    if predicted == actual_label:
                        st.success(f"실제 라벨: {LABELS[actual_label]} · 예측 일치")
                    else:
                        st.error(f"실제 라벨: {LABELS[actual_label]} · 오분류")
                st.caption(f"실행 장치: {device} · best epoch: {checkpoint.get('epoch', '기록 없음')}")

with video_tab:
    render_video_ui(ROOT, checkpoints(), load_model, predict, bool(active_training_processes()))

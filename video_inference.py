"""Local video inference and H.264 export with frame-level decisions."""
import csv
import json
import math
import subprocess
from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
from PIL import Image, ImageOps


def prepare_video_input(image, settings=None):
    """Crop original frame, then rotate; model's existing preprocessing follows."""
    settings = settings or {}
    x0, x1 = settings.get("x_percent", [0, 100])
    y0, y1 = settings.get("y_percent", [0, 100])
    rotation = settings.get("rotation", 0)
    if not (0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100):
        raise ValueError("판독 영역의 시작 위치는 끝 위치보다 작아야 합니다.")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("회전은 0, 90, 180, 270도만 가능합니다.")
    width, height = image.size
    box = (int(width * x0 / 100), int(height * y0 / 100),
           int(width * x1 / 100), int(height * y1 / 100))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("판독 영역이 너무 작습니다.")
    cropped = image.crop(box)
    return cropped.rotate(-rotation, expand=True) if rotation else cropped


def region_preview(frame, settings):
    shown = frame.copy()
    height, width = shown.shape[:2]
    x0, x1 = settings["x_percent"]
    y0, y1 = settings["y_percent"]
    cv2.rectangle(shown, (int(width * x0 / 100), int(height * y0 / 100)),
                  (min(width - 1, int(width * x1 / 100)), min(height - 1, int(height * y1 / 100))),
                  (255, 200, 0), max(2, width // 150))
    return cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)


def annotate(frame, probability, threshold):
    height, width = frame.shape[:2]
    violation = probability > threshold
    color = (40, 45, 215) if violation else (70, 150, 35)
    border = max(4, round(min(height, width) * 0.018))
    header = max(48, round(height * 0.105))
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, border)
    cv2.rectangle(frame, (0, 0), (width, header), color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "WARNING" if violation else "NORMAL"
    scale = max(0.35, min(width / 350, header / 70))
    thickness = max(1, round(scale * 2))
    size = cv2.getTextSize(title, font, scale, thickness)[0]
    cv2.putText(frame, title, ((width - size[0]) // 2, int(header * 0.53)),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    detail = f"FOUL {probability * 100:.1f}%  |  TH {threshold:.2f}"
    small = min(scale * 0.42, width / 650)
    size = cv2.getTextSize(detail, font, small, 1)[0]
    cv2.putText(frame, detail, ((width - size[0]) // 2, int(header * 0.88)),
                font, small, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_video(source, output_dir, infer, threshold, model_name, progress=None, input_settings=None):
    """infer receives configured RGB crop; labels always derive from its probability."""
    source, output_dir = Path(source), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    encoder = None
    output = output_dir / "result.mp4"
    temporary = output_dir / "encoding.mp4"
    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다. MP4(H.264) 파일인지 확인해 주세요.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("영상 FPS를 읽을 수 없습니다.")
        ok, frame = cap.read()
        if not ok:
            raise ValueError("영상에 읽을 수 있는 프레임이 없습니다.")
        h, w = frame.shape[:2]
        factor = min(1.0, 1280 / max(w, h))
        width, height = max(2, int(w * factor) // 2 * 2), max(2, int(h * factor) // 2 * 2)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                   "-r", str(fps), "-i", "pipe:0", "-i", str(source),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-movflags", "+faststart", str(temporary)]
        count = violations = 0
        with (output_dir / "encoding.log").open("wb") as errors, \
                (output_dir / "frame_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            writer = csv.writer(file)
            writer.writerow(["frame", "time_seconds", "prediction", "violation_probability", "threshold"])
            while ok:
                # Predict before drawing the warning onto the output.
                original = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                probability = float(infer(prepare_video_input(original, input_settings)))
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("모델이 유효하지 않은 확률을 반환했습니다.")
                violation = probability > threshold
                rendered = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                rendered = annotate(rendered, probability, threshold)
                encoder.stdin.write(rendered.tobytes())
                writer.writerow([count, count / fps, "반칙" if violation else "정상", probability, threshold])
                count += 1
                violations += int(violation)
                if progress and (count == 1 or count % 12 == 0):
                    progress(count, total)
                ok, frame = cap.read()
            encoder.stdin.close()
            if encoder.wait(timeout=120) != 0:
                raise RuntimeError("결과 영상 저장 실패: encoding.log를 확인해 주세요.")
        if total > 0 and count < total - 1:
            raise RuntimeError(f"영상이 중간에 끊겼습니다 ({count}/{total} 프레임). 원본을 확인해 주세요.")
        temporary.replace(output)
        summary = {"source": source.name, "model": model_name, "threshold": threshold,
                   "input_settings": input_settings or {"x_percent": [0, 100], "y_percent": [0, 100], "rotation": 0},
                   "frames": count, "normal_frames": count - violations, "violation_frames": violations,
                   "fps": fps, "duration_seconds": count / fps, "output": str(output)}
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "summary.txt").write_text(
            f"모델: {model_name}\nThreshold: {threshold:.2f}\n전체 프레임: {count}\n"
            f"정상: {count - violations}\n반칙: {violations}\nFPS: {fps}\n"
            f"입력 설정: {json.dumps(summary['input_settings'], ensure_ascii=False)}\n"
            "판정: 반칙 확률이 Threshold를 초과하면 반칙. 프레임 수는 반칙 사건 수가 아닙니다.\n",
            encoding="utf-8-sig")
        return summary
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()


def render_video_ui(root, available, load_model, predict, training_active=False):
    import streamlit as st

    st.subheader("영상 업로드 및 판독")
    if not available:
        st.info("학습을 완료한 후 영상을 판독할 수 있습니다.")
        return
    chosen = st.selectbox("영상 판독 모델", available,
                          format_func=lambda p: str(p.relative_to(root)), key="video_model")
    threshold = st.slider("영상 Threshold", 0.0, 1.0,
                          float(st.session_state.get("decision_threshold", 0.5)), 0.01, key="video_threshold")
    st.caption("초록색 NORMAL = 정상 · 빨간색 WARNING = 반칙. 확률이 기준값을 초과하면 반칙입니다.")
    uploaded = st.file_uploader("판독할 영상", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    st.caption("카메라에 손과 패드가 함께 보이는 원본 영상을 사용하세요. 모든 프레임을 판독하며 긴 영상은 시간이 걸립니다.")
    st.info("현재 모델은 사진 전체를 분류합니다. 손과 패드 경계를 직접 측정하는 기능은 아니므로, "
            "새 영상에서는 정상 장면도 반칙으로 오판정할 수 있습니다. 저장 전에 정상·이탈 장면을 미리 확인하세요.")
    settings = {"x_percent": [0, 100], "y_percent": [0, 100], "rotation": 0}
    valid_settings = True
    digest = None
    if uploaded is not None:
        import hashlib
        digest = hashlib.sha256(uploaded.getbuffer()).hexdigest()
        preview_directory = root / "video_results" / "previews"
        preview_directory.mkdir(parents=True, exist_ok=True)
        preview_source = preview_directory / (digest + Path(uploaded.name).suffix.lower())
        if not preview_source.exists():
            preview_source.write_bytes(uploaded.getbuffer())
        preview_cap = cv2.VideoCapture(str(preview_source))
        preview_fps = preview_cap.get(cv2.CAP_PROP_FPS)
        preview_count = int(preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preview_cap.release()
        if preview_count > 0 and preview_fps > 0:
            st.subheader("모델 입력 영역·방향")
            st.caption("패드 전체와 밖으로 나온 손가락이 모두 포함되게 영역을 정하세요. "
                       "너무 좁게 자르면 반칙 증거가 사라집니다. 회전은 시계 방향입니다.")
            left, right = st.columns(2)
            with left:
                settings["x_percent"] = list(st.slider("가로 범위 (%)", 0, 100, (0, 100),
                                                        key=f"video_crop_x_{digest[:12]}"))
                settings["y_percent"] = list(st.slider("세로 범위 (%)", 0, 100, (0, 100),
                                                        key=f"video_crop_y_{digest[:12]}"))
            with right:
                settings["rotation"] = st.selectbox("입력 회전", [0, 90, 180, 270],
                                                     format_func=lambda angle: f"{angle}°",
                                                     key=f"video_rotation_{digest[:12]}")
            valid_settings = all(settings[key][0] < settings[key][1] for key in ("x_percent", "y_percent"))
            if not valid_settings:
                st.error("가로·세로 범위의 양 끝을 서로 다르게 지정하세요.")
            second = st.number_input("미리 볼 시간 (초)", min_value=0.0,
                                     max_value=(preview_count - 1) / preview_fps,
                                     value=0.0, step=0.1, key=f"preview_time_{digest[:12]}")
            @st.cache_data(show_spinner=False, max_entries=8)
            def read_preview(path, frame_number):
                reader = cv2.VideoCapture(path)
                try:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, frame = reader.read()
                    if not ok:
                        raise ValueError("미리보기 프레임을 읽지 못했습니다.")
                    return frame
                finally:
                    reader.release()
            frame_number = int(second * preview_fps)
            selected_frame = read_preview(str(preview_source), frame_number)
            settings_key = json.dumps(settings, sort_keys=True)
            if valid_settings:
                original = Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB))
                prepared = prepare_video_input(original, settings)
                cols = st.columns(2)
                with cols[0]:
                    st.image(region_preview(selected_frame, settings), caption="원본에서 선택한 영역", width=320)
                with cols[1]:
                    st.image(prepared, caption="영역 선택·회전 후 이미지", width=320)
            if st.button("이 장면 판독", disabled=training_active or not valid_settings):
                cap = cv2.VideoCapture(str(preview_source))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * preview_fps))
                    ok, frame = cap.read()
                finally:
                    cap.release()
                if ok:
                    try:
                        model, device, checkpoint = load_model(str(chosen), chosen.stat().st_mtime_ns)
                        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        probability = predict(model, device, checkpoint, prepare_video_input(rgb, settings), threshold)[1][1]
                        st.session_state.video_frame_preview = {
                            "key": (digest, str(chosen), chosen.stat().st_mtime_ns, settings_key, frame_number),
                            "frame": frame, "probability": probability, "second": second,
                            "model_input": ImageOps.pad(prepare_video_input(rgb, settings),
                                (checkpoint["image_size"], checkpoint["image_size"]),
                                method=Image.Resampling.BICUBIC, color=(124,116,104))}
                    except Exception as error:
                        st.error(f"장면 판독 실패: {error}")
            preview = st.session_state.get("video_frame_preview")
            if preview and preview["key"] == (digest, str(chosen), chosen.stat().st_mtime_ns, settings_key, frame_number):
                shown = annotate(preview["frame"].copy(), preview["probability"], threshold)
                st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), width=360)
                st.image(preview["model_input"], caption="실제 모델 입력 (정규화 전)", width=224)
                st.caption(f"{preview['second']:.2f}초 장면 · 반칙 확률 {preview['probability']:.4f} · "
                           f"현재 기준 {threshold:.2f} · "
                           f"{'반칙' if preview['probability'] > threshold else '정상'}")
                st.caption("Threshold를 바꾸면 위 장면의 판정도 즉시 바뀝니다. "
                           "기준을 높이면 정상 오탐이 줄 수 있지만 실제 반칙을 놓칠 수 있습니다.")
    if training_active:
        st.info("학습이 끝난 후 영상 판독을 시작해 주세요.")
    if st.button("영상 판독 및 저장", type="primary", disabled=uploaded is None or training_active or not valid_settings):
        directory = root / "video_results" / (uuid4().hex[:12])
        directory.mkdir(parents=True, exist_ok=False)
        source = directory / ("input" + Path(uploaded.name).suffix.lower())
        source.write_bytes(uploaded.getbuffer())
        bar = st.progress(0.0, text="모델 준비 중...")
        try:
            model, device, checkpoint = load_model(str(chosen), chosen.stat().st_mtime_ns)
            def infer(image):
                return predict(model, device, checkpoint, image, threshold)[1][1]
            def update(count, total):
                bar.progress(min(count / total, 0.99) if total > 0 else 0.0,
                             text=f"판독 중: {count} / {total if total > 0 else '?'} 프레임")
            summary = process_video(source, directory, infer, threshold,
                                    str(chosen.relative_to(root)), update, input_settings=settings)
            summary["source_digest"] = digest
            summary["checkpoint_mtime"] = chosen.stat().st_mtime_ns
            (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state.video_result = summary
            bar.progress(1.0, text="판독 및 저장 완료")
        except Exception as error:
            st.error(f"영상 처리 실패: {error}")
    summary = st.session_state.get("video_result")
    if summary:
        output = Path(summary["output"])
        st.caption(f"저장된 결과 · 모델: {summary['model']} · Threshold: {summary['threshold']:.2f}")
        if (summary["threshold"] != threshold or summary.get("input_settings") != settings
                or summary["model"] != str(chosen.relative_to(root))
                or (digest is not None and summary.get("source_digest") != digest)
                or summary.get("checkpoint_mtime") != chosen.stat().st_mtime_ns):
            st.warning("아래 영상은 현재 선택한 영상·모델·입력 설정과 다를 수 있는 이전 결과입니다. "
                       "현재 설정을 적용하려면 '영상 판독 및 저장'을 다시 누르세요.")
        st.caption(f"저장에 사용한 입력 설정: {summary.get('input_settings', '전체 화면')}")
        st.video(str(output))
        st.write(f"전체 {summary['frames']}프레임 · 정상 {summary['normal_frames']} · 반칙 {summary['violation_frames']}")
        st.caption(f"저장 위치: {output.relative_to(root)}")
        for name, label, mime in [("result.mp4", "판독 영상 다운로드", "video/mp4"),
                                  ("frame_results.csv", "프레임별 결과 CSV", "text/csv"),
                                  ("summary.txt", "결과 요약 TXT", "text/plain")]:
            with (output.parent / name).open("rb") as file:
                st.download_button(label, file, file_name=name, mime=mime, key=f"video_download_{name}")

# 마우스패드 손 이탈 분류: MambaVision-T

정상=0, 반칙=1. 공식 ImageNet-1K 가중치를 로드한 뒤 분류기를 2개 출력으로 교체합니다.
원본 이미지를 수정하지 않으며 HEIC/HEIF/JPG/PNG 등을 읽습니다.

## 환경 준비

공식 Mamba 연산은 CUDA 확장에 의존합니다. NVIDIA GPU가 있는 Linux 또는 Windows의 WSL2 Ubuntu 환경을 사용하세요.
현재 PC는 Ubuntu-22.04(WSL2)의 `/opt/hand-mamba` 가상환경에 설치되어 있습니다.
Python 3.10, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, MambaVision 1.2.0,
Mamba SSM 2.2.4, timm 1.0.15를 사용합니다.
RTX 3060 Laptop GPU에서 실제 HEIC 2장으로 사전학습 가중치 로드, 순전파,
역전파 및 AdamW 업데이트까지 검증했습니다. 결과는 environment_check.json에 있습니다.
이는 환경 검사이며 전체 데이터 학습이나 정확도 평가 결과는 아닙니다.

Windows PowerShell에서 프로젝트 폴더를 열고 바로 실행할 수 있습니다:

```powershell
.\train.ps1 --epochs 30 --batch-size 8 --output runs/experiment_01
```

학습 후 판정 UI 실행:

```powershell
.\run-ui.ps1
```

브라우저에서 `http://localhost:8501`이 열립니다. UI에서 데이터셋 폴더를 지정하고
epochs, batch size, 실험 이름을 정해 학습할 수 있습니다. 학습이 끝나면 test 폴더 사진을
골라 실제 라벨과 예측 결과를 비교하거나, 별도 이미지를 업로드해 판정할 수 있습니다.
test 판정 화면은 전체 이미지를 정상/반칙 폴더별 4열 갤러리로 표시합니다.
각 이미지 아래에 예측 라벨, 실제 라벨, 반칙 확률, 오분류(FN/FP)를 보여줍니다.
상단에는 선택한 모델의 test 성능 표와 혼동행렬 개수를 표시하며 TXT 다운로드도 지원합니다.
학습 시간 기록이 없는 모델은 '기록 없음'으로 표시합니다.
판정 탭의 Threshold 슬라이더(0.00~1.00, 기본 0.50)로 기준값을 조절할 수 있습니다.
반칙 확률이 기준값을 초과하면 반칙이며, 같거나 작으면 정상입니다.
변경 시 전체 갤러리·폴더 집계·성능 표·업로드 판정에 반영됩니다.
test 전체 확률은 캐시를 재사용하며 재학습하지 않습니다. TXT에도 기준값을 기록합니다.
VS Code에서는 `Ctrl+Shift+B` 또는 `Terminal > Run Build Task`로도 실행할 수 있습니다.
UI를 종료하려면 실행 중인 VS Code 터미널에서 `Ctrl+C`를 누릅니다.

### 영상 판독

`4. 영상 판독` 탭에서 모델과 Threshold를 선택하고 MP4/MOV/AVI/MKV를 업로드한 후
`영상 판독 및 저장`을 누릅니다. 각 프레임을 학습과 같은 전처리로 판독합니다.
정상은 초록색 NORMAL, 반칙은 빨간색 WARNING으로 표시하며 반칙 확률도 영상에 넣습니다.
`video_results/실행ID/`에 결과 MP4, 프레임별 CSV, 요약 TXT가 저장됩니다.
UI에서 재생하거나 다운로드할 수 있습니다. 오디오가 있으면 AAC로 포함합니다.
출력은 원본 평균 FPS 기준 H.264이며 긴 변의 최대 크기는 1280px입니다.
원본에 이미 경고 자막이 있으면 모델 입력에도 들어가므로 자막 없는 원본 사용을 권합니다.
영상 업로드 후 가로·세로 범위(%)와 시계 방향 회전(0/90/180/270°)을 지정할 수 있습니다.
시간을 선택하면 원본의 판독 영역과 변환된 화면을 볼 수 있습니다.
`이 장면 판독`을 누르면 실제 모델 입력과 확률이 나타납니다.
미리보기와 전체 영상 저장 모두 같은 영역 선택·회전 후 기존 학습 전처리를 적용합니다.
전체 화면/0°가 기본값이며, 입력 설정은 summary.json과 summary.txt에 기록합니다.
영역 조정 자체가 정확도 향상을 보장하지 않습니다. 패드 밖으로 나온 손가락까지 포함해야 합니다.

전용 환경은 현재 WSL의 root 계정에서 관리합니다. 실행 파일이 이 계정과 환경을 지정합니다.
설치 버전 목록은 requirements-installed.txt, 새 Ubuntu 환경 설치 스크립트는 setup_wsl.sh입니다.
아래 수동 설치 안내는 다른 환경에 설치할 때 참고하세요.

1. https://pytorch.org/get-started/locally/ 에서 환경에 맞는 **CUDA PyTorch와 torchvision** 설치 명령을 선택합니다.
2. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 결과가 True인지 확인합니다.
3. `python -m pip install packaging ninja wheel setuptools` 실행 후 `python -m pip install -r requirements.txt`를 실행합니다.
4. mamba-ssm 빌드가 필요하면 PyTorch CUDA 버전과 호환되는 CUDA Toolkit/nvcc가 필요합니다.
   공식 설치 안내: https://github.com/state-spaces/mamba#installation
   빌드 격리 때문에 설치가 실패하면 해당 환경에서 `python -m pip install mamba-ssm --no-build-isolation` 후 requirements 설치를 다시 시도합니다.

첫 학습은 공식 사전학습 가중치를 다운로드하므로 인터넷 연결이 필요합니다.
패키지 설치 및 GPU 호환성은 실제 실행 환경에서 확인해야 합니다.

## 데이터 구조

```text
hand_img/
  정상/train/    (244장)
  정상/test/     (27장)
  반칙/train/    (197장)
  반칙/test/     (22장)
```

기본 경로는 스크립트 옆 hand_img입니다. `--data-dir`로 다른 위치를 지정할 수 있습니다.
WSL에서 프로젝트로 이동하는 예: `cd /mnt/c/Project/Model`
데스크톱 원본은 WSL에서 `/mnt/c/Users/JAEWON/Desktop/hand_img`입니다.

## 실행

파일 목록과 분할 확인만 하기 (PyTorch 불필요, 이미지 디코딩 검사는 아님):

```bash
python train_mambavision.py --check-data
```

학습:

```bash
python train_mambavision.py --epochs 30 --batch-size 8 --output runs/experiment_01
```

기본 분할은 기존 train 441장에서 학습 353장, 검증 88장입니다.
각 클래스별로 고정 seed=42를 사용하며 실제 분할은 split.json에 저장합니다.
이는 이미지 단위 분할이므로 동일 인물/연속 촬영 그룹의 독립성을 보장하지 않습니다.
팀 비교 전 촬영 그룹 누수를 확인하고 split.json의 동일한 분할을 다른 모델에도 적용하세요.
test 폴더는 학습/최적 모델 선택 과정에서 읽지 않습니다.

첫 3 epoch는 분류기만 학습하고 이후 전체 모델을 작은 학습률로 미세조정합니다.
작은 데이터셋이므로 BatchNorm의 사전학습 running statistics를 유지합니다.
기본값은 FP32입니다. GPU 메모리가 부족하면 `--batch-size 4` 또는 `2`로 줄이세요.
고해상도 HEIC 원본은 첫 사용 때 `.image_cache/224`에 축소·캐시하며 4개 작업자가 병렬 처리합니다.
첫 epoch는 캐시 생성 때문에 느릴 수 있지만 이후 epoch는 원본 HEIC를 다시 해제하지 않습니다.
원본 파일의 크기나 수정 시간이 바뀌면 해당 이미지의 캐시를 자동으로 새로 만듭니다.
검증 반칙 F1을 최우선으로, 동률이면 검증 loss로 best.pt를 고릅니다.
미세조정 단계에서 7 epoch 동안 개선이 없으면 조기 종료합니다.

전처리는 EXIF 방향 보정 → RGB 변환 → 비율 유지 및 여백 추가(224×224) → 정규화입니다.
손끝을 잘라낼 수 있는 랜덤 크롭, MixUp, CutMix는 사용하지 않습니다.
학습 시 약한 밝기/대비/채도 변화만 적용합니다.
전체 사진 축소로 작은 이탈이 사라질 수 있으므로 경계 사례는 따로 확인하세요.

## 최종 test 평가

검증 결과로 설정을 확정한 후 실행합니다. test 결과로 반복 튜닝하지 마세요.

```bash
python train_mambavision.py --evaluate runs/experiment_01/best.pt
```

저장 파일:

- best.pt: 최적 모델 가중치, 라벨 매핑, 전처리 정보 (학습 재개용 optimizer 상태는 없음)
- split.json, config.json, versions.json: 분할, 실행 설정, 패키지 버전
- history.json: epoch별 학습/검증 loss와 지표
- test_metrics.json, test_predictions.csv: 별도 평가 실행 시 생성
- validation_results.txt: 학습 완료 시 최적 epoch의 검증 결과 표 자동 저장
- test_results.txt: `--evaluate` 실행 시 최종 test 결과 표 자동 저장
- training_summary.json: 전체 학습 시간과 최적 epoch 기록

TXT는 Accuracy, Precision, Recall, F1, ROC-AUC, 학습 시간(초), 추론 시간(ms/장),
전체 파라미터 수 및 TP/TN/FP/FN을 탭으로 구분해 저장합니다.
반칙을 양성으로 계산하며 FN/FP 이미지 경로도 포함합니다. 주요 실패 유형은 사진을 보고 수동 분류합니다.
추론 시간은 3회 워밍업 후 GPU forward만 측정한 배치 평균이며 파일 읽기와 전처리는 제외합니다.
모델끼리 같은 GPU, 이미지 크기, batch size로 비교하세요.

혼동행렬의 행은 실제 클래스, 열은 예측 클래스이며 순서는 정상, 반칙입니다.
확률이 더 큰 클래스로 판정합니다. 지표는 정확도, 반칙 precision/recall/F1, 정상 오탐률입니다.
seed를 고정하더라도 CUDA 연산 및 환경 차이에 따른 완전한 재현성은 보장하지 않습니다.

공식 모델/API: https://github.com/NVlabs/MambaVision

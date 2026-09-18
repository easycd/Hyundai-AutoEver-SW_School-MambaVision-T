"""Plain-text comparison report. Positive class: violation (1)."""
from pathlib import Path


def roc_auc(targets, scores):
    positive = [s for y, s in zip(targets, scores) if y == 1]
    negative = [s for y, s in zip(targets, scores) if y == 0]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def write_report(path, split, result, rows, predictions, train_seconds, params, batch_size, gpu):
    def fmt(value):
        return 'N/A' if value is None else f'{value:.6f}'
    tn, fp = result['confusion_matrix'][0]
    fn, tp = result['confusion_matrix'][1]
    lines = [
        f'평가 데이터: {split} ({len(rows)}장)',
        '양성(Positive): 반칙=1 / 음성(Negative): 정상=0',
        'Precision, Recall, F1-score: 반칙 클래스 기준 / 점수 범위: 0~1',
        '판정: 두 클래스 중 확률이 큰 클래스 (동률은 정상)',
        '',
        '모델\tAccuracy\tPrecision\tRecall\tF1-score\tROC-AUC\tTrain Time (s)\tInference Time (ms/image)\tParams',
        '\t'.join(['MambaVision-T', fmt(result['accuracy']), fmt(result['violation_precision']),
                   fmt(result['violation_recall']), fmt(result['violation_f1']), fmt(result['roc_auc']),
                   fmt(train_seconds), fmt(result['inference_ms_per_image']), str(params)]),
        '', '모델\tTP\tTN\tFP\tFN\t주요 실패 유형',
        f'MambaVision-T\t{tp}\t{tn}\t{fp}\t{fn}\t수동 확인 필요',
        '', 'TP: 반칙을 반칙으로 판정 / TN: 정상을 정상으로 판정',
        'FP: 정상을 반칙으로 판정 / FN: 반칙을 정상으로 놓침',
        f'GPU: {gpu} / 평가 batch size: {batch_size}',
        'Train Time: 전체 학습 루프 경과 시간(검증·데이터 로딩·체크포인트 저장 포함, 모델 다운로드 제외)',
        'Inference Time: GPU 동기화 후 측정한 forward 시간 / 이미지 수. 3회 워밍업 제외.',
        '이미지 읽기·전처리·GPU 전송·후처리 제외. 배치당 시간을 이미지 수로 나눈 값이며 단일 요청 지연과 다름.',
        'Params: 분류기를 포함한 전체 파라미터 수. N/A는 측정 기록이 없거나 계산 불가.',
        '', '[FN: 실제 반칙 → 정상으로 놓친 이미지]',
    ]
    lines.extend(r['path'] for r, p in zip(rows, predictions) if r['label'] == 1 and p == 0)
    if fn == 0:
        lines.append('없음')
    lines.append('\n[FP: 실제 정상 → 반칙으로 오탐한 이미지]')
    lines.extend(r['path'] for r, p in zip(rows, predictions) if r['label'] == 0 and p == 1)
    if fp == 0:
        lines.append('없음')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')

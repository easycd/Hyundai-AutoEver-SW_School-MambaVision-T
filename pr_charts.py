"""Precision/recall charts using the cached test probabilities."""
import math


def curve_points(targets, probabilities, current_threshold):
    from hand_data import metrics

    # All attainable PR operating points, with equal scores handled together.
    positives = sum(targets)
    pr = [{"Recall": 0.0, "Precision": 1.0, "order": 0}]
    tp = fp = 0
    groups = {}
    for label, probability in zip(targets, probabilities):
        groups.setdefault(probability, []).append(label)
    for probability in sorted(groups, reverse=True):
        labels = groups[probability]
        tp += sum(labels)
        fp += len(labels) - sum(labels)
        pr.append({"Recall": tp / positives if positives else 0.0,
                   "Precision": tp / (tp + fp), "order": len(pr)})

    thresholds = {i / 100 for i in range(101)} | {current_threshold}
    # Include both sides of every jump for the strict probability > threshold rule.
    for probability in probabilities:
        thresholds.add(probability)
        if probability > 0:
            thresholds.add(math.nextafter(probability, -math.inf))
    by_threshold = []
    for threshold in sorted(thresholds):
        values = metrics(targets, [int(p > threshold) for p in probabilities])
        for label, key in [("Precision", "violation_precision"), ("Recall", "violation_recall")]:
            by_threshold.append({"Threshold": threshold, "지표": label, "값": values[key]})
    return pr, by_threshold


def render_pr_charts(results, threshold, scores):
    import altair as alt
    import streamlit as st

    pr, by_threshold = curve_points([r["actual"] for r in results],
                                    [r["violation_probability"] for r in results], threshold)
    selected = {"Recall": scores["violation_recall"], "Precision": scores["violation_precision"],
                "Threshold": threshold}
    scale = alt.Scale(domain=[0, 1])
    pr_line = alt.Chart(alt.Data(values=pr)).mark_line(color="#2563eb", point=True).encode(
        x=alt.X("Recall:Q", scale=scale), y=alt.Y("Precision:Q", scale=scale),
        order="order:Q", tooltip=["Recall:Q", "Precision:Q"])
    pr_chart = pr_line
    if any(r["predicted"] == 1 for r in results):
        marker = alt.Chart(alt.Data(values=[selected])).mark_point(
            color="#f97316", size=160, filled=True).encode(
                x="Recall:Q", y="Precision:Q", tooltip=["Threshold:Q", "Precision:Q", "Recall:Q"])
        pr_chart = pr_line + marker
    threshold_line = alt.Chart(alt.Data(values=by_threshold)).mark_line().encode(
        x=alt.X("Threshold:Q", scale=scale), y=alt.Y("값:Q", scale=scale, title="Score"),
        color=alt.Color("지표:N", scale=alt.Scale(domain=["Precision", "Recall"],
                                                range=["#2563eb", "#16a34a"])),
        order="Threshold:Q", tooltip=["Threshold:Q", "지표:N", "값:Q"])
    rule = alt.Chart(alt.Data(values=[{"Threshold": threshold}])).mark_rule(
        color="#f97316", strokeDash=[5, 3]).encode(x="Threshold:Q")
    current_points = alt.Chart(alt.Data(values=[
        {"Threshold": threshold, "값": selected["Precision"]},
        {"Threshold": threshold, "값": selected["Recall"]},
    ])).mark_point(color="#f97316", size=90, filled=True).encode(x="Threshold:Q", y="값:Q")
    st.subheader("Precision–Recall 곡선")
    left, right = st.columns(2)
    with left:
        st.altair_chart(pr_chart.properties(title="Precision–Recall", height=300), width="stretch")
    with right:
        st.altair_chart((threshold_line + rule + current_points).properties(
            title="Threshold에 따른 Precision / Recall", height=300), width="stretch")
    st.caption(f"주황색은 현재 기준값 {threshold:.2f}의 위치입니다. 왼쪽은 Recall과 Precision의 관계, "
               "오른쪽은 기준값 변화에 따른 두 지표입니다. test 확률을 재사용하므로 추가 학습은 없습니다.")
    st.caption("PR 곡선의 (Recall=0, Precision=1)은 관례적인 시작점입니다. "
               "반칙 판정이 0장일 때 성능 표와 오른쪽 그래프는 Precision을 0으로 표시하며 왼쪽 선택점은 생략합니다.")

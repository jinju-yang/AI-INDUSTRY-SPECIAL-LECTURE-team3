"""평가 스크립트: 라벨링된 CSV로 분류 성능 + 검색 성능 비교
사용법: python evaluate.py data/labeled_sample.csv [임계값]
CSV 열: post_id, claim, gold_label(근거 있음/근거와 모순/판단 불가), gold_evidence_ids(;로 구분, 없으면 빈칸), labeler
"""
import sys, json, csv, os
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from factcheck import Retriever, load_verifier, decide, KO

path = sys.argv[1] if len(sys.argv) > 1 else "data/labeled_sample.csv"
th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
INV = {v: k for k, v in KO.items()}
ORDER = ["근거 있음", "판단 불가", "근거와 모순"]

corpus = json.load(open(os.environ.get("CORPUS", "data/corpus.json"), encoding="utf-8"))
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
R, V = Retriever(corpus), load_verifier()

# ---- 1) 검색 성능: 방식별 Recall@K (정답 근거가 있는 주장만) ----
has_ev = [r for r in rows if r["gold_evidence_ids"].strip()]
print(f"## 검색 성능 (정답 근거 있는 주장 {len(has_ev)}개)")
for mode in ("bm25", "char", "hybrid"):
    out = []
    for K in (1, 3, 5):
        hit = 0
        for r in has_ev:
            gold = set(int(x) for x in r["gold_evidence_ids"].split(";"))
            got = {i for i, _ in R.search(r["claim"], K, mode)}
            hit += bool(gold & got)
        out.append(f"R@{K}={hit/len(has_ev):.2f}" if has_ev else "n/a")
    print(f"  {mode:7s}", "  ".join(out))

# ---- 2) 분류 성능 (검색은 hybrid top-3) ----
y_true, y_pred, conf = [], [], []
for r in rows:
    hits = R.search(r["claim"], 3, "hybrid")
    probs = [V.proba(corpus[i]["text"], r["claim"]) for i, _ in hits]
    lab, c = decide(probs, th)
    y_true.append(r["gold_label"]); y_pred.append(KO[lab]); conf.append(c)
print(f"\n## 분류 성능 (임계값 {th}, 주장 {len(rows)}개)")
print(f"Accuracy {accuracy_score(y_true,y_pred):.3f} | Macro-F1 {f1_score(y_true,y_pred,average='macro',labels=ORDER,zero_division=0):.3f}")
print(classification_report(y_true, y_pred, labels=ORDER, zero_division=0, digits=2))
print("혼동행렬 (행=정답, 열=예측, 순서:", ORDER, ")\n", confusion_matrix(y_true, y_pred, labels=ORDER))

# ---- 3) 책임 있는 AI 지표 ----
n = len(rows)
pred_sup = [i for i in range(n) if y_pred[i] == "근거 있음"]
false_sup = [i for i in pred_sup if y_true[i] != "근거 있음"]
undecided = sum(p == "판단 불가" for p in y_pred)
miss_con = sum(1 for i in range(n) if y_true[i] == "근거와 모순" and y_pred[i] != "근거와 모순")
print("\n## 책임 있는 AI 지표")
print(f"판단 불가 비율: {undecided}/{n}")
print(f"잘못된 '근거 있음'(오판): {len(false_sup)}/{len(pred_sup)}  <- 가장 위험한 오류")
print(f"놓친 모순: {miss_con}/{sum(t=='근거와 모순' for t in y_true)}")

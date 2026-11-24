"""사실 검증 파이프라인 프로토타입 (로컬 테스트용)

흐름: 게시글 → 문장 분리 → 주장 후보 선별 → 근거 검색(BM25 + 문자 TF-IDF) → 주장·근거 관계 분류 → 판정

※ 이 샌드박스에서 불가능한 부분은 자리표시자로 대체했습니다.
  - 주장 추출: Azure OpenAI(LLM) 대신 규칙 기반 필터
  - 의미 검색: Sentence-BERT 대신 문자 n-gram TF-IDF
  - 관계 분류: BERT 대신 TF-IDF + Logistic Regression (KLUE-NLI 학습)
"""
import json, re
import numpy as np
import scipy.sparse as sp
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ["entailment", "neutral", "contradiction"]
KO = {"entailment": "근거 있음", "contradiction": "근거와 모순", "neutral": "판단 불가"}

_kiwi = Kiwi()


def split_sentences(text):
    return [s.text.strip() for s in _kiwi.split_into_sents(text) if s.text.strip()]


def tokens(text):
    keep = {"NNG", "NNP", "NR", "SN", "SL", "VV", "VA"}
    out = []
    for t in _kiwi.tokenize(text):
        if t.tag in keep:
            out.append(t.form)
    return out


# ---------- 1) 주장 후보 선별 (LLM 대체용 규칙) ----------
CUE = ["부작용", "생긴다", "된다", "없다", "좋다", "낫", "효과", "금지", "안전", "위험", "부담", "복용", "먹", "해야", "않", "수 있"]
EXPERIENCE = ["샀어요", "했어요", "갔어요", "봤어요", "샀다"]


def extract_claims(text):
    claims = []
    for s in split_sentences(text):
        if any(e in s for e in EXPERIENCE):
            continue
        if re.search(r"\d", s) or any(c in s for c in CUE):
            claims.append(s)
    return claims


# ---------- 2) 근거 검색 ----------
class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        texts = [c["text"] for c in corpus]
        self.bm25 = BM25Okapi([tokens(t) for t in texts])
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.M = self.vec.fit_transform(texts)

    def search(self, query, k=3, mode="hybrid"):
        """mode: 'bm25' | 'char' | 'hybrid'  (모델 비교 실험용)"""
        b = np.array(self.bm25.get_scores(tokens(query)))
        c = cosine_similarity(self.vec.transform([query]), self.M).ravel()
        norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        score = {"bm25": norm(b), "char": norm(c), "hybrid": 0.5 * norm(b) + 0.5 * norm(c)}[mode]
        idx = np.argsort(-score)[:k]
        return [(int(i), float(score[i])) for i in idx]


# ---------- 3) 관계 분류 (KLUE-NLI 학습) ----------
class Verifier:
    def __init__(self, train_path="data/klue-nli-v1.1_train.json"):
        tr = json.load(open(train_path, encoding="utf-8"))
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        self.vec.fit([x["premise"] for x in tr] + [x["hypothesis"] for x in tr])
        X = self._feats([x["premise"] for x in tr], [x["hypothesis"] for x in tr])
        y = [LABELS.index(x["gold_label"]) for x in tr]
        self.clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, y)

    def _feats(self, prem, hyp):
        P, H = self.vec.transform(prem), self.vec.transform(hyp)
        return sp.hstack([H, P.multiply(H), abs(P - H)]).tocsr()

    def proba(self, premise, claim):
        return self.clf.predict_proba(self._feats([premise], [claim]))[0]  # [ent, neu, con]


def load_verifier(bert_path=None):
    """bert_path(또는 환경변수 BERT_PATH)가 있으면 학습된 BERT, 없으면 옛 TF-IDF 모델을 쓴다."""
    import os
    bert_path = bert_path or os.environ.get("BERT_PATH", "")
    if bert_path:
        from bert_verifier import BertVerifier
        return BertVerifier(bert_path)
    return Verifier()


# ---------- 4) 판정 ----------
def decide(prob_list, threshold):
    """여러 근거 문장 중 가장 확신하는 쪽을 채택, 확신이 임계값 미만이면 판단 불가."""
    best_ent = max(p[0] for p in prob_list)
    best_con = max(p[2] for p in prob_list)
    if max(best_ent, best_con) < threshold:
        return "neutral", max(best_ent, best_con)
    return ("entailment", best_ent) if best_ent >= best_con else ("contradiction", best_con)


def run(text, retriever, verifier, k=3, threshold=0.5):
    results = []
    for claim in extract_claims(text):
        hits = retriever.search(claim, k)
        probs = [verifier.proba(retriever.corpus[i]["text"], claim) for i, _ in hits]
        label, conf = decide(probs, threshold)
        results.append({
            "claim": claim, "label": label, "label_ko": KO[label], "confidence": conf,
            "evidence": [
                {"id": i, "score": s, "text": retriever.corpus[i]["text"],
                 "source": retriever.corpus[i]["source"],
                 "p_ent": float(p[0]), "p_neu": float(p[1]), "p_con": float(p[2])}
                for (i, s), p in zip(hits, probs)
            ],
        })
    return results

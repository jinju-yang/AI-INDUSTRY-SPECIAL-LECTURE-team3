import json
import streamlit as st
import os
from factcheck import Retriever, load_verifier, run

st.set_page_config(page_title="게시글 사실 검증 (프로토타입)", layout="wide")
st.title("온라인 게시글 사실 검증 (프로토타입)")
def esc(t):
    """streamlit 마크다운이 ~(취소선), $(수식)으로 해석하지 않도록 이스케이프"""
    return str(t).replace("~", "\\~").replace("$", "\\$")


st.caption("건강·의약품 정보 시범 분야 · 결과는 참고용이며 최종 판단은 사용자에게 있습니다.")

@st.cache_resource(show_spinner="모델과 근거 인덱스를 준비하는 중...")
def load(bert_path):
    corpus = json.load(open("data/corpus.json", encoding="utf-8"))
    return Retriever(corpus), load_verifier(bert_path)

sample = json.load(open("data/sample_post.json", encoding="utf-8"))["text"]

with st.sidebar:
    bert_path = st.text_input("BERT 모델 폴더 (비우면 옛 TF-IDF 모델)", value=os.environ.get("BERT_PATH", ""))
    th = st.slider("판단 임계값 (낮으면 '판단 불가'가 줄고 오판이 늘어요)", 0.30, 0.90, 0.50, 0.05)
    k = st.slider("검색할 근거 문장 수", 1, 5, 3)

try:
    R, V = load(bert_path.strip())
except Exception as e:
    st.error(f"모델을 불러오지 못했어요: {e}")
    st.stop()
st.sidebar.caption("판정 모델: " + ("BERT" if bert_path.strip() else "TF-IDF(옛 모델)"))

text = st.text_area("게시글 본문을 붙여넣으세요", value=sample, height=180, key="post")
if st.button("검증하기", type="primary"):
    results = run(text, R, V, k=k, threshold=th)
    ICON = {"entailment": "🟢", "contradiction": "🔴", "neutral": "⚪"}
    st.subheader(f"검증이 필요한 문장 {len(results)}개")
    for r in results:
        st.markdown(f"{ICON[r['label']]} **{r['label_ko']}** (확신도 {r['confidence']:.2f}) — {esc(r['claim'])}")
        with st.expander("근거 보기"):
            for e in r["evidence"]:
                st.markdown(f"· {esc(e['text'])}")
                st.caption(f"출처: {esc(e['source'])} | 검색점수 {e['score']:.2f} | 근거 있음 {e['p_ent']:.2f} / 판단 불가 {e['p_neu']:.2f} / 모순 {e['p_con']:.2f}")

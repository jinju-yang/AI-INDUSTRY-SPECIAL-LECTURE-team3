# GLP-1 게시글 사실 검증 — 모델 파트

파이프라인: 문장 분리 → 주장 선택(규칙) → 근거 검색(BM25 + 글자 n-gram) → 3분류(근거 있음 / 판단 불가 / 근거와 모순) → 임계값 판정 → Streamlit 화면

## 파일
- `factcheck.py` 핵심 파이프라인 (`load_verifier()`가 BERT_PATH 유무로 BERT/TF-IDF 선택)
- `bert_verifier.py`, `finetune_nli.py` BERT 판정기와 학습 (1단계 KLUE-NLI, 2단계 도메인)
- `llm_judge.py` LLM 판정 비교 (Azure OpenAI 또는 프롬프트 내보내기)
- `evaluate.py` 검색 Recall, 정확도, Macro-F1, 위험 오류(근거 없는데 근거 있음 / 놓친 모순)
- `app.py` Streamlit 화면, `labeling_app.py`·`agreement.py`·`labeling_guide.md` 라벨링 도구
- `data/corpus_team.json` 근거 문장 58개, `team_claims_all.json` 라벨 65개(A/B), `labeled_team_agree.csv` 두 라벨러 일치 51개

## 실행
```
pip install -r requirements.txt   # kiwipiepy rank_bm25 scikit-learn streamlit transformers
set CORPUS=data/corpus_team.json
python evaluate.py data/labeled_team_agree.csv
set BERT_PATH=out/klue_model      # Colab에서 finetune_nli.py klue 로 학습
```

## 현재 결과 (팀 데이터, KLUE-NLI만 학습한 BERT, 임계값 0.5)
- 검색 Recall@3: 정답 근거가 있는 31개 중 0.90
- 판정 정확도 0.53 (다수 클래스 기준 0.51), Macro-F1 0.36 → 도메인 불일치로 서비스 판정기로 부족
- 다음: LLM 판정 비교, 도메인 학습, 모순 라벨 확보

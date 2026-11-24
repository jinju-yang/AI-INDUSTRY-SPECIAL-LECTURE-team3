"""LLM 판정 비교 실험 (BERT 대신/와 비교용)
같은 검색 결과(top-k 근거)를 LLM에 주고 근거 있음/판단 불가/근거와 모순 중 하나를 고르게 한 뒤, 사람 라벨과 비교한다.

사용법
 1) Azure OpenAI로 바로 실행
      set AZURE_OPENAI_ENDPOINT=https://<리소스>.openai.azure.com/
      set AZURE_OPENAI_KEY=<키>
      set AZURE_OPENAI_DEPLOYMENT=<배포이름>
      pip install openai
      python llm_judge.py run --out data/llm_pred.json
 2) API 없이: 프롬프트만 뽑아서 아무 LLM 챗에 붙여넣기
      python llm_judge.py export            # data/llm_prompts.txt 생성
      (LLM 답을 data/llm_pred.json 형식으로 저장)
 3) 평가
      python llm_judge.py eval data/llm_pred.json
data/llm_pred.json 형식: [{"id":"C001","label":"근거 있음","evidence":3,"reason":"..."}, ...]
"""
import sys, os, json, re, argparse
from collections import Counter
from factcheck import Retriever

CORPUS = os.environ.get("CORPUS", "data/corpus_team.json")
CLAIMS = os.environ.get("CLAIMS", "data/team_claims_all.json")
LABELS = ["근거 있음", "판단 불가", "근거와 모순"]

SYSTEM = """너는 온라인 게시글의 의학 관련 주장을 근거 문장과 비교해 사실 여부를 판정하는 검증기다.
반드시 아래 근거 문장에 적힌 내용만 사용한다. 네가 아는 배경지식으로 보충하거나 추측하지 않는다.

판정 기준
- 근거 있음: 주장의 핵심 내용과 수치·조건이 모두 근거 문장에서 확인된다.
- 근거와 모순: 근거 문장이 주장과 직접 반대되거나, 수치·조건·대상이 서로 다르다.
- 판단 불가: 근거 문장에 없는 내용이 하나라도 주장에 들어 있거나, 근거만으로 확인할 수 없다.
확신이 없으면 근거 있음 대신 판단 불가를 고른다.

출력은 JSON 한 줄만: {"label": "근거 있음|판단 불가|근거와 모순", "evidence": <가장 관련 있는 근거 번호 또는 null>, "reason": "한 문장"}"""


def build_user(claim, hits):
    ev = "\n".join(f"[근거{n}] {t}" for n, t in hits)
    return f"[주장]\n{claim}\n\n[근거 문장]\n{ev}"


def retrieve(k=3):
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    R = Retriever(corpus)
    claims = json.load(open(CLAIMS, encoding="utf-8"))
    out = []
    for c in claims:
        hits = [(i + 1, corpus[i]["text"]) for i, _ in R.search(c["claim"], k)]  # 근거 번호는 엑셀 번호(1부터)
        out.append((c, hits))
    return out


def cmd_export(a):
    rows = retrieve(a.k)
    with open("data/llm_prompts.txt", "w", encoding="utf-8") as f:
        f.write("# 아래 [시스템] 프롬프트를 먼저 LLM에 주고, 이어서 [ID] 블록을 하나씩 보내 JSON 답을 받으세요.\n\n[시스템]\n" + SYSTEM + "\n\n")
        for c, hits in rows:
            f.write(f"===== [ID] {c['id']} =====\n{build_user(c['claim'], hits)}\n\n")
    print("data/llm_prompts.txt 생성:", len(rows), "개")


def cmd_run(a):
    from openai import AzureOpenAI
    cli = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_KEY"],
                      api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    dep = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    preds = []
    for c, hits in retrieve(a.k):
        r = cli.chat.completions.create(model=dep, temperature=0,
                                        messages=[{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": build_user(c["claim"], hits)}])
        txt = r.choices[0].message.content
        try:
            j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            j = {"label": "판단 불가", "evidence": None, "reason": "파싱 실패: " + txt[:80]}
        j["id"] = c["id"]; preds.append(j)
        print(c["id"], j["label"])
    json.dump(preds, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_eval(a):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
    claims = {c["id"]: c for c in json.load(open(CLAIMS, encoding="utf-8"))}
    preds = {p["id"]: p for p in json.load(open(a.pred, encoding="utf-8"))}
    ids = [i for i in claims if i in preds]
    ag = [i for i in ids if claims[i]["A"] == claims[i]["B"]]
    y = [claims[i]["A"] for i in ag]; p = [preds[i]["label"] for i in ag]
    print(f"## 사람 두 명이 일치한 {len(ag)}개 기준")
    print("Accuracy %.3f | Macro-F1 %.3f | kappa %.3f" % (accuracy_score(y, p), f1_score(y, p, average="macro", labels=LABELS), cohen_kappa_score(y, p, labels=LABELS)))
    print("다수 클래스(판단 불가)만 찍었을 때 정확도 %.3f" % (Counter(y).most_common(1)[0][1] / len(y)))
    print(LABELS); print(confusion_matrix(y, p, labels=LABELS))
    fs = sum(1 for a_, b_ in zip(y, p) if b_ == "근거 있음" and a_ != "근거 있음")
    mc = sum(1 for a_, b_ in zip(y, p) if a_ == "근거와 모순" and b_ != "근거와 모순")
    print(f"가장 위험한 오류: 정답이 근거 있음이 아닌데 근거 있음으로 판정 {fs}건 / 놓친 모순 {mc}/{sum(1 for a_ in y if a_=='근거와 모순')}")
    for who in ("A", "B"):
        yy = [claims[i][who] for i in ids]; pp = [preds[i]["label"] for i in ids]
        print(f"라벨러 {who} 전체 {len(ids)}개 기준 Accuracy %.3f" % accuracy_score(yy, pp))
    print("\n## 틀린 것 (일치 기준)")
    for i in ag:
        if preds[i]["label"] != claims[i]["A"]:
            print(i, "정답", claims[i]["A"], "| 예측", preds[i]["label"], "|", claims[i]["claim"][:50], "|", str(preds[i].get("reason", ""))[:60])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("export", cmd_export), ("run", cmd_run)):
        s = sp.add_parser(n); s.add_argument("-k", type=int, default=3); s.add_argument("--out", default="data/llm_pred.json"); s.set_defaults(f=f)
    s = sp.add_parser("eval"); s.add_argument("pred"); s.set_defaults(f=cmd_eval)
    a = ap.parse_args(); a.f(a)

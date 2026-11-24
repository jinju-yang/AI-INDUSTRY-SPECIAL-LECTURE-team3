"""BERT 미세조정 (Colab GPU용). 두 단계로 나눠 실행한다.

  1단계 (일반 NLI):  python finetune_nli.py klue
      KLUE-NLI 학습 25k쌍으로 klue/roberta-base 를 학습 -> out/klue_model
  2단계 (도메인):    python finetune_nli.py domain data/labels_A.csv data/labels_B.csv ...
      1단계 모델에서 이어서, 직접 라벨링한 CSV로 추가 학습 -> out/domain_model

Colab 사용: 런타임 > 런타임 유형 > T4 GPU 선택 후
  !pip -q install transformers datasets accelerate scikit-learn kiwipiepy rank_bm25
  !python finetune_nli.py klue
"""
import csv, json, os, sys, urllib.request
import numpy as np

LABELS = ["entailment", "neutral", "contradiction"]          # factcheck.py 와 같은 순서
KO2EN = {"근거 있음": "entailment", "판단 불가": "neutral", "근거와 모순": "contradiction"}
BASE = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_{}.json"
MODEL = "klue/roberta-base"


def load_klue(split):
    path = f"data/klue-nli-v1.1_{split}.json"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(BASE.format(split), path)
    d = json.load(open(path, encoding="utf-8"))
    return [(x["premise"], x["hypothesis"], LABELS.index(x["gold_label"])) for x in d]


def build_domain_pairs(csv_paths, corpus_path="data/corpus.json"):
    """라벨링 CSV -> (premise, claim, label) 목록.
    근거 있음/모순: 정답 근거 문서를 premise 로 사용 (여러 개면 각각 한 쌍).
    판단 불가: 검색 top-3 문서를 premise 로 사용 (근거가 있어도 주장을 못 뒷받침하는 사례를 배우게 함)."""
    from factcheck import Retriever
    corpus = json.load(open(corpus_path, encoding="utf-8"))
    R = Retriever(corpus)
    pairs = []
    for p in csv_paths:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            label = LABELS.index(KO2EN[r["gold_label"]])
            claim = r["claim"].strip()
            ids = [int(i) for i in r["gold_evidence_ids"].split(";") if i.strip() != ""]
            if label != 1 and ids:
                for i in ids:
                    pairs.append((corpus[i]["text"], claim, label))
            else:
                for i, _ in R.search(claim, 3):
                    pairs.append((corpus[i]["text"], claim, 1))
    return pairs


def train(pairs_train, pairs_dev, init, out, epochs, lr, bs=32):
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments)
    from sklearn.metrics import f1_score, accuracy_score
    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=3)

    class DS(torch.utils.data.Dataset):
        def __init__(self, pairs):
            enc = tok([p for p, _, _ in pairs], [h for _, h, _ in pairs],
                      truncation=True, max_length=128, padding="max_length", return_token_type_ids=False)
            self.enc, self.y = enc, [y for _, _, y in pairs]
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    def metrics(ev):
        pred, y = ev.predictions.argmax(1), ev.label_ids
        sup = pred == 0
        return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
                "false_support": float((y[sup] != 0).mean()) if sup.any() else 0.0,   # 위험 지표
                "missed_contradiction": float((pred[y == 2] != 2).mean()) if (y == 2).any() else 0.0}

    args = TrainingArguments(output_dir=out + "_ckpt", num_train_epochs=epochs, learning_rate=lr,
                             per_device_train_batch_size=bs, per_device_eval_batch_size=64,
                             eval_strategy="epoch", save_strategy="no", fp16=torch.cuda.is_available(),
                             warmup_steps=int(0.1 * len(pairs_train) / bs * epochs), weight_decay=0.01, logging_steps=100, report_to="none")
    tr = Trainer(model=model, args=args, train_dataset=DS(pairs_train), eval_dataset=DS(pairs_dev),
                 compute_metrics=metrics)
    tr.train()
    print("최종 평가:", tr.evaluate())
    model.save_pretrained(out); tok.save_pretrained(out)
    print("저장:", out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "klue":
        train(load_klue("train"), load_klue("dev"), MODEL, "out/klue_model", epochs=3, lr=2e-5)
    elif stage == "domain":
        pairs = build_domain_pairs(sys.argv[2:])
        rng = np.random.RandomState(0); rng.shuffle(pairs)
        cut = int(len(pairs) * 0.8)       # 주의: 같은 주장이 train/dev 양쪽에 섞이지 않게, 실제로는 주장 단위로 나누는 게 안전
        train(pairs[:cut], pairs[cut:], "out/klue_model", "out/domain_model", epochs=5, lr=1e-5, bs=16)

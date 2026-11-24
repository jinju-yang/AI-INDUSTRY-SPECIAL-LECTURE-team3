"""학습된 BERT 를 factcheck.Verifier 와 같은 인터페이스(proba)로 감싼 것.
사용: from bert_verifier import BertVerifier; verifier = BertVerifier("out/domain_model")
(app.py / evaluate.py 에서 Verifier() 대신 이걸 넣으면 된다)"""
class BertVerifier:
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.T = 1.0   # 확률 보정용 temperature (calibrate 로 정함)

    def proba(self, premise, claim):
        enc = self.tok(premise, claim, truncation=True, max_length=128, return_tensors="pt", return_token_type_ids=False)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0] / self.T
        return self.torch.softmax(logits, -1).numpy()      # [ent, neu, con]

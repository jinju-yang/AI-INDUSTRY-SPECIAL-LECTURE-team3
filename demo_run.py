import json, time, os
from factcheck import *
t=time.time()
corpus=json.load(open(os.environ.get('CORPUS','data/corpus.json'),encoding='utf-8'))
post=json.load(open(os.environ.get('POST','data/sample_post.json'),encoding='utf-8'))
R=Retriever(corpus); V=load_verifier()
print("모델·인덱스 준비 %.0f초"%(time.time()-t))
for th in (0.5,):
    res=run(post['text'],R,V,k=3,threshold=th)
    print(f"\n=== 임계값 {th} | 추출된 주장 {len(res)}개 ===")
    hit=cor=0
    for r in res:
        g=next((x for x in post['gold'] if x['key'] in r['claim']),None)
        ev_ids=[e['id'] for e in r['evidence']]
        gl=g['label'] if g else '-'
        ok = (g is not None and g['label']==r['label'])
        print(f"- {r['claim']}")
        print(f"   예측: {r['label_ko']} (확신 {r['confidence']:.2f}) | 정답: {KO.get(gl,'-')} | 검색 top3 근거id {ev_ids}" + (f" (정답근거 {g['ev']})" if g else ''))
        for e in r['evidence'][:1]:
            print(f"   top1 근거[{e['id']}]: {e['text'][:48]}…  (ent {e['p_ent']:.2f} / neu {e['p_neu']:.2f} / con {e['p_con']:.2f})")
    gold_claims=[g for g in post['gold']]
    print("\n정답 주장 %d개 중 추출된 수:" % len(gold_claims), sum(any(g['key'] in r['claim'] for r in res) for g in gold_claims))
    # 검색 Recall@3 (근거가 있는 주장만)
    rec=[];
    for g in gold_claims:
        if not g['ev']: continue
        r=next((r for r in res if g['key'] in r['claim']),None)
        if r: rec.append(any(e['id'] in g['ev'] for e in r['evidence']))
    print("검색 Recall@3 (근거 있는 주장 기준): %d/%d"%(sum(rec),len(rec)))
    acc=[ (next((x for x in gold_claims if x['key'] in r['claim']),{'label':None})['label']==r['label']) for r in res if any(x['key'] in r['claim'] for x in gold_claims)]
    print("판정 일치: %d/%d"%(sum(acc),len(acc)))

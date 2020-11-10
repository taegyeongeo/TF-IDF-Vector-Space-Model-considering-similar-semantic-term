from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from eunjeon import Mecab
from tqdm import tqdm
import time

with open("corpus.txt","rb") as f:
    documents=pickle.load(f)

with open("co_occur.txt","rb") as f:
    co_occur=pickle.load(f)

with open("co_occur_voca.txt","rb") as f:
    co_occur_voca=pickle.load(f)

co_occur_voca_keys=[key for key in co_occur_voca.keys()]

mecab=Mecab()
vectorizer = TfidfVectorizer(analyzer='word')
tfidf_dict=vectorizer.fit_transform(['\n'.join(contents) for _,_,contents in documents.values()])
title_dict=vectorizer.transform([title for title,_,_ in documents.values()])
vocab=vectorizer.vocabulary_
vocab_size=len(vocab)

def search(q):
    start=time.time()
    query_token=mecab.nouns(q)
    query=[' '.join(query_token)]
    query_vec=vectorizer.transform(query).toarray()
    relevence_term_with_query=dict()
    for token in query_token:
        try:
            co_occur_vec=co_occur.getcol(co_occur_voca[token])
        except KeyError:
            continue
        subidx=co_occur_vec.nonzero()[0]
        res=[]
        for idx in subidx:
            val=co_occur_vec[idx,0]
            res.append((idx,val))   
        textrank=[(co_occur_voca_keys[i],count) for i,count in sorted(res,key=lambda x:x[1],reverse=True)[:11]]
        relevence_term_with_query[token]=textrank

    relevance_pages=set()
    for token in query_token:
        try:
            relevance_pages.update(tfidf_dict.getcol(vocab[token]).nonzero()[0].tolist())
            relevance_pages.update(title_dict.getcol(vocab[token]).nonzero()[0].tolist())
        except KeyError:
            pass

    w=0.5
    res=[]
    t=tqdm(total=len(relevance_pages))
    for i in relevance_pages:
        t.update(1)
        sim_a=0; sim_b=0
        for token in query_token:
            try:
                sim_a+=query_vec[0,vocab[token]]*title_dict[i,vocab[token]]
                sim_b+=query_vec[0,vocab[token]]*tfidf_dict[i,vocab[token]]
            except KeyError:
                continue
            sum=0
            for _,cnt in relevence_term_with_query[token]:
                sum+=cnt
            for item,cnt in relevence_term_with_query[token]:
                if item is not token:
                    try:
                        sim_b+=(cnt/sum)*title_dict[i,vocab[item]]
                    except KeyError:
                        sim_b+=(cnt/sum)
                    
        sim=w*sim_a+(1-w)*sim_b
        if sim<0.1: continue
        res.append((i,sim))
    docs=list(documents.values())
    res_df=pd.DataFrame([(docs[id][0],docs[id][1],docs[id][2],sim) for id,sim in sorted(res,key=lambda x:x[1],reverse=True)[0:20]],columns=['title','url','contents','similarity'])
    print("\n검색 수행시간: ", round(time.time()-start,3),"sec",sep='')
    return res_df

    search('대한민국 국보 1호')

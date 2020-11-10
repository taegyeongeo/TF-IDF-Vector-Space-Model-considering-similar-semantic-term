import json
import regex as re
from eunjeon import Mecab
import os
from tqdm import tqdm
import pickle

def numToString(n):
    for i in range(n):
        yield ''.join('0' for _ in range(2-len(str(i))))+str(i)

dirIdx=['AA','AB','AC','AD','AE','AF','AG','AH']
mecab=Mecab()
documents=dict()
keyword_dict=dict()
t=tqdm(total=512761)

documents=dict()
# 단어빈도계산 tf
for dirChar in dirIdx:
    dir=f'corpus\{dirChar}'
    numOfFiles=len(os.listdir(dir))
    for wikiNum in numToString(numOfFiles):
        with open(f"corpus\{dirChar}\wiki_{wikiNum}",encoding='utf-8') as f:
            for doc in f:
                data=json.loads(doc)
                contents=[' '.join([token for token in mecab.nouns(re.sub('[^A-Za-z0-9가-힣]', '', line)) if token not in stopwords]) for line in filter(None,data['text'].split('\n'))]
                title=data['title']
                url=data['url']
                id=data['id']
                documents[id]=(title,url,contents)
                t.update(1)

with open("corpus.txt","wb") as f:
    data=pickle.dumps(documents)
    f.write(data)
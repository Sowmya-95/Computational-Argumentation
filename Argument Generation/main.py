import pandas as pd
import json
from transformers import pipeline

summarizer = pipeline("summarization")

def main():
    ESSAYS_PROMT_CORPUS_PATH = './data/essay-prompt-corpus.json'
    with open(ESSAYS_PROMT_CORPUS_PATH, "r") as f:
        data = json.load(f)
    TRAINTESTSPLITPATH = './data/train-test-split.csv'
    trainTestSplit = pd.read_csv(TRAINTESTSPLITPATH,sep=';')
    
    # Split
    test_ids = sorted([int(fn[-3:]) for fn in trainTestSplit[trainTestSplit.SET == "TEST"].ID.values])
    test_split = list(filter(lambda x: x["id"] in test_ids, data))
    test_split = sorted(test_split, key=lambda x: int(x["id"]))    
    
    summary = []
    for ess in test_split:
        summary.append(genSummary(ess['text']))
    
    generatePredictionJSON(summary,test_ids)


def generatePredictionJSON(predictions,ids):
    output = {
        'id': [],
        'prompt': []
    }
    for p in range(len(predictions)):
        output['id'].append(int(ids[p]))
        output['prompt'].append(str(predictions[p]))
    
    o = pd.DataFrame(output)
    o.to_json('predictions.json',orient='records')

def genSummary(essay):
    sum = summarizer(essay)
    sum = sum[0]['summary_text']
    sum = sum.split('.')[0] 
    return sum 


if __name__ == '__main__':
    main()

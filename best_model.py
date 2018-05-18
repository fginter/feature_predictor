import argparse
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('models', nargs='+', help='.history.json files')
    args = parser.parse_args()
    scores=[]
    for fname in args.models:
        with open(fname) as f:
            epochs,hist=json.load(f)
            score=min(hist["val_loss"])
            scores.append((score,fname))
    scores.sort()
    best_score,model_name=scores[0]
    print(model_name.replace(".history.json",""))
    

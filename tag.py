import model as model
import data
import random
import numpy
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import sys
import io
import select

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def tagged_output(out,feat_val_dict):
    """out is output of the model"""
    feat_val_dict_rev={}
    for feat, val_dict in feat_val_dict.items():
        feat_val_dict_rev[feat]=dict((id,v) for v,id in val_dict.items())

    output_features=[feat for feat in sorted(feat_val_dict.keys())]
    
    out_pred=numpy.vstack([numpy.argmax(p,axis=-1) for p in out]).T  #examples by output
    assert out_pred.shape[1]==len(output_features)

    result=[]
    for out_row in out_pred:
        row_tags=[]
        for feat,pred in zip(output_features,out_row):
            label=feat_val_dict_rev[feat][pred]
            if label!="<UNSET>":
                row_tags.append(feat+"="+label)
        
        if row_tags:
            result.append("|".join(sorted(row_tags,key=lambda item:item.split("=",1)[0].lower())))
        else:
            result.append("_")
    return result

def nonblocking_batches(f=sys.stdin,timeout=10,batch_lines=100000):
    """Yields batches of the input (as string), always ending with an empty line.
       Batch is formed when at least batch_lines are read, or when no input is seen in timeour seconds
       Stops yielding when f is closed"""
    line_buffer=[]
    while True:
        ready_to_read=select.select([f], [], [], timeout)[0] #check whether f is ready to be read, wait at least timeout (otherwise we run a crazy fast loop)
        if not ready_to_read:
            # Stdin is not ready, yield what we've got, if anything
            if line_buffer:
                yield "".join(line_buffer)
                line_buffer=[]
            continue #next try
        
        # f is ready to read!
        # Since we are reading conll, we should always get stuff until the next empty line, even if it means blocking read
        while True:
            line=f.readline()
            if not line: #End of file detected --- I guess :D
                if line_buffer:
                    yield "".join(line_buffer)
                    return
            line_buffer.append(line)
            if not line.strip(): #empty line
                break

        # Now we got the next sentence --- do we have enough to yield?
        if len(line_buffer)>batch_lines:
            yield "".join(line_buffer) #got plenty
            line_buffer=[]

def tag_conllu(inp,out,p,word_vec_vocab,errors=None):
    inp_dict,outp_dict,outp_features=data.prep_data(inp,args.model_file+".dicts.json",word_vec_vocab,word_seq_len=p.word_seq_len(),shuffle=False)
    print("inp shape words",inp_dict["inp_char_seq"].shape,file=sys.stderr)
    preds=p.model.predict(inp_dict)
    tags=tagged_output(preds,p.feat_val_dict)
    correct=0
    total=0
    idx=0
    for tree,comments in inp:
        if comments:
            print("\n".join(comments),file=out)
        for word in tree:
            if errors is not None:
                total+=1
                if tags[idx]!=word[FEATS]:
                    err_list=errors.setdefault((tags[idx],word[FEATS]),[0,set()])
                    err_list[0]+=1
                    err_list[1].add(word[FORM])
                else:
                    correct+=1
            print(*word[:FEATS],tags[idx],*word[FEATS+1:],sep="\t",file=out)
            idx+=1
        print(file=out)
    out.flush()
    return correct, total

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model-file')
    parser.add_argument('--embeddings', help='.vector or .bin')
    parser.add_argument('--errstats',default=False,action="store_true",help="You are tagging test set, print stats")
    parser.add_argument('inputfiles',nargs="*",help="(IGNORED AS OF NOW) list of files to parse")
    args = parser.parse_args()

    p=model.Predictor()
    p.load_model(args.model_file)
    print(p.word_emb_dim(),file=sys.stderr)
    #l=p.model.get_layer("emb_word")
    #print("EMB LAYER CONFIG",p.get_config()["batch_input_shape"])
    try:
        word_emb_length,word_emb_dim=p.word_emb_dim()
        assert instanceof(word_emb_length,int) and word_emb_length>2
    except:
        word_emb_length=p.model.get_layer("emb_word").get_config()["input_dim"] #some older saved models don't have word_emb_dim()
        word_emb_dim=None
    word_embeddings=data.read_embeddings(args.embeddings,word_emb_length-2) #-2 because two dimensions will be added
    del word_embeddings.vectors #we should never need these, we are only after the vocabulary here, really
    #print(p.model.summary(),file=sys.stderr)
    print("wordlen/model",p.word_seq_len(),file=sys.stderr)

    if args.errstats:
        err={}
    else:
        err=None

    correct=0
    total=0

    print("INPUTFILES:",args.inputfiles,file=sys.stderr)
    if not args.inputfiles:
        for batch in nonblocking_batches(timeout=0.5,batch_lines=200000):
            inp=list(data.read_conll(batch.split("\n")))
            print("input file batch length",len(inp),"trees",file=sys.stderr)
            batch_correct,batch_total=tag_conllu(inp,sys.stdout,p,word_embeddings.vocab,err)
            correct+=batch_correct
            total+=batch_total
          
    
    if args.errstats:
        print("ACC={:.2f}%  ({}/{})".format(correct/total*100,correct,total),file=sys.stderr)
        for (pred,gold),err_list in sorted(err.items(),key=lambda item: item[1][0], reverse=True):
            print(err_list[0],"       GS=",gold,"       PRD=",pred,"     ",", ".join(sorted(err_list[1])[:4]),file=sys.stderr)

    

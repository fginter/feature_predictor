import sys
import json

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conll(inp,max_sent=0,drop_tokens=True):
    comments=[]
    sent=[]
    yielded=0
    for line in inp:
        line=line.strip()
        if line.startswith("#"):
            comments.append(line)
        elif not line:
            if sent:
                yield sent,comments
                yielded+=1
                if max_sent>0 and yielded==max_sent:
                    break
                sent,comments=[],[]
        else:
            cols=line.split("\t")
            if drop_tokens and "-" in cols[ID]:
                continue
            sent.append(cols)
    else:
        if sent:
            yield sent,comments


def build_dicts(inp):
    char_dict={"<PAD>":0,"<OOV>":1}
    pos_dict={"<OOV>":0}
    deprel_dict={"<OOV>":0}
    feat_val_dict={} #"number" ->  {"<UNSET>":0,"sg":1}
    for tree,comments in read_conll(inp):
        for cols in tree:
            for char in cols[FORM]:
                char_dict.setdefault(char,len(char_dict))
            pos_dict.setdefault(cols[UPOS],len(pos_dict))
            deprel_dict.setdefault(cols[DEPREL],len(deprel_dict))
            if cols[FEATS]!="_":
                for feat_val in cols[FEATS].split("|"):
                    feat,val=feat_val.split("=",1)
                    feat_dict=feat_val_dict.setdefault(feat,{"<UNSET>":0})
                    feat_dict.setdefault(val,len(feat_dict))
    return char_dict,pos_dict,deprel_dict,feat_val_dict

def vectorize_word(cols,output_features,char_dict,pos_dict,deprel_dict,feat_val_dict):
    """ `cols`  one line of conllu"""
    #Stuff on input
    char_seq=[char_dict.get(char,char_dict["<OOV>"]) for char in cols[FORM]]
    pos=pos_dict.get(cols[UPOS],pos_dict["<OOV>"])
    deprel=deprel_dict.get(cols[DEPREL],deprel_dict["<OOV>"])
    #Stuff on output
    outputs=[]
    example_feats={}
    if cols[FEATS]!="_":
        for feat_val in cols[FEATS].split("|"):
            feat,val=feat_val.split("=",1)
            example_feats[feat]=val
    for feat in output_features: #The feature we want
        if feat in example_feats: #yes it was set!
            feat_dict=feat_val_dict[feat]
            outputs.append(feat_dict.get(example_feats[feat],feat_dict["<UNSET>"])) #Unknown feature, guess we pretend unset...?
        else:
            #No it was not set
            outputs.append(feat_val_dict[feat]["<UNSET>"])
    return [char_seq,pos,deprel],outputs
    
    
def vectorize_data(inp,dicts_filename,output_features=None):
    with open(dicts_filename,"rt") as f:
        char_dict,pos_dict,deprel_dict,feat_val_dict=json.load(f)
    if output_features is None:
        output_features=[feat for feat in sorted(feat_val_dict.keys())]
    result=[]
    for tree,comments in read_conll(inp):
        for cols in tree:
            result.append(vectorize_word(cols,output_features,char_dict,pos_dict,deprel_dict,feat_val_dict))
    return result

if __name__=="__main__":
    #char_dict,pos_dict,deprel_dict,feat_val_dict=build_dicts(sys.stdin)
    #with open("dicts_fi.json","wt") as f:
    #    json.dump([char_dict,pos_dict,deprel_dict,feat_val_dict],f)

    ##TEST
    data=vectorize_data(sys.stdin,"dicts_fi.json")
    print("Len=",len(data))
    for d in data[1000:1005]:
        print(d,end="\n\n")
    

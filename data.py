import sys
import json
import random
import numpy
import numpy.random
import keras.utils
from keras.preprocessing.sequence import pad_sequences
import model
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conll(inp,max_sent=0,drop_tokens=True,drop_nulls=True):
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
            if drop_nulls and "." in cols[ID]:
                continue
            sent.append(cols)
    else:
        if sent:
            yield sent,comments

def read_embeddings(embeddings_filename,max_rank_emb):
    """Reads .vector or .bin file, modifies it to include <OOV> and <PADDING>"""
    if embeddings_filename.endswith(".bin"):
        binary=True
    else:
        binary=False
    gensim_vectors=KeyedVectors.load_word2vec_format(embeddings_filename, binary=binary, limit=max_rank_emb)
    gensim_vectors.vocab["<OOV>"]=Vocab(index=1)
    gensim_vectors.vocab["<PADDING>"]=Vocab(index=0)
    for word_record in gensim_vectors.vocab.values():
        word_record.index+=2
    two_random_rows=numpy.random.uniform(low=-0.01, high=0.01, size=(2,gensim_vectors.vectors.shape[1]))
    # stack the two rows, and the embedding matrix on top of each other
    gensim_vectors.vectors=numpy.vstack([two_random_rows,gensim_vectors.vectors])
    gensim_vectors.vectors=keras.utils.normalize(gensim_vectors.vectors,axis=0)
    gensim_vectors.vectors=keras.utils.normalize(gensim_vectors.vectors)
    return gensim_vectors
    


            
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

def vectorize_word(cols,left_deps,right_deps,output_features,char_dict,pos_dict,deprel_dict,feat_val_dict,word_vec_vocab):
    """ `cols`  one line of conllu"""
    #Stuff on input
    char_seq=[char_dict.get(char,char_dict["<OOV>"]) for char in cols[FORM]]
    left_deprel=[deprel_dict.get(deprel,deprel_dict["<OOV>"]) for deprel in left_deps]
    right_deprel=[deprel_dict.get(deprel,deprel_dict["<OOV>"]) for deprel in right_deps]
    pos=pos_dict.get(cols[UPOS],pos_dict["<OOV>"])
    deprel=deprel_dict.get(cols[DEPREL],deprel_dict["<OOV>"])
    if cols[FORM] in word_vec_vocab:
        word=word_vec_vocab[cols[FORM]].index
    elif cols[FORM].lower() in word_vec_vocab:
        word=word_vec_vocab[cols[FORM].lower()].index
    else:
        word=word_vec_vocab["<OOV>"].index
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
    return [char_seq,word,left_deprel,right_deprel,pos,deprel],outputs
    
    
def vectorize_data(inp,dicts_filename,word_vec_vocab):
    """ `word_vec_vocab`  is gensim's KeyedVectors.vocab with <OOV> and <PADDING> present"""
    with open(dicts_filename,"rt") as f:
        char_dict,pos_dict,deprel_dict,feat_val_dict=json.load(f)
    
    output_features=[feat for feat in sorted(feat_val_dict.keys())]
    result=[]
    for tree,comments in inp:
        deprels=[[] for _ in range(len(tree))]
        for row_idx,cols in enumerate(tree):
            if cols[HEAD]!="0":
                deprels[int(cols[HEAD])-1].append((row_idx,cols[DEPREL])) #index by head, list of deprels
        for row_idx,cols in enumerate(tree):
            left_deps=[deprel for (deprel_idx,deprel) in deprels[row_idx] if deprel_idx<row_idx]
            right_deps=[deprel for (deprel_idx,deprel) in deprels[row_idx] if deprel_idx>row_idx]
            result.append(vectorize_word(cols,left_deps,right_deps,output_features,char_dict,pos_dict,deprel_dict,feat_val_dict,word_vec_vocab))
    return result, output_features

def get_inp_outp(vectorized_data,output_features,word_seq_len,shuffle=False):
    """vectorized_data - (data,feature names) produced by vectorize_data()
       returns ready-made dictionaries of inputs and outputs named by layer
       word_seq_len can be None for max padding"""
    if shuffle:
        random.shuffle(vectorized_data)
    inputs=numpy.array([item[0] for item in vectorized_data])
    inputs_dict={"inp_char_seq":pad_sequences(inputs[:,0],padding="pre",maxlen=word_seq_len),\
                 "inp_word":inputs[:,1],\
                 "inp_left_deps":pad_sequences(inputs[:,2],padding="pre",maxlen=5),\
                 "inp_right_deps":pad_sequences(inputs[:,3],padding="post",maxlen=5),\
                 "inp_pos":inputs[:,4],\
                 "inp_deprel":inputs[:,5]}
    outputs=numpy.array([item[1] for item in vectorized_data])
    outputs_dict=dict((("out_"+model.normname(feat),outputs[:,i]) for i,feat in enumerate(output_features)))
    return inputs_dict,outputs_dict


def prep_data(inp,dicts_filename,word_vec_vocab,word_seq_len=None,shuffle=False):
    data,output_features=vectorize_data(inp,dicts_filename,word_vec_vocab)
    inputs_dict,outputs_dict=get_inp_outp(data,output_features,word_seq_len,shuffle)
    return inputs_dict,outputs_dict,output_features
    

if __name__=="__main__":
    pass

import tensorflow as tf


import sys
import argparse
import data

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed, Flatten, GlobalMaxPooling1D
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape,Dropout
from keras.optimizers import SGD, Adam
from keras.initializers import Constant
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.regularizers import l2 as L2Reg


import numpy
import json
import re
import h5py

import random

def normname(n):
    return re.sub("[^a-zA-Z0-9]","_",n)

class Predictor:

    def build_model(self,dicts_filename,word_seq_len,word_vec,*args,**kwargs):
        """`word_vec is gensim's KeyedVectors`"""
        
        self.dicts_filename=dicts_filename
        with open(self.dicts_filename,"rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)


    def load_model(self,model_name):
        with open(model_name+".model.json", "rt") as f:
            self.model=model_from_json(f.read())
            self.model.load_weights(model_name+".weights.h5")
        with open(model_name+".dicts.json","rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)
        
    def save_model(self,file_name):
        model_json = self.model.to_json()
        with open(file_name+".model.json","w") as f:
            print(model_json,file=f)
        with open(file_name+".dicts.json","w") as f:
            json.dump((self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict),f)
    
    def word_seq_len(self):
        return self.model.get_layer("inp_char_seq").get_config()["batch_input_shape"][1]

    def word_emb_dim(self):
        l=self.model.get_layer("emb_word")
        word_emb_length=l.get_config()["input_dim"]
        word_emb_dim=l.get_config()["output_dim"]
        return word_emb_length,word_emb_dim
        


class WEmbDepPredictor(Predictor): #+Word embeddings +sequence of L/R dependents 

    def build_model(self,dicts_filename,word_seq_len,word_vec,**kwargs):
        super().build_model(dicts_filename,word_seq_len,word_vec,**kwargs)

        char_emb_dim=100
        pos_emb_dim=100
        deprel_emb_dim=100
        rnn_dim=500


        
        lr=kwargs.get("lr",0.001)
        dr=kwargs.get("dr",0.0)
        kern_l2=L2Reg(kwargs.get("kern_l2",0.0))
        act_l2=L2Reg(kwargs.get("act_l2",0.0))
        
        #vectorized train is list of words
        #each word is (input,output)
        #input is [[...char sequence...], pos, deprel]
        #output is [ classnum, classnum, classnum ] with as many classes as there are features
        
        inp_chars=Input(name="inp_char_seq",shape=(word_seq_len,)) #this is a sequence
        inp_wrd=Input(name="inp_word",shape=(1,))
        inp_left_deps=Input(name="inp_left_deps",shape=(5,)) #
        inp_right_deps=Input(name="inp_right_deps",shape=(5,)) #
        inp_pos=Input(name="inp_pos",shape=(1,)) #one POS
        inp_deprel=Input(name="inp_deprel",shape=(1,)) #one DEPREL

        word_emb=Flatten()(Embedding(word_vec.vectors.shape[0],word_vec.vectors.shape[1],name="emb_word",trainable=False,mask_zero=False,weights=[word_vec.vectors])(inp_wrd))
        chars_emb=Embedding(len(self.char_dict),char_emb_dim,mask_zero=False,embeddings_initializer=Constant(value=0.01))(inp_chars)
        lr_deps_emb=Embedding(len(self.deprel_dict),deprel_emb_dim,mask_zero=False,embeddings_initializer=Constant(value=0.01))
        left_deps_emb=lr_deps_emb(inp_left_deps)
        right_deps_emb=lr_deps_emb(inp_right_deps)
        pos_emb=Flatten()(Embedding(len(self.pos_dict),pos_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_pos))
        drel_emb=Flatten()(Embedding(len(self.deprel_dict),deprel_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_deprel))

        
        rnn_out_seq=Bidirectional(CuDNNLSTM(rnn_dim,kernel_regularizer=kern_l2,activity_regularizer=act_l2,return_sequences=True))(Dropout(rate=dr)(chars_emb))
        ldeps_rnn_out_seq=Bidirectional(CuDNNLSTM(rnn_dim,kernel_regularizer=kern_l2,return_sequences=True))(Dropout(rate=dr)(left_deps_emb))
        rdeps_rnn_out_seq=Bidirectional(CuDNNLSTM(rnn_dim,kernel_regularizer=kern_l2,return_sequences=True))(Dropout(rate=dr)(right_deps_emb))

        rnn_out=GlobalMaxPooling1D()(rnn_out_seq)
        ldeps_rnn_out=GlobalMaxPooling1D()(ldeps_rnn_out_seq)
        rdeps_rnn_out=GlobalMaxPooling1D()(rdeps_rnn_out_seq)

        cc=Concatenate()([word_emb,rnn_out,ldeps_rnn_out,rdeps_rnn_out,pos_emb,drel_emb])
        hidden=Dense(500,activation="tanh")(cc)
        outputs=[]
        for feat_name in sorted(self.feat_val_dict.keys()):
            outputs.append(Dense(len(self.feat_val_dict[feat_name]),name="out_"+normname(feat_name),activation="softmax")(hidden))
        
        self.model=Model(inputs=[inp_wrd,inp_chars,inp_left_deps,inp_right_deps,inp_pos,inp_deprel], outputs=outputs)
        self.optimizer=Adam(lr,amsgrad=True)
        self.model.compile(optimizer=self.optimizer,loss="sparse_categorical_crossentropy")

        
def acc(out,out_gold):
    out_pred=numpy.vstack([numpy.argmax(p,axis=-1) for p in out]).T  #examples by output
    out_gold=numpy.vstack(out_gold).T                               #examples by output
    matching=numpy.where(out_pred==out_gold,1,0)  #examples by output
    row_match_count=numpy.sum(matching,axis=1)   #examples, count of matching outputs
    correct=len(numpy.where(row_match_count>=24)[0])
    total=matching.shape[0]
    print("ACC=",correct/total,"(",correct,"/",total,")")

            

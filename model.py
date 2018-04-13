# import tensorflow as tf
# ### Only needed for me, not to block the whole GPU, you don't need this stuff
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
# ### ---end of weird stuff


import sys
import argparse
import data

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed, Flatten
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape
from keras.optimizers import SGD, Adam
from keras.initializers import Constant
#from keras.layers import CuDNNLSTM as LSTM  #massive speedup on graphics cards
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import numpy
import json
import re
import h5py

def normname(n):
    return re.sub("[^a-zA-Z0-9]","_",n)

class Predictor:

    def build_model(self,dicts_filename,word_seq_len):

        char_emb_dim=100
        pos_emb_dim=100
        deprel_emb_dim=100
        rnn_dim=200
        
        self.dicts_filename=dicts_filename
        with open(self.dicts_filename,"rt") as f:
            self.char_dict,self.pos_dict,self.deprel_dict,self.feat_val_dict=json.load(f)
        #vectorized train is list of words
        #each word is (input,output)
        #input is [[...char sequence...], pos, deprel]
        #output is [ classnum, classnum, classnum ] with as many classes as there are features
        
        inp_chars=Input(shape=(word_seq_len,)) #this is a sequence
        inp_pos=Input(shape=(1,)) #one POS
        inp_deprel=Input(shape=(1,)) #one DEPREL

        chars_emb=Embedding(len(self.char_dict),char_emb_dim,mask_zero=True,embeddings_initializer=Constant(value=0.01))(inp_chars)
        pos_emb=Flatten()(Embedding(len(self.pos_dict),pos_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_pos))
        drel_emb=Flatten()(Embedding(len(self.deprel_dict),deprel_emb_dim,embeddings_initializer=Constant(value=0.01))(inp_deprel))

        rnn_out=Bidirectional(LSTM(rnn_dim))(chars_emb)

        cc=Concatenate()([rnn_out,pos_emb,drel_emb])
        hidden=Dense(201,activation="tanh")(cc)
        outputs=[]
        for feat_name in sorted(self.feat_val_dict.keys()):
            outputs.append(Dense(len(self.feat_val_dict[feat_name]),name=normname(feat_name+"_out"),activation="softmax")(hidden))
        
        self.model=Model(inputs=[inp_chars,inp_pos,inp_deprel], outputs=outputs)
        self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")

    def save_model(self,file_name):
        model_json = self.model.to_json()
        with open(file_name+".model.json", "w") as f:
            print(model_json,file=f)
        

if __name__=="__main__":
    data_train=data.vectorize_data(sys.stdin,"dicts_fi.json")
    import random
    random.shuffle(data_train)
    #data_train=data_train[:3000]
    inputs=[item[0] for item in data_train]
    outputs=[item[1] for item in data_train]

    inputs=numpy.array(inputs)
    inputs_chars,inputs_pos,inputs_deprel=pad_sequences(inputs[:,0],padding="post"),inputs[:,1],inputs[:,2]

    outputs=numpy.array(outputs)
    print("Inp shape",inputs_chars.shape)
    print("Out shape",outputs.shape)
    m=Predictor()
    _,word_seq_len=inputs_chars.shape
    m.build_model("dicts_fi.json",word_seq_len)
    m.save_model("first_try")
    save_cb=ModelCheckpoint(filepath="first_try.weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    hist=m.model.fit([inputs_chars,inputs_pos,inputs_deprel],[outputs[:,i] for i in range(outputs.shape[1])],verbose=1,batch_size=200,epochs=15,validation_split=0.1,callbacks=[save_cb])
    with open("first_try.history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)
        
    

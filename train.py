import model as model
import data
import random
import numpy
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-file', help='.conllu')
    parser.add_argument('--devel-file', help='.conllu')
    parser.add_argument('--dicts-file', help='.json')
    parser.add_argument('--model-file', help='file-name-prefix to save to')
    args = parser.parse_args()

    with open(args.train_file) as f:
        train_conllu=data.read_conll(f)
        inputs_train_dict,outputs_train_dict,output_features=data.prep_data(train_conllu,args.dicts_file,word_seq_len=None,shuffle=True)
    word_seq_len=inputs_train_dict["inp_char_seq"].shape[1]
    with open(args.devel_file) as f:
        devel_conllu=data.read_conll(f)
        inputs_devel_dict,outputs_devel_dict,output_features_dev=data.prep_data(devel_conllu,args.dicts_file,word_seq_len=word_seq_len,shuffle=False)
        assert output_features==output_features_dev

    m=model.Predictor()
    m.build_model(args.dicts_file,word_seq_len)

    model_name=args.model_file
    m.save_model(model_name)
    save_cb=ModelCheckpoint(filepath=model_name+".weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb=EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')
    hist=m.model.fit(x=inputs_train_dict, y=outputs_train_dict, validation_data=(inputs_devel_dict,outputs_devel_dict), verbose=1, batch_size=200, epochs=100, callbacks=[save_cb,es_cb])
    with open(model_name+".history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)


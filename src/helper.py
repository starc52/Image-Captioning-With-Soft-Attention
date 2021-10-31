import numpy as np
import random
import os.path
import pickle
from PIL import Image
from os import path
from time import time
import string

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import glorot_uniform

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import nltk


model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

def testTrainSplit(ratio):# ratio is a list in the format `[test, validation, train]`
    testNum = int((30000./np.sum(np.array(ratio)))*ratio[0])
    validNum = int((30000./np.sum(np.array(ratio)))*ratio[1])
    # data_location = '/ssd_scratch/cvit/starc52/Flickr-30K/'
    data_location = '../../data/'
    filename = data_location+'flickr30k_images/results.csv'
    f = open(filename,'r')
    doc = f.read()
    total=len(doc.split('\n'))
    print(total)
    trainValidNum=total-1-testNum
    descriptions = dict()
    lines=doc.split('\n')
    for line in lines:
        if line=='':
            continue
        # print("hahaha line", line)
        tokens = line.split('|')
        # print("hahaha tokens", tokens)
        img_name = tokens[0].split('.')[0]
        if tokens == ['']:
            break
        if img_name not in descriptions:
            descriptions[img_name] = []
        if len(tokens)==2:
            print(tokens)
        descriptions[img_name].append(tokens[2])
    images=list(descriptions.keys())
    print(len(images))
    print("Ye hamari images print ho rahi hain\n",images)
    images.remove("image_name")
    f.close()
    testImages=random.sample(images, testNum)
    test_filename=data_location+'flickr30k_images/test_results.csv'
    test_f = open(test_filename, 'w')
    for i in testImages:
        images.remove(i)
        temp_str=""
        for j in range(5):
            temp_str = temp_str+i+".jpg"+"|"+" "+str(j)+"|"+descriptions[i][j]+"\n"
        test_f.write(temp_str)
    test_f.close()
    validImages=random.sample(images, validNum)
    valid_filename=data_location+'flickr30k_images/valid_results.csv'
    valid_f = open(valid_filename, 'w')
    for i in validImages:
        images.remove(i)
        temp_str=""
        for j in range(5):
            temp_str = temp_str+i+".jpg"+"|"+" "+str(j)+"|"+descriptions[i][j]+"\n"
        valid_f.write(temp_str)
    valid_f.close()
    train_filename=data_location+'flickr30k_images/train_results.csv'
    train_f = open(train_filename, 'w')
    for i in images:
        temp_str=""
        for j in range(5):
            temp_str =temp_str+i+".jpg"+"|"+" "+str(j)+"|"+descriptions[i][j]+"\n"
        train_f.write(temp_str)
    train_f.close()
    return descriptions

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def makeDescDict(filename):
    f = open(filename,'r')
    doc = f.read()
    ## Constructing dictionary with each image name as key and a list having corresponding 5 captions
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split('|')
        img_name = tokens[0].split('.')[0]

        if tokens == ['']:
            break

        if img_name not in descriptions:
            descriptions[img_name] = []
        descriptions[img_name].append(tokens[2])

    dict_items=descriptions.items()
    # dict_items=list(dict_items)
    # print(dict_items)
    f.close()
    return descriptions

def dataPreprocess(pathOfFile):#pathOfFile is in the format "flickr30k_images/*.csv"
    ## Reading the file
    # data_location = '/ssd_scratch/cvit/52/Flickr-30K/'
    data_location = '../../data/'
    filename = data_location+pathOfFile
    print("POF",pathOfFile)
    nameOfFile = pathOfFile.split("/")[1].split(".")[0]

    descriptions = makeDescDict(filename)
    ## Pre-processing the dictionary

    table = str.maketrans('', '', string.punctuation)

    for key, items in descriptions.items():
        for i in range(len(items)):

            sentence = items[i]
        
            sentence = sentence.split(' ')
            sentence = [word.lower() for word in sentence]
            sentence = [word.translate(table) for word in sentence]
            sentence = [word for word in sentence if len(word)>1]
            sentence = [word for word in sentence if word.isalpha()]

            items[i] = ' '.join(sentence)

        
    ## Building vocabulary set
    truthVal=path.isfile(data_location+"flickr30k_images/vocab_"+nameOfFile+".csv")
    freqVocab=[]
    if truthVal == False:
        vocab = set()

        for keys, items in descriptions.items():
            for i in range(len(items)):
                sentence = items[i]

                sentence = sentence.split()
                for word in sentence:
                    vocab.add(word)

        print(len(vocab))
        ## Steps to do:
        ## Creating dataset of 10000 most frequent words. 
        all_sentences = []
        count = {}
        threshold = 10
        freqVocab=[]
        for keys, items in descriptions.items():
            for i in range(len(items)):
                sentence = items[i]
                all_sentences.append(sentence)
        for i in range(len(all_sentences)):
            sentence = all_sentences[i]
            words = sentence.split(' ')
            for word in words:
                count[word] = count.get(word, 0)+1
        for word in count:
            if count[word]>threshold:
                freqVocab.append(word)
        vocab_f = open(data_location+"flickr30k_images/vocab_"+nameOfFile+".csv", 'w')
        temp_str=""
        for i in freqVocab:
            temp_str=temp_str+i+","
        temp_str=temp_str[:-1]
        vocab_f.write(temp_str)
        vocab_f.close()
    else:
        vocab_f = open(data_location+"flickr30k_images/vocab_"+nameOfFile+".csv", 'r')
        vocab_str = vocab_f.read()
        freqVocab = vocab_str.split(",")
    return freqVocab

def to_lines(descriptions):
    all_desc = list()
    print("descriptions",type(descriptions))
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
def max_length(descriptions):
    lines = to_lines(descriptions)
    max=-1
    for d in lines:
        comma_sep=d.split(",")
        for j in comma_sep:
            if(len(j.split())>max):
                max=len(j.split())
    return max

def getoutput(image, wordtoidx):
    sentence = 'startseq'
    for i in range(max_caption_length):
        words = sentence.split()
        idxs = []
        for word in words:
            if word in wordtoidx:
                idx = wordtoidx[word]
                idxs.append(idx)
        idxs = pad_sequences([idxs], maxlen=max_caption_length)
        y = model.predict([image, idxs],verbose=0)
        y = np.argmax(y)
        y_word = idxtoword[y]
        sentence = sentence + " " + y_word
        if y_word == 'endseq':
            break
    return sentence

def getidxarrs(freqVocab):
    idxtoword = {}
    wordtoidx = {}
    idx = 1
    for w in freqVocab:
        wordtoidx[w] = idx
        idxtoword[idx] = w
        idx += 1
    return idx,wordtoidx,idxtoword

tempo={}
# data generator, intended to be used in a call to model.fit_generator()
# def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
W=0
def data_generator(descriptions, wordtoix, max_length, num_photos_per_batch):
    print("Inside Data generator")
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            global W
            W+=1
            global tempo
            # retrieve the photo feature
            # photo = photos[key+'.jpg']
            # if str("/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/"+key+'.jpg') in tempo.keys():
            if str("../../data/flickr30k_images/flickr30k_images/"+key+'.jpg') in tempo.keys():
                # photo=tempo[str("/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/"+key+'.jpg')]
                photo=tempo[str("../../data/flickr30k_images/flickr30k_images/"+key+'.jpg')]
            else:
                # photo = encode("/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/"+key+'.jpg')
                photo = encode("../../data/flickr30k_images/flickr30k_images/"+key+'.jpg')
                # tempo[str("/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/"+key+'.jpg')] = photo
                tempo[str("/../../data/flickr30k_images//flickr30k_images/"+key+'.jpg')] = photo
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                print("Yield ke pehle, W=",W)
                yield ((np.array(X1), np.array(X2)), np.array(y))
                X1, X2, y = list(), list(), list()
                n=0


temp=dataPreprocess("flickr30k_images/train_results.csv")
print(len(temp))

temp2=dataPreprocess("flickr30k_images/valid_results.csv")
freqVocab = list(set(temp).union(set(temp2)))
print(len(freqVocab))
embedding_dim = 200
embeddings_idx={}

glove_locn="../../data/glove_vectors/glove.6B.200d.txt"
f=open(glove_locn,encoding='utf-8')
for line in f:
    word_values=line.split()
    word=word_values[0]
    coefs = np.asarray(word_values[1:], dtype='float32')
    embeddings_idx[word] = coefs
f.close()

idx,wordtoidx,idxtoword=getidxarrs(freqVocab)
vocab_size = len(idxtoword) + 1 # one for appended 0's
print("Vocab Size:",vocab_size)
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoidx.items():
    embedding_vector = embeddings_idx.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding idx will be all zeros
        embedding_matrix[i] = embedding_vector
print("embedding matrix shape=",embedding_matrix.shape)
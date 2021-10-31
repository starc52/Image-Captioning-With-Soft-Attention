import numpy as np
import random
import os.path
import pickle
from PIL import Image
from os import path
from time import time

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
from keras.models import load_model
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model    
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def dataPreprocess(pathOfFile):#pathOfFile is in the format "flickr30k_images/*.csv"
    ## Reading the file
    data_location = '../../data/'
    filename = data_location+pathOfFile
    nameOfFile = pathOfFile.split("/")[1].split(".")[0]
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
    dict_items=list(dict_items)
    # print(dict_items)

    f.close()
    ## Pre-processing the dictionary

    # table = str.maketrans('', '', string.punctuation)

    for key, items in descriptions.items():
        for i in range(len(items)):

            sentence = items[i]
        
            sentence = sentence.split(' ')
            sentence = [word.lower() for word in sentence]
            sentence = [word for word in sentence if word.isalpha()]
    #        sentence = [word.translate(table) for word in sentence]

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
        # frequency=[]
        # for j in tqdm(vocab):
        #     count=0
        #     for keys, items in descriptions.items():
        #         for i in range(len(items)):
        #             sentence = items[i]
 
        #             sentence = sentence.split()
        #             for word in sentence:
        #                 if word == j:
        #                     count+=1
        #     temp = [j, count]
        #     frequency.append(temp)
        # frequency.sort(key = lambda x: x[1])
        # freqVocab=[]
        # if len(frequency)>10000:
        #     frequency=frequency[0:10000]
        #     for i in frequency:
        #         freqVocab.append(i[0])
        # else:
        #     for i in frequency:
        #         freqVocab.append(i[0])

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

embedding_dim = 200
temp=dataPreprocess("flickr30k_images/train_results.csv")
print(len(temp))
temp2=dataPreprocess("flickr30k_images/valid_results.csv")
freqVocab = list(set(temp).union(set(temp2)))
print(len(freqVocab))

print("preprocessing Done")
idxtoword = {}
wordtoidx = {}
idx = 1

for w in freqVocab:
    wordtoidx[w] = idx
    idxtoword[idx] = w
    idx += 1
print("wordtoidx Done")

max_caption_length=73
vocab_size = len(idxtoword) + 1 # one for appended 0's


# inputs1 = Input(shape=(2048,))
# fe1 = Dropout(0.5)(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)
# inputs2 = Input(shape=(max_caption_length,))
# se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
# se2 = Dropout(0.5)(se1)
# se3 = LSTM(256)(se2)
# decoder1 = add([fe2, se3])
# decoder2 = Dense(256, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)
# model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model = load_model('../../data/models/model_4.h5')

## Test set de diya



# testTrainSplit([1, 1, 8])

def getoutput(image, wordtoidx):
    sentence = 'startseq'
    for i in range(max_caption_length):
        words = sentence.split()
        idxs = []
        print("Inside getoutput")
        # print(wordtoidx.items())
        # print(wordtoidx)
        for word in words:
            print("word",word)
            if word in wordtoidx:
                # print("####################################\n Hello\n#################################")
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

test_filename = "../../data/flickr30k_images/test_results.csv"

test_set_dash = open(test_filename, 'r')
test_set = test_set_dash.read()

print("before final")
count=0

lines = test_set.split("\n")
line = lines[5]

print("Count = ", count)
words = line.split('|')
img_name = words[0]
print(img_name)
img_path = '../../data/flickr30k_images/flickr30k_images/' + img_name
encoded_img = encode(img_path)
img = encoded_img.reshape(1,2048)
output = getoutput(img, wordtoidx)
print(output)
img_vals=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
plt.imshow(img_vals)
plt.show()
count+=1

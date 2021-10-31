from helper import *

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

embedding_dim = 200
temp=dataPreprocess("flickr30k_images/train_results.csv")
print(len(temp))
temp2=dataPreprocess("flickr30k_images/valid_results.csv")
freqVocab = list(set(temp).union(set(temp2)))
print(len(freqVocab))
print("preprocessing Done")

idx,wordtoidx,idxtoword=getidxarrs(freqVocab)
print("wordtoidx Done")

max_caption_length=73
vocab_size = len(idxtoword) + 1 # one for appended 0's

model = load_model('/home/starc52/models/model_19.h5')

test_filename = "/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/test_results.csv"

test_set_dash = open(test_filename, 'r')
test_set = test_set_dash.read()

print("before final")
count=0
data_location = '/ssd_scratch/cvit/starc52/Flickr-30K/'
filename = data_location+"flickr30k_images/test_results.csv"
f = open(filename,'r')
doc = f.read()
## Constructing dictionary with each image name as key and a list having corresponding 5 captions
descriptions = dict()
for line in doc.split('\n'):
        tokens = line.split('|')
        img_name = tokens[0].split('.')[0]
        #print(img_name)
        if tokens == ['']:
            break
        if img_name not in descriptions:
            descriptions[img_name] = []
        descriptions[img_name].append(tokens[2])
for line in test_set.split('\n'):
	if count == 0:
		count+=1
		continue
	count+=1
	print("Count = ", count)
	words = line.split('|')
	img_name = words[0]
	print(img_name)
	img_path = '/ssd_scratch/cvit/starc52/Flickr-30K/flickr30k_images/' + img_name
	encoded_img = encode(img_path)
	img = encoded_img.reshape(1,2048)
	output = getoutput(img, wordtoidx)
	print(output)
	print(descriptions[img_name.split('.')[0]])
	references=[list(filter(lambda x:x != '' and x!='.' and x!=',', descriptions[img_name.split('.')[0]][i].split(" "))) for i in range(5)]
	hypothesis = output.split(" ")
	print(references, hypothesis)
	BLEUscore = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
	print(BLEUscore)
	img_vals=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
	img_vals=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
	plt.imshow(img_vals)
	plt.show()
	if count>1:
	    break

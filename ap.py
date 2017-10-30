import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

def cleantext(text):
    return re.sub('[!|@|#|$|%|^|&|*|(|)|{|}|;|:|[|,|.|/|<|>|?|\|||`|~|-|=|_|+|\-\-|\'\']'," ",text)

rootdir='/home/shail/Desktop/ML/HW2/data/20_newsgroups'
j=0
dict={}

#### TRAIN DATA ####

for subdir, dirs, files in os.walk(rootdir):
    if files:
        j+=1
        dict[j]={}
        i = 0
        for file in files:
            i+=1
            if i>500:
                break
            else:
                filepath = subdir + os.sep + file
                if filepath.endswith(""):
                    f = open(filepath, 'rb')
                    example_sent = f.read()
                    removeSpecialChars = cleantext(example_sent)
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(removeSpecialChars)
                    for w in word_tokens:
                        #print w
                        if w not in stop_words:
                            if w in dict[j]:
                                dict[j][w]=dict[j][w]+1
                            else:
                                dict[j][w]=1



                                
###### TEST DATA ######
for subdir, dirs, files in os.walk(rootdir):
    if files:
        j+=1
        dict[j]={}
        i = 0
        for file in files:
            i+=1
            if i<501:
                break
            else:
                filepath = subdir + os.sep + file
                if filepath.endswith(""):
                    f = open(filepath, 'rb')
                    example_sent = f.read()
                    removeSpecialChars = cleantext(example_sent)
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(removeSpecialChars)
                    for w in word_tokens:
                        #print w
                        if w not in stop_words:
                            if w in dict[j]:
                                dict[j][w]=dict[j][w]+1
                            else:
                                dict[j][w]=1

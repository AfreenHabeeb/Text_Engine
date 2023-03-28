
import re, os, shutil
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd

'''Code Explained:
    1.remove empty chats
    2.three things happening here:
        a) tokenize chat into individual words
        b) POS tagging on each word
        c) if sentiment score <= 0.25 then it is a complaint else a query
    3.retain only the words tagged as nouns and adjectives
    4.keep lists with 2 or more words'''


##POS tagging
def returnNnAdj(line):
    nounAdjList = []
    for x in line:
        if x[1] in ['NN', 'NNP', 'NNS', 'NNPS', 'ADJ', 'VBP', 'VB', 'ADV', 'VBD', 'VBG', 'VBN', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            nounAdjList.append(x[0])

    return nounAdjList

def returnNoun(line):
    nounList = []
    for x in line:
        if x[1] in ['NN', 'NNP', 'NNS', 'NNPS']:
            nounList.append(x[0])

    return nounList

##Tagging sentence as Complaint or Query
def sentLabel(x):
    if x <= 0.25:  # any thing less than equals to 0 is 'Complaint' (negative) and above it is a 'Query' (query)
        return 'complaint'
    else:
        return 'query'

## Function to remove unicode from tokens and make them stringtype
def remUnicode(line):
    tokensList = []
    for i in line:
        i = i.encode('utf8')
        tokensList.append(i)
    return tokensList

##Removing Junk words
def removeBoguswords(tokens, bogusWords):
    # these are responses by customers which are concatenated with the last or first word of another sentence

    #stopEnglish = stopwords.words('english')
    #tokens = [upperCaseWord.lower() for upperCaseWord in tokens if upperCaseWord.lower() not in stopEnglish]
    tokens = [upperCaseWord.lower() for upperCaseWord in tokens]
    familyList = ['son', 'daughter', 'husband', 'wife', 'father', 'mother', 'brother', 'sister']
    tokens1 = []
    for word in tokens:
        if word in familyList:
            word = 'family'
        tokens1.append(word)
    tokens = tokens1
    #tokens =[word for word in tokens if word not in familyList]
    tokens2 = [word for word in tokens if word not in bogusWords]

    return tokens2


def lemmatization(line):
    lmtzr = WordNetLemmatizer()
    lemme_list = [lmtzr.lemmatize(x) for x in line]

    return lemme_list

def removeunilengthtokens(tokens):
    tokens2 = [i for i in tokens if len(i) > 1]

    return tokens2

def replaceText(line, dic):
    for i, j in dic.items():
        line = line.replace(i,j)
    return line

#def getString(tokens):
#    return " ".join(tokens)

def fnCombWord(line):
    combSentence = ''
 
    for word in line:
        combSentence += ' '+word
    return combSentence.strip()


def fnPOSTagger(cc_df, repoDirectoryPath, folderToWrite):   

    path1 = repoDirectoryPath + '/bogusWordsList.txt'
    f = open(path1, "r")
    bogusWords = [pattern.strip() for pattern in f.readlines()]
    #Reading specialPrograms.txt file
    path2 = repoDirectoryPath + '/specialPrograms.txt'
    g = open(path2, "r")
    specialPrograms = {}
    for pattern in g.readlines():
        pattern = pattern.strip()
        keyvalue = pattern.split(':')
        specialPrograms[keyvalue[0]] = keyvalue[1]

    
  
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(word_tokenize)
    
 
    filter1=cc_df[cc_df.columns[1]].apply(lambda x: True if len(x)>2 else False)
    cc_df=cc_df[filter1]
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(fnCombWord)
    
    
    
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(lambda x: re.sub('[^a-zA-Z0-9]+', ' ', x.replace('*', "").replace('-', "")).strip())
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(word_tokenize)
    
    cc_df.loc[:,10]=cc_df[cc_df.columns[2]].apply(sentLabel)
    
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(lemmatization)
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(lambda x: removeBoguswords(x,bogusWords))
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(removeunilengthtokens)
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(pos_tag)
    
    #pos_tags_NNADJ #5
    cc_df.loc[:,20]=cc_df[cc_df.columns[1]].apply(returnNnAdj)
    
    #tokens #6
    cc_df.loc[:,30]=cc_df[cc_df.columns[1]].apply(returnNoun)
    
    """cc_df[cc_df.columns[0]]=cc_df[cc_df.columns[0]].astype(str).apply(lambda x: x.encode('utf8'))
   
    cc_df[cc_df.columns[5]]=cc_df[cc_df.columns[5]].apply(remUnicode)"""
    

    filter2=cc_df[cc_df.columns[5]].apply(lambda x: True if len(x)>0 else False)
    #cc_df_topic=pd.DataFrame()
    cc_df_topic=cc_df[filter2].copy()
    ###########
    cc_df_topic=cc_df_topic[cc_df_topic.columns[[0,6]]]
    cc_df_topic=cc_df_topic.rename(columns={cc_df_topic.columns[0]:'IDNumber', cc_df_topic.columns[1]:'Nn'})

    
    cc_df=cc_df[cc_df.columns[[0,5,4,2,3]]]
    cc_df=cc_df.rename(columns={ cc_df.columns[0]:'IDNumber', cc_df.columns[1]:'pos_tags_NNADJ', cc_df.columns[2]:'label', cc_df.columns[3]:'score', cc_df.columns[4]:'NERtokens'})
    cc_df['pos_tags_NNADJ'] = cc_df['pos_tags_NNADJ'].apply(fnCombWord)
    
    
    """REMOVE"""
    pathToSave=folderToWrite+'POS_Tagger/'
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
    cc_df.to_csv(pathToSave+'cc_df.csv',header=True, index=False)
    cc_df_topic.to_csv(pathToSave+'cc_df_topic.csv',header=True, index=False)

    return cc_df, cc_df_topic
    









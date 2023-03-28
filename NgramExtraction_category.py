import numpy as np
from datetime import datetime
import os, shutil, string, re, json, sys
from collections import Counter
from nltk.util import ngrams
import pandas as pd



def remMiddleSpace(x):
    strToken = x.split(' ')
    tokLen = len(strToken)
    if tokLen > 1:
        return strToken[0] + strToken[1]  # a short cut
    else:
        return x

def remNone(val):
    return str(val) or ''

def getWordCount(line):
    wordCountList = []
    wordCounts = Counter(line)
    for item in wordCounts.items():
        text = str(item[0] + ":" + str(item[1]))
        wordCountList.append(text)
    wordCountText = ",".join(wordCountList)
    return wordCountText


'''def get_ngrams(text, min=1, max=4):
    s = []
    text1 = []

    for j in text:
        if j not in text1:
            text1.append(j)

    for n in range(min, max):
        if n == 1:
            for ngram in ngrams(text, n):
                s.append('_'.join(str(i) for i in ngram))
        else:
            for ngram in ngrams(text1, n):
                s.append('_'.join(str(i) for i in ngram))
    return s'''

def get_ngrams(text, min=1, max=4):
    text=text.split(' ')
    s = []

    for n in range(min, max):
        if n == 1:
            for ngram in ngrams(text, n):
                s.append('_'.join(str(i) for i in ngram))
        else:
            for ngram in ngrams(text, n):
                ngram1 = []
                for x in ngram:
                    if x not in ngram1:
                        ngram1.append(x)
                ngram1_t = tuple(ngram1)
                s.append('_'.join(str(i) for i in ngram1_t))
    return s


def appendChatID(id_no,gram_tokens):
    myList = gram_tokens.split(",")
    newList = []
    for items in myList:
        x = id_no + ',' + items
        newList.append(x)

    return newList

def getGrams(line):
    grams = line.split(":")[0]
    return grams

def getGramsCount(line):
    grams = line.split(":")[0]
    gramsCount = str.count(grams, '_') + 1
    return gramsCount

def getFrequency(line):
    grams = line.split(":")[1]
    return int(grams)

def remove_bogus_grams(tokens, bogusBigrams):

    for pattern in bogusBigrams:
        tokens = [word for word in tokens if not re.findall(pattern=pattern, string=word, flags=re.IGNORECASE)]

    return tokens



def getOnlyBiTriGrams(tokens, topicOnGrams):

    if topicOnGrams == 2:
        return ([token for token in tokens if token.count('_') == 1])
    elif topicOnGrams == 3:
        return ([token for token in tokens if token.count('_') == 2])




def fnExtractNgrams(folderToWrite, transformed_ce, outputDataFormat, repoDirectoryPath, topicOnGrams, runNgram, prodRun):

    path = repoDirectoryPath + '/bogusBigrams.txt'
    f = open(path, "r")
    bogusBigrams = [pattern.strip() for pattern in f.readlines()]

    
    #initialize cc_df_topic
    cc_df_topic=pd.DataFrame()
    
    cc_df_topic=transformed_ce[transformed_ce.columns[[0,1]]].copy().astype(str).reset_index(drop=True)

    #cc_df_topic=transformed_ce[transformed_ce.columns[[0,3]]].copy().astype(str)

    cc_df_topic[cc_df_topic.columns[1]]=cc_df_topic[cc_df_topic.columns[1]].apply(remNone)
    cc_df_topic[cc_df_topic.columns[1]]=cc_df_topic[cc_df_topic.columns[1]].apply(get_ngrams)
    cc_df_topic[cc_df_topic.columns[1]]=cc_df_topic[cc_df_topic.columns[1]].apply(lambda x: remove_bogus_grams(x,bogusBigrams))
    

    #iniialize wordsCount
    wordCounts=pd.DataFrame()
    
    #get IDNumber and generate GramToken counts from cc_df_topic
    wordCounts=cc_df_topic[cc_df_topic.columns[[0,1]]].copy()
    wordCounts.iloc[:,1]=wordCounts[wordCounts.columns[1]].apply( getWordCount)
    

    id_token_list=[]
    for i in range(0,len(wordCounts)):
        id_token_list.append(appendChatID(wordCounts.iloc[i,0], wordCounts.iloc[i,1]))

    #flatten list
    id_token_list = [y for x in id_token_list for y in x]
    

    #create a third column 
    for i in range(0,len(id_token_list)):
        wordCounts.loc[i,10]=id_token_list[i]
        

    #Now the first column has IDNumber
    wordCounts[wordCounts.columns[0]]=wordCounts[wordCounts.columns[2]].apply(lambda x: x.split(',')[0])
    
    #The third column has tokens and their counts
    wordCounts[wordCounts.columns[2]]=wordCounts[wordCounts.columns[2]].apply(lambda x: x.split(',')[1])
    
    #Create new columns for GramTokens, NgramNum and Frequency
    wordCounts[wordCounts.columns[1]]= wordCounts[wordCounts.columns[2]].apply( getGrams)
    wordCounts.loc[:,20]= wordCounts[wordCounts.columns[2]].apply( getGramsCount)
    wordCounts.loc[:,30]= wordCounts[wordCounts.columns[2]].apply( getFrequency)
    
    #drop columns 3 as all the info has been taken and it is not needed anymore
    wordCounts.drop(wordCounts.columns[2],axis=1, inplace=True)
    
    #set schema
    wordCounts.rename(columns={wordCounts.columns[0]: "IDNumber", wordCounts.columns[1]: "GramToken",
                           wordCounts.columns[2]: 'NgramNum', wordCounts.columns[3]: "Frequency" }, inplace=True)

    wordCounts=wordCounts.astype({'IDNumber':'str','GramToken':'str','NgramNum':'int','Frequency':'int'})

    # Outputs list of ChatID, gram, Ngram#, Frequency
    OP1_df = wordCounts.copy()

    if runNgram.lower() == 'y':
        if outputDataFormat.lower() == 'file':
            #Write Ngrams file
            try:
                
              
                pathToSaveGrams = folderToWrite + 'Ngrams_Extraction/'
                if not os.path.exists(pathToSaveGrams):
                    os.makedirs(pathToSaveGrams)


                #os.chdir(folderToWrite)  # change directory
                pathToSaveGrams = pathToSaveGrams + 'ng.csv'
                OP1_df.to_csv(pathToSaveGrams, header=True, index=False)

                
                #OP1_df.to_csv('ng.csv', header=True, index=False)
            except Exception as error:
                print ('Error writing the NGrams files')
                print(error)

        else:
            print ('Output Data Format should be file')
            sys.exit()
            
    
    
    cc_df_topic[cc_df_topic.columns[1]]=cc_df_topic[cc_df_topic.columns[1]].apply(lambda x: getOnlyBiTriGrams(x,topicOnGrams))

    #cc_df_topic=cc_df_topic.astype({'IDNumber':'str','tokens':'str'})

    return cc_df_topic

    
    
    
    
    
    
    
    
    
    
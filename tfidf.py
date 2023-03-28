import sys, os

import numpy as np
import pandas as pd
"""
def fnGetWords(indexList):
    wordsList = []
    for i in indexList:
        wordsList.append(vocabDict[i])
    return wordsList
"""
def fnGoodWords(line,highInfoWords):
    wordList = []
    for word in line[1]:
        if word in highInfoWords:
            wordList.append(word)
    return wordList

def fnCombWord(line):

    combSentence=' '.join(line)
    return combSentence.strip()

def fnTfidf(cc_df, repoDirectoryPath, folderToWrite):

    #reading the specialPrograms.txt file

    path = repoDirectoryPath + '/specialPrograms.txt'
    g = open(path, "r")
    specialPrograms = {}
    for pattern in g.readlines():
        pattern = pattern.strip()
        keyvalue = pattern.split(':')
        specialPrograms[keyvalue[0]] = keyvalue[1]


    ## TFIDF
    # type(preTfIdfSchema) -- pyspark.sql.types.StructType
    '''preTfIdfSchema = StructType([StructField(name = 'IDNumber', dataType = StringType()),
                                StructField(name = 'tokens', dataType = ArrayType(elementType = StringType())),
                                StructField(name = 'label', dataType = StringType()),
                                StructField(name = 'score', dataType = StringType()),
                                StructField(name='NERtokens', dataType=StringType())])

    '''
    
    
    transformed_ce=cc_df[cc_df.columns[[0,1,2,3,4]]]

    
    #transformed_ce[transformed_ce.columns[1]]=transformed_ce[transformed_ce.columns[1]].apply(fnCombWord)
    transformed_ce.rename(columns={transformed_ce.columns[0]:'IDNumber', transformed_ce.columns[1]:'tokens', transformed_ce.columns[2]:'label',
                                   transformed_ce.columns[3]:'score', transformed_ce.columns[4]:'NERtokens' }, inplace=True)

    transformed_ce=transformed_ce.astype({'IDNumber':'str', 'tokens':'str', 'label':'str', 'score':'str', 'NERtokens':'str' })
    
    
    
    """REMOVE"""
    pathToSave=folderToWrite+'tfidf/'
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
    transformed_ce.to_csv(pathToSave+'transformed_ce.csv',header=True, index=False)


    # return transformed_ce
    return transformed_ce

















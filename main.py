#Setting up the configuration parameter file
import configparser
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
from dataCleaning import fnReadData
from sentimentalAnalysis import fnSentimentAnalyzer
from NgramExtraction_category import fnExtractNgrams
from posTagging import fnPOSTagger
from tfidf import fnTfidf
from topicModelling import fntopicModelling
import warnings
warnings.filterwarnings("ignore")

def function_call(path):
    #taking configuration file path from the terminal
    #config_path=sys.argv[1]
    config_path=path
    config = configparser.ConfigParser()
    config.readfp(open( config_path ))

    ### Reading configuration parameter file ###
    inputDataFormat = config.get('Config Parms', 'inputDataFormat') #hive/file
    outputDataFormat = config.get('Config Parms', 'outputDataFormat') #output data format
    outputFilesLocation = config.get('Config Parms', 'outputFilesLocation')
    repoDirectoryPath = config.get('Config Parms', 'repoDirectoryPath')
    runSentiment = config.get('Config Parms', 'runSentiment')
    runNgram = config.get('Config Parms', 'runNgram')
    topicOnGrams = config.get('Config Parms', 'topicOnGrams')
    optimizeTopics = config.get('Config Parms', 'optimizeTopics')
    u_numOfTopics = config.get('Config Parms', 'u_numOfTopics')
    prodRun = config.get('Config Parms', 'prodRun')
    requireTopics = config.get('Config Parms', 'requireTopics')
    inputFileNameWithPath = config.get('Config Parms', 'inputFileNameWithPath') #if file --> input file path
    inputDataSeparator = config.get('Config Parms', 'inputDataSeparator') #if file --> separator
    inputHeader = config.get('Config Parms', 'inputHeader')
    generateUniqueID = config.get('Config Parms', 'generateUniqueID')
    groupByID = config.get('Config Parms', 'groupByID')



    starttime = datetime.now()

    if inputHeader == 'False':
        inputHeader = False
    else:
        inputHeader = True

    if len(topicOnGrams) > 0:
        topicOnGrams = int(topicOnGrams) #converting topic on grams to integer

    if len(u_numOfTopics) > 0:
        u_numOfTopics = int(u_numOfTopics) #converting input number of topics to integer

    #Read data and clean data
    cc_df, cc_df1= fnReadData(inputDataFormat, inputDataSeparator, inputFileNameWithPath, inputHeader, generateUniqueID, groupByID)
    print('---> Done Data Cleaning !!!')

    endtime = datetime.now()
    timeElapsed = (endtime-starttime).total_seconds()
    print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')

    folderToWrite = None
    if outputDataFormat == 'file':
        #chdir(outputFilesLocation)  # change directory
        fileName = 'Text_Analytics_'
        folderToWrite = outputFilesLocation + '/'+ fileName + str(datetime.now().date())+'_'+ str(datetime.now().hour)+'_'+str(datetime.now().minute)
        #os.mkdir(folderToWrite)
        folderToWrite = folderToWrite + '/'

    ### Calling sentimentAnalysis.py
    cc_df, sentiDf = fnSentimentAnalyzer(cc_df, repoDirectoryPath, folderToWrite, cc_df1, outputDataFormat, runSentiment, prodRun)
    print('---> Done Sentiment Analysis !!!')


    endtime = datetime.now()
    timeElapsed = (endtime-starttime).total_seconds()
    print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')



    if runNgram.lower() == 'y' or requireTopics.lower() == 'y':
        cc_df, cc_df_topic = fnPOSTagger(cc_df, repoDirectoryPath, folderToWrite)
        print('---> Done POS Tagging !!!')
        #print(cc_df.count())
        endtime = datetime.now()
        timeElapsed = (endtime-starttime).total_seconds()
        print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')


        transformed_ce  = fnTfidf(cc_df, repoDirectoryPath, folderToWrite)
        print('---> Done TFIDF !!!')
        endtime = datetime.now()
        timeElapsed = (endtime-starttime).total_seconds()
        print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')



    if runNgram.lower() == 'y' or requireTopics.lower() == 'y':
        ## Calling NgramExtraction_category.py
        cc_df_topic_grams = fnExtractNgrams( folderToWrite, transformed_ce, outputDataFormat, repoDirectoryPath, topicOnGrams, runNgram, prodRun)
        endtime = datetime.now()
        timeElapsed = (endtime-starttime).total_seconds()
        print ('---> Sentiment Analysis and NGram Extraction Done !!!')
        print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')


    #topic modelling

    if requireTopics.lower() == 'y':
        ##Topic Modelling - Calling TF-IDF to get hign information words for topic modelling
        if topicOnGrams in [2, 3] :
            dfTopic = cc_df_topic_grams
        else :
            dfTopic = cc_df_topic


        topic_vis_data, lda_data=fntopicModelling( dfTopic, folderToWrite, u_numOfTopics, optimizeTopics)
        print('---> Done topic modelling !!!')
        endtime = datetime.now()
        timeElapsed = (endtime-starttime).total_seconds()
        print('Time Taken : ' + str(int(timeElapsed/60.0)) + ' mins')

        print("Please check the output")

        exit_code = '000'

    return exit_code


exit_code = function_call(path="C:/Users/HABEEAF/Desktop/Afreen/TextEngine_python_v2/TextEngine/conf/TE_configuration_params_v3.txt")





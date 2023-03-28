import configparser
import ast

def text_read_write(path,inputDataFormat, inputDBname, inputTablename, inputFileNameWithPath, inputDataSeparator,
                    inputHeader, generateUniqueID,groupByID, outputDataFormat, outputFilesLocation, repoDirectoryPath,
                    runSentiment, runNgram, requireTopics,optimizeTopics, topicOnGrams, u_numOfTopics,
                    outputDBname, outputSentimentScoreTable, outputNgramsTable, prodRun):
    config = configparser.ConfigParser()
    config.read(path)
    config_section=config.sections()
    if('Config Parms' in config_section):
        config['Config Parms']['inputDataFormat']=inputDataFormat
        config['Config Parms']['inputDBname']=inputDBname
        config['Config Parms']['inputTablename']=inputTablename
        config['Config Parms']['inputFileNameWithPath'] = inputFileNameWithPath
        config['Config Parms']['inputDataSeparator'] = inputDataSeparator
        config['Config Parms']['inputHeader']=inputHeader
        config['Config Parms']['generateUniqueID']=generateUniqueID
        config['Config Parms']['groupByID'] = groupByID
        config['Config Parms']['repoDirectoryPath'] = repoDirectoryPath
        config['Config Parms']['runSentiment']=runSentiment
        config['Config Parms']['runNgram']=runNgram
        config['Config Parms']['requireTopics'] = requireTopics
        config['Config Parms']['optimizeTopics'] = optimizeTopics
        config['Config Parms']['topicOnGrams'] = topicOnGrams
        config['Config Parms']['u_numOfTopics'] = u_numOfTopics
        config['Config Parms']['outputDataFormat'] = outputDataFormat
        config['Config Parms']['outputFilesLocation'] = outputFilesLocation
        config['Config Parms']['outputDBname']=outputDBname
        config['Config Parms']['outputSentimentScoreTable']=outputSentimentScoreTable
        config['Config Parms']['outputNgramsTable']=outputNgramsTable
        config['Config Parms']['prodRun'] = prodRun
        with open(path, 'w') as configfile:
            config.write(configfile)
        return "Configuration Parameters have been added successfully!!!"
    else:
        return "cannot find config params in file"


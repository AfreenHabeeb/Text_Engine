from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

import pandas as pd
import re, sys, os


sid = SentimentIntensityAnalyzer()

def sentScore(text, biasedTokens):
    # return engagement id, chat, aggregate sentiment score and existing customer


    text = re.sub('[^a-zA-Z0-9]+', ' ', text.replace('*', "").replace('-', "")).strip()
    line1 = text.split()

    token1 = [token for token in line1 if token.lower() not in biasedTokens]

    line2 = " ".join(token1)
    return (sid.polarity_scores(line2)['compound'])

def sentLabel(x):
    if x <= 0.25:  # any thing less than equals to 0 is 'Complaint' (negative) and above it is a 'Query' (query)
        return 'Complaint'
    else:
        return 'Query'

def nerorganization(text, nerOrg):

    tokens = word_tokenize(text)

    orgList = [token.upper() for token in tokens if token.lower() in nerOrg]

    return list(set(orgList))

def strconv(inp_list):
    convertedText = ",".join(inp_list)
    return convertedText

def replaceText(line, dic):
    for i, j in dic.items():
        i = i.lower()
        #line = line.replace(i,j)
        #line = re.sub(i, j, line, flags=re.I)  
        line = re.sub(r"\b%s\b" % i, j, line, flags=re.I)  
    return line


def fnSentimentAnalyzer(cc_df, repoDirectoryPath, folderToWrite, cc_df1, outputDataFormat,  runSentiment, prodRun):

    path = repoDirectoryPath + '/biasedTokens.txt'
    f = open(path, "r")
    biasedTokens = [pattern.strip() for pattern in f.readlines()]
    
    path1 = repoDirectoryPath + '/nerOrganisations.txt'
    g = open(path1, "r")
    nerOrg = [pattern.strip() for pattern in g.readlines()]

    #Reading specialPrograms.txt file
    path2 = repoDirectoryPath + '/specialPrograms.txt'
    h = open(path2, "r")
    specialPrograms = {}
    for pattern in h.readlines():
        pattern = pattern.strip()
        keyvalue = pattern.split(':')
        specialPrograms[keyvalue[0]] = keyvalue[1]

    
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply(lambda x: replaceText(x, specialPrograms))
    cc_df.loc[:,10]=cc_df[cc_df.columns[1]].apply(lambda x: sentScore(x, biasedTokens))
    cc_df.loc[:,20]=cc_df[cc_df.columns[1]].apply(lambda x: nerorganization(x, nerOrg))  
    
    cc_df_sa=cc_df.copy()   
    
    cc_df_sa.iloc[:,3]=cc_df_sa.iloc[:,3].apply(strconv)
    cc_df_sa.loc[:,30]=cc_df[cc_df_sa.columns[2]].apply(sentLabel)
    
    sa_df=cc_df_sa.copy()
    
    sa_df.rename(columns={sa_df.columns[0]: "IDNumber", sa_df.columns[1]: "text", sa_df.columns[2]: 'Score', sa_df.columns[3]: "NERtokens" , sa_df.columns[4]: 'label'}, inplace=True)
    cc_df.rename(columns={cc_df.columns[0]:'IDNumber', cc_df.columns[1]:'text',cc_df.columns[2]:'Score', cc_df.columns[3]:'NERtokens'}, inplace=True)
    # sa_df=sa_df.astype({'IDNumber':'str','text':'str','Score':'float','NERtokens':'str','label':'str'})
    
    
    cc_df1.rename(columns={cc_df1.columns[0]:'IDNumber', cc_df1.columns[1]:'text'}, inplace=True)
    
    
    if runSentiment.lower() == 'y':
        # Merging the two dataframes for the output
        
        cc_df1['IDNumber']=cc_df1['IDNumber'].astype(str)
        # Joined_data = sa_df.join(cc_df1.set_index("IDNumber"), on="IDNumber", lsuffix="dupl").drop('textdupl', axis=1)
        Joined_data = sa_df.join(cc_df1.set_index("IDNumber"), on="IDNumber", lsuffix="dupl")
        
        cols=Joined_data.columns.tolist()
        cols=[cols[0]]+[cols[4]]+cols[1:4]+cols[5:len(cols)]
        Joined_data=Joined_data[cols]


        if prodRun.lower() != 'y' :
            if outputDataFormat.lower() == 'file':
                #Write Sentiment Analysis file
                try:
                    #os.chdir(folderToWrite)  # change directory
                    pathToSaveSA = folderToWrite + 'Sentiment_Analysis/'
                    if not os.path.exists(pathToSaveSA):
                        os.makedirs(pathToSaveSA)
                        
                    pathToSaveSA=pathToSaveSA+'SA.csv'
                    Joined_data.to_csv(pathToSaveSA, header=True, index=False)

                except Exception as error:
                    print ('Error writing the sentiment analysis files')
                    print(error)

            else:
                print ('Output Data Format should be file')
                sys.exit()

        else:
            print ('prodRun = y selected, data will be written into in merge function')
    return cc_df, Joined_data








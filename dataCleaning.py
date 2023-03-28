import re, sys


import numpy as np
import pandas as pd

def removeMasking(a):
    try:
        # convert content to string since Spark does not have replace method
        # a is the "customercontent" column. One row at a time comes in.
        #a = a.lower()
        # remove punctuations
        # a = re.sub('[^a-zA-Z0-9]+', ' ', a.replace('*',"").replace('-',"")).strip()
        a = a.replace('*', "").replace('-', "").strip()
        # remove multiple white spaces
        a = re.sub('\s+', ' ', a.replace('*', "").replace('-', "")).strip()
        a = re.sub('x{2,}', '', a)
        a = re.sub('X{2,}', '', a)


        # the zero indicates no parsing error, 1 means ascii to unicode conversion error
        return a
    except AttributeError:
        return 0

def convertUniqueIdToString(id):
    return str(id)


def fnReadData( inputDataFormat, inputDataSeparator, path, inputHeader, generateUniqueID, groupByID):
    if inputDataFormat == 'file':
        # Reading a tab delimited file
        try:
            cc_df=pd.read_csv( path , encoding='latin-1', sep=inputDataSeparator)
            
            # taking first 100 entries as sample input
            cc_df=cc_df.head(100).copy()
        except Exception as error:
            print ('Error reading the input file')
            print(error)
            sys.exit()
    else:
        print ('Input Data Format should be either file')
        sys.exit(1)

    # Replacing the '.' in the column names with '_'
    cols = cc_df.columns
    new_cols = []
    for column in cols:
        x = column.replace(".", "_")
        new_cols.append(x)
    cols = new_cols
    cc_df = cc_df[cols]

    # Generating unique identifier
    if generateUniqueID.lower() == 'y':
    
        # generated_ids=pd.DataFrame(np.arange(1,len(cc_df), dtype=int), columns='IDNumber')
        # cc_df.insert(0, 'IDNumber', generated_ids)
        # cc_df.insert(loc=0, column='idNumber', value=np.arange(len(cc_df)))
        cc_df.insert(loc=0, column='idNumber', value=np.arange(len(cc_df)))
        cc_df[cc_df.columns[0]]=cc_df[cc_df.columns[0]].astype(str)
    else:
        cc_df=cc_df.rename(columns={cc_df.columns[0]:'IDNumber', cc_df.columns[1]: 'text'})

    if groupByID.lower() == 'y':
        # grouping logic goes here - Group by Engagement ID/ MTN Number / any ID
        cc_df.rename(columns={cc_df.columns[0]: "IDNumber", cc_df.columns[1]: "Text"}, inplace=True)
        
        cc_df = cc_df.groupby('IDNumber',as_index=False).agg({'Text':'.'.join , 'cat':lambda x: x.iloc[0]})
    #
    # else:
    #     # dropping duplicate Engagement IDs/MTNs/ any ID
    #     cc_df = cc_df.drop_duplicates(subset='IDNumber', keep="first")
        
          
        
    # breaking the data frame into 2 dataframes (cc_df1 is the backup)
    cc_df1 = cc_df.copy()


    cc_df=cc_df[cc_df.columns[[0,1]]].copy()
    # Choosing the IDNumber and Text
    print("Printing cc_df")
    print(cc_df.head(5))
    cc_df[cc_df.columns[1]]=cc_df[cc_df.columns[1]].apply( removeMasking)

    print("Printing cc_df1")
    print(cc_df1.head(5))

    return cc_df, cc_df1   # cc_df - Contains only ID, Text and cc_df1 contains original data (as-is)













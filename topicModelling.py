

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np

def optimize_topics(cc_df):
    x=cc_df[cc_df.columns[1]].apply(' '.join)

    my_docs=x.tolist()

    #dtm_corpus=np.array(my_docs)
    vectorizer =  TfidfVectorizer()
    dtm_corpus= vectorizer.fit_transform(my_docs)

    max_topics=50
    

    WSSSE = np.zeros(max_topics)  # Within Set Sum of Squared Errors
    diff = np.zeros(max_topics)
    diff2 = np.zeros(max_topics)
    diff3 = np.zeros(max_topics)

    distorsions = []
    ks = range(2, max_topics + 1)
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dtm_corpus)
        distorsions.append(kmeans.inertia_)
        
        WSSSE[k - 1] = kmeans.inertia_
        if k > 1:
            diff[k - 1] = WSSSE[k - 1] - WSSSE[k - 2]
        if k > 2:
            diff2[k - 1] = diff[k - 1] - diff[k - 2]
        if k > 3:
            diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

    elbow = np.argmin(diff3[3:]) + 3
    num_of_topics = ks[elbow - 1]



    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 51), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.savefig('elbow_curve.png')
    
    

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(5, 51), diff3[4:])
    plt.grid(True)
    plt.title('diff curve')
    plt.savefig('diff_curve.png')
    
    
    return num_of_topics

    

#def fntopicModelling(cc_df, num_topics):
def fntopicModelling(cc_df, folderToWrite, u_numOfTopics, optimizeTopics):
    if optimizeTopics.lower()=='y':
        num_topics=optimize_topics(cc_df)
    else:
        num_topics=u_numOfTopics
    #num_topics=3


    my_docs=cc_df[cc_df.columns[1]].tolist()

    
    #create unique word dictionary
    from gensim import corpora
    unique_dictionary=corpora.Dictionary(my_docs)
    
    #create document term matrix
    dtm_corpus=[unique_dictionary.doc2bow(text) for text in my_docs]
    
    import gensim
    lda = gensim.models.ldamodel.LdaModel
    
    #optimize(num_topics)
    
    from pprint import pprint
    lda_model = lda(dtm_corpus, num_topics=num_topics, id2word = unique_dictionary, passes=15)

    print('LDA Results: ')
    pprint(lda_model.print_topics())

    #storing lda information    
    
    top_words_per_topic = []
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 10)])

    
    top_words=pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'Weights'])
    
    

    
    #pyldavis
    corpora.MmCorpus.serialize('corpus.mm', dtm_corpus)
    corp = gensim.corpora.MmCorpus('corpus.mm')

    import pyLDAvis
    import pyLDAvis.gensim

    vis = pyLDAvis.gensim.prepare(lda_model, corp, unique_dictionary, sort_topics=True)
    
    topic_terms= vis.topic_info


    
    #returning only those topic words that fall in Default category
    #return topic_terms[topic_terms.Category=='Default'][topic_terms.columns[2]], topic_terms.reset_index()
    
    
    pathToSave=folderToWrite+'LDA_visualization/'
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
        pyLDAvis.save_html(vis,pathToSave+'topic_vis.html')
        lda_model.save(pathToSave+'lda_model.model')
        
        top_words.to_csv(pathToSave+"top_words.csv")

    #returning all topic words
    return topic_terms.reset_index(), top_words


    
    
    

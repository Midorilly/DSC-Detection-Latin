import matplotlib.pyplot as plt

# Though the following import is not directly being used, it is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np

from sklearn.cluster import AffinityPropagation
import pandas as pd
import pickle
import torch
import logging
import graphQuery 
from sklearn.decomposition import PCA
import os
import importlib
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn import metrics

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def deserialize(file):
    f = open(file, 'rb')
    obj = pickle.load(f)
    return obj

def serialize(file, item):
    pkl = open(file, 'wb')
    pickle.dump(item, pkl)
    pkl.close

def getDataset(path, word):
    bc = pd.DataFrame(data=deserialize(os.path.join(path, 'BC', word, word+'.bert')))
    ad = pd.DataFrame(data=deserialize(os.path.join(path, 'AD', word, word+'.bert')))
    frames = [bc, ad]
    dataset = pd.concat(frames, ignore_index=True)
    return dataset

def getExternalKnowledge(dataset : pd.DataFrame, mode, embeddings):

    dates = []
    authors = []
    books = []

    updatedDataset = dataset.copy(True)
    dropped = 0
    for i, row in dataset.iterrows():
        info = graphQuery.executeQuery(graphQuery.authorQuery, row['hash'])
        period = graphQuery.executeQuery(graphQuery.dateQuery, row['hash'])
        if (len(info) > 0) and (len(period) > 0):
            for o in info:
                authors.append(o.authorLabel)
                books.append(o.bookLabel)  
                if mode == 'concat':
                    authorEmb = getEmbedding(embeddings, o.author)
                    encodedAuthors.append(authorEmb)
                    bookEmb = getEmbedding(embeddings, o.book)
                    encodedBooks.append(bookEmb)

            for o in period:
                if '-' in o.date or '0000' in o.date:
                    dates.append('BC')
                else:
                    dates.append('AD')
        elif len(info) == 0 or len(period) == 0:
            dropped += 1
            updatedDataset = updatedDataset.drop(index=i)

    if mode == 'naive':
        labelEncoder = LabelEncoder()
        encodedAuthors = labelEncoder.fit_transform(authors)
        encodedBooks = labelEncoder.fit_transform(books)

    updatedDataset['date'] = pd.Series(dates, index=updatedDataset.index)
    updatedDataset['author'] = pd.Series(authors, index=updatedDataset.index)
    updatedDataset['book'] = pd.Series(books, index=updatedDataset.index)
    if mode != 'bert':
        updatedDataset['encodedAuthor'] = pd.Series(encodedAuthors, index=updatedDataset.index)
        updatedDataset['encodedBook'] = pd.Series(encodedBooks, index=updatedDataset.index)

    updatedDataset = combineFeatures(updatedDataset, mode)

    return updatedDataset

def getEmbedding(embeddings, node):
    return embeddings[str(node)]

def combineFeatures(dataset, mode):
    features = []
    if mode == 'naive':
        for i, row in dataset.iterrows():
            features.append([row['encodedAuthor']*0.5, row['encodedBook']*0.5] + row['target'].tolist())
    elif mode == 'concat':
        for i, row in dataset.iterrows():
            c = row['encodedAuthor'].tolist() + row['encodedBook'].tolist() + row['target'].tolist()
            features.append(c)
    elif mode == 'bert':
        for i, row in dataset.iterrows():
            features.append(row['target'].tolist())
    
    dataset['features'] = features
    
    return dataset

def predict(dataset, uniqueLabels, threshold):

    changed = False
    for i in uniqueLabels:
        subset = dataset[dataset.cluster == i]
        clusterSize = subset.shape[0]
        adCount = (subset.date == 'AD').sum()
        
        percentage = (float(adCount) / float(clusterSize))
        if percentage >= threshold :
            changed = True

    if changed:
        prediction = 1
    else: 
        prediction = 0
    
    return prediction

def gridSearch():
    hyperparameters = {'damping': [0.7, 0.8, 0.9, 0.95], 
                       'preference' : [None, -50, -100, -120, -140, -160, -180, -200]}
    groundtruth = pd.read_csv(groundtruthPath, header=0)

    silhouettes = {}
    
    for d in hyperparameters['damping']:
        for p in hyperparameters['preference']:
            logger.info('[ESTIMATING] {}, {} ...'.format(d, p))
            estimator = AffinityPropagation(damping=d, preference=p)
            temp = []
            for i, row in groundtruth.iterrows():
                updatedGroundtruth = groundtruth
                dataset = getDataset(sentencesPath, row['word'])
                enhancedDataset = getExternalKnowledge(dataset)
                X = np.array(enhancedDataset['features'].values.tolist())

                if len(X) > 1 :
                    labels = estimator.fit_predict(X)
                    enhancedDataset['cluster'] = pd.Series(labels, index=enhancedDataset.index)
                    uniqueLabels = np.unique(labels)

                    if len(uniqueLabels) >= 2:
                        silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
                        temp.append((row['word'], silhouette))
                        
                else:
                    updatedGroundtruth = updatedGroundtruth.drop(index=i)
            silhouettes[(d,p)] = temp

    return silhouettes

def getBestParams(silhouettes):
    groundtruth = pd.read_csv(groundtruthPath, header=0)
    search = {}
    for idx, row in groundtruth.iterrows():            
        tmpMax = 0
        tmpK = None
        for k, item in silhouettes.items():
            for i in item:
                if i[0] == row['word'] :
                    if i[1] > tmpMax :
                        tmpMax = i[1]
                        tmpK = k
        search[row['word']] = (tmpMax, tmpK)

    return search

# python3 --sentencesPath --groundtruthPath --outputPath

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--bertPath', help='path to Latin BERT embeddings', required=True)
    parser.add_argument('-g', '--groundtruthPath', help='path to groundtruth', required=True)
    parser.add_argument('-o', '--outputPath', help='path to output', required=True)
    parser.add_argument('-e', '--embeddingsPath', help='path to node embeddings', required=False)
    parser.add_argument('-m', '--mode', help='concat/naive/bert', required=True)
    args = vars(parser.parse_args())
    sentencesPath=args['sentencesPath']
    groundtruthPath=args['groundtruthPath']
    outputPath=args['outputPath']
    embeddingsPath=args['embeddingsPath']
    mode=args['mode']

    importlib.reload(graphQuery)
    embeddings = deserialize(embeddingsPath)
    # grid-search
    #silhouettes = gridSearch()
    #serialize(os.path.join(outputPath, 'silhouettesScore-'+mode+'.dct'), silhouettes)
    silhouettes = deserialize(os.path.join(outputPath, 'silhouettesScore-'+mode+'.dct'))
    bestParams = getBestParams(silhouettes)
    print(bestParams)

    groundtruth = pd.read_csv(groundtruthPath, header=0)
    thresholds = np.arange(0.6, 1.0, 0.001)

    evaluation = {}
    predictionDict = {}
    #embList = []
    #graphList = []
    predictionList = []
    
    for idx, row in groundtruth.iterrows():
        if row['word'] in list(bestParams.keys()):
            estimator = AffinityPropagation(damping=bestParams[row['word']][1][0], preference=bestParams[row['word']][1][1])
            logger.info('[TARGET WORD] {}'.format(row['word']))

            dataset = getDataset(sentencesPath, row['word'])
            enhancedDataset = getExternalKnowledge(dataset)

            # for evaluating BERT only 
            #data = [(Y, 'emb', enhancedDataset)] 
            #Y = np.array(enhancedDataset['target'].values.tolist())

            # new features
            X = np.array(enhancedDataset['features'].values.tolist())
            
            data = [(X, 'graph', enhancedDataset)]
            i = 1
            fig = plt.figure(figsize=(10, 5))
            for (set, name, frame) in data:   
                ax = fig.add_subplot(1, 2, i, projection="3d", elev=48, azim=134)
                i += 1
                if len(set) > 10 :           
                    labels = estimator.fit_predict(set)
                    tmpFrame = frame
                    tmpFrame['cluster'] = pd.Series(labels, index=tmpFrame.index)
                    tmpFrame[['words', 'book', 'author', 'date', 'cluster']].to_csv(os.path.join(outputPath, row['word']+'_'+name+'.csv'))
                    uniqueLabels = np.unique(labels)

                    bestAccuracy = 0
                    bestTh = 0
                    bestPrediction = None
                    for th in thresholds:
                        prediction = predict(tmpFrame, uniqueLabels, th)
                        accuracy = metrics.accuracy_score([prediction], [row['type']])
                        if accuracy > bestAccuracy:
                            bestAccuracy = accuracy
                            bestTh = th
                            bestPrediction = prediction
                            break
                        else:
                            bestPrediction = prediction
                    evaluation[(row['word'], name)] = (bestTh, bestAccuracy)
                    #if name == 'emb':
                    #    embList.append(bestPrediction)
                    #elif name == 'graph':
                    #    graphList.append(bestPrediction)
                    predictionList.append(bestPrediction)

                    # for plotting
                    pca = PCA(2)
                    df = pca.fit_transform(set)
                    for l in uniqueLabels:
                        ax.scatter(df[labels == l, 0], df[labels == l, 1], label=l)

                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    ax.set_xlabel("1st eigenvector")
                    ax.set_ylabel("2nd eigenvector")
                    ax.set_title('[{}] {} clusters'.format(row['word'], len(uniqueLabels)))

            plt.savefig(os.path.join(outputPath, row['word']+'.png'))

    serialize(os.path.join(outputPath, 'evaluation.dct'), evaluation)

    logger.info('[EVALUATION]')
    accuracy = metrics.accuracy_score(groundtruth['type'].values.tolist(), predictionList)
    precision = metrics.average_precision_score(groundtruth['type'].values.tolist(), predictionList)
    f1micro = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList, average='micro')
    f1macro = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList, average='macro')
    f1 = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList)

    #embAccuracy = metrics.accuracy_score(groundtruth['type'].values.tolist(), embList)
    #embPrecision = metrics.average_precision_score(groundtruth['type'].values.tolist(), embList)
    #embF1micro = metrics.f1_score(groundtruth['type'].values.tolist(), embList, average='micro')
    #embF1macro = metrics.f1_score(groundtruth['type'].values.tolist(), embList, average='macro')
    #embF1 = metrics.f1_score(groundtruth['type'].values.tolist(), embList)

    logger.info('[{}]\nAverage Accuracy: {}\nAverage Precision: {}\nF1-score: {}\nF1-micro: {}\nF1-macro: {}'.format(mode.upper(), accuracy, precision, f1, f1micro, f1macro))
    #logger.info('[BERT]\nAverage Accuracy: {}\nAverage Precision: {}\nF1-score: {}\nF1-micro: {}\nF1-macro: {}'.format(embAccuracy, embPrecision, embF1, embF1micro, embF1macro))

        


   

    



    
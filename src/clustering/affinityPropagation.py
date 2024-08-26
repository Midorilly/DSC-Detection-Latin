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
from scipy.signal import convolve2d
from time import localtime, strftime, time

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
                #if mode == 'concat':
                #if mode == 'concat' or mode =='product_book_bert' or mode == 'product_author_bert' or mode == 'product' or mode == 'sum_product' or mode == 'concat_product' or mode == 'max' or mode == 'min' or mode == 'conv' or mode == 'circ_conv':
                if mode != 'naive' and mode != 'bert':
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

# author: Vito Vicenti
def circular_convolution2d(A, B):
    m, n = A.shape
    p, q = B.shape
    M = max(m, p)
    N = max(n, q)
    A_padded = np.pad(A, ((0, M - m), (0, N - n)), mode='constant')
    B_padded = np.pad(B, ((0, M - p), (0, N - q)), mode='constant')
    A_fft = np.fft.fft2(A_padded)
    B_fft = np.fft.fft2(B_padded)
    C_fft = A_fft * B_fft
    C_padded = np.fft.ifft2(C_fft)
    C = np.real(C_padded)
    return C

# author: Vito Vicenti
def combineFeatures(dataset, mode):

    features = []

    if mode == 'naive':

        #for i, row in dataset.iterrows():
        features = [dataset['encodedAuthor'].values.tolist()*0.5, dataset['encodedBook'].values.tolist()*0.5] + dataset['target'].tolist()

    elif mode == 'concat':

        #for i, row in dataset.iterrows():
        features = dataset['encodedAuthor'].values.tolist() + dataset['encodedBook'].values.tolist() + dataset['target'].values.tolist()
        #    features.append(c)

    elif mode == 'bert':

        features = dataset['target'].values.tolist()
        #for i, row in dataset.iterrows():
        #    features.append(row['target'].tolist())

    else:

        author_emb = np.array(dataset['encodedAuthor'].values.tolist())
        book_emb = np.array(dataset['encodedBook'].values.tolist())
        bert_emb = np.array(dataset['target'].values.tolist())
   
        if mode == 'product_author_bert':

            n_components = min(author_emb.shape[0], bert_emb.shape[0], bert_emb.shape[1])
            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_auth_emb = pca.fit_transform(author_emb)

            value1 = np.dot(new_auth_emb, new_bert_emb)
            norm1 = np.linalg.norm(value1)
            normalized1 = value1 / norm1

            for i in range(len(normalized1)):
                features.append(normalized1[i])

        elif mode == 'product_book_bert':

            n_components = min(book_emb.shape[0], book_emb.shape[1], bert_emb.shape[0], bert_emb.shape[1])

            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_book_emb = pca.fit_transform(book_emb)

            value2 = np.dot(new_book_emb, new_bert_emb)
            norm2 = np.linalg.norm(value2)
            normalized2 = value2 / norm2

            for i in range(len(normalized2)):
                features.append(normalized2[i])

        #Concat(Book*BERT, Author*BERT)
        elif mode == 'concat_product':

            n_components = min(author_emb.shape[0], bert_emb.shape[0], bert_emb.shape[1])
            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_auth_emb = pca.fit_transform(author_emb)

            value1 = np.dot(new_auth_emb, new_bert_emb)
            norm1 = np.linalg.norm(value1)
            normalized1 = value1 / norm1

            n_components = min(book_emb.shape[0], book_emb.shape[1], bert_emb.shape[0], bert_emb.shape[1])
            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_book_emb = pca.fit_transform(book_emb)

            value2 = np.dot(new_book_emb, new_bert_emb)
            norm2 = np.linalg.norm(value2)
            normalized2 = value2 / norm2

            for i in range(len(value1)):
                combined = np.concatenate([normalized1[i], normalized2[i]])
                features.append(combined)

        elif mode == 'max' or mode == 'min':

            n_components = min(author_emb.shape[0], book_emb.shape[0], bert_emb.shape[0], bert_emb.shape[1])
            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_auth_emb = pca.fit_transform(author_emb)
            new_book_emb = pca.fit_transform(book_emb)

            if mode == 'max':
                max_auth_book = np.maximum(new_auth_emb, new_book_emb)
                max_combined = np.maximum(max_auth_book, new_bert_emb)

                for i in range(len(dataset)):
                    features.append(max_combined[i])

            if mode == 'min':
                min_auth_book = np.minimum(new_auth_emb, new_book_emb)
                min_combined = np.minimum(min_auth_book, new_bert_emb)

                for i in range(len(dataset)):
                    features.append(min_combined[i])

        elif mode == 'conv':
            
            conv_AB = convolve2d(author_emb, book_emb, mode='same')  
            result = convolve2d(conv_AB, bert_emb, mode='same')
            
            norm = np.linalg.norm(result)
            normalized = result / norm

            for i in range(len(normalized)):
                features.append(normalized[i])

        elif mode == 'circ_conv':

            AB_conv = circular_convolution2d(author_emb, book_emb)
            ABC_conv = circular_convolution2d(AB_conv, bert_emb)

            norm = np.linalg.norm(ABC_conv)
            normalized = ABC_conv / norm

            for i in range(len(normalized)):
                features.append(normalized[i])

        #(Book+author)*BERT
        elif mode == 'sum_product':

            comb_emb = author_emb + book_emb

            n_components = min(comb_emb.shape[0], bert_emb.shape[0], bert_emb.shape[1])
            pca = PCA(n_components=n_components)
            new_bert_emb = pca.fit_transform(bert_emb)
            new_comb_emb = pca.fit_transform(comb_emb)
            value = np.dot(new_comb_emb, new_bert_emb)
            norm = np.linalg.norm(value)
            normalized = value / norm

            for i in normalized:
                features.append(i)

    dataset['features'] = features
    
    return dataset


'''def combineFeatures(dataset, mode):
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
    
    return dataset'''

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

def gridSearch(mode, embeddings):
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
                enhancedDataset = getExternalKnowledge(dataset, mode, embeddings)
                X = np.array(enhancedDataset['features'].values.tolist())

                if len(X) > 1 :
                    labels = estimator.fit_predict(X)
                    enhancedDataset['cluster'] = pd.Series(labels, index=enhancedDataset.index)
                    uniqueLabels = np.unique(labels)

                    if len(uniqueLabels) >= 2 and len(uniqueLabels) <= (len(enhancedDataset['features'].values.tolist()) - 1):
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
    sentencesPath=args['bertPath']
    groundtruthPath=args['groundtruthPath']
    mode=args['mode']
    outputPath=os.path.join(args['outputPath'], 'log_' + mode + '_' + strftime("%m%d%H%M%S", localtime()))
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
        os.mkdir(os.path.join(outputPath, 'img'))
        os.mkdir(os.path.join(outputPath, 'clusters'))
    embeddingsPath=args['embeddingsPath']
   
    importlib.reload(graphQuery)
    embeddings = deserialize(embeddingsPath)
    # grid-search
    #silhouettes = gridSearch(mode, embeddings)
    #serialize(os.path.join(outputPath, 'silhouettesScore-'+mode+'.dct'), silhouettes)
    #silhouettes = deserialize(os.path.join(outputPath, 'silhouettesScore-'+mode+'.dct'))
    #bestParams = getBestParams(silhouettes)
    #print(bestParams)

    groundtruth = pd.read_csv(groundtruthPath, header=0)
    thresholds = np.arange(0.7, 1.0, 0.001)

    evaluation = {}
    predictionDict = {}
    #embList = []
    #graphList = []
    predictionList = []
    thresholdDict = {}
    
    for idx, row in groundtruth.iterrows():
        #if row['word'] in list(bestParams.keys()):
            #estimator = AffinityPropagation(damping=bestParams[row['word']][1][0], preference=bestParams[row['word']][1][1])
            estimator = AffinityPropagation()
            logger.info('[TARGET WORD] {}'.format(row['word']))

            dataset = getDataset(sentencesPath, row['word'])
            enhancedDataset = getExternalKnowledge(dataset, mode, embeddings)

            # for evaluating BERT only 
            #Y = np.array(enhancedDataset['target'].values.tolist())
            #data = [(Y, 'emb', enhancedDataset)] 

            # new features
            X = np.array(enhancedDataset['features'].values.tolist())
            data = [(X, mode, enhancedDataset)]

            i = 1
            fig = plt.figure(figsize=(8, 5))
            for (set, name, frame) in data:   
                ax = fig.add_subplot(1, 1, i, projection="3d", elev=48, azim=134)
                i += 1
                if len(set) > 10 :           
                    labels = estimator.fit_predict(set)
                    tmpFrame = frame
                    tmpFrame['cluster'] = pd.Series(labels, index=tmpFrame.index)
                    tmpFrame[['words', 'book', 'author', 'date', 'cluster']].to_csv(os.path.join(outputPath, 'clusters', row['word']+'_'+name+'.csv'))
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
                    evaluation[(row['word'], mode)] = (bestTh, bestAccuracy)
                    #if name == 'emb':
                    #    embList.append(bestPrediction)
                    #elif name == 'graph':
                    #    graphList.append(bestPrediction)
                    predictionList.append(bestPrediction)

                    # for plotting
                    pca = PCA(3)
                    df = pca.fit_transform(set)
                    for l in uniqueLabels:
                        ax.scatter(df[labels == l, 0], df[labels == l, 1], label=l)

                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    ax.zaxis.set_ticklabels([])
                    ax.set_xlabel("1st eigenvector")
                    ax.set_ylabel("2nd eigenvector")
                    ax.set_zlabel("3rd eigenvector")
                    ax.set_title('[{}] {} clusters with {}'.format(row['word'], len(uniqueLabels), mode))

            plt.savefig(os.path.join(outputPath, 'img', row['word']+'-'+mode+'.png'))
            plt.close()

    serialize(os.path.join(outputPath, 'evaluation.dct'), evaluation)

    logger.info('[EVALUATION]')
    #accuracy = metrics.accuracy_score(groundtruth['type'].values.tolist(), predictionList)
    #precision = metrics.average_precision_score(groundtruth['type'].values.tolist(), predictionList)
    #f1micro = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList, average='micro')
    #f1macro = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList, average='macro')
    #f1 = metrics.f1_score(groundtruth['type'].values.tolist(), predictionList)

    #embAccuracy = metrics.accuracy_score(groundtruth['type'].values.tolist(), embList)
    #embPrecision = metrics.average_precision_score(groundtruth['type'].values.tolist(), embList)
    #embF1micro = metrics.f1_score(groundtruth['type'].values.tolist(), embList, average='micro')
    #embF1macro = metrics.f1_score(groundtruth['type'].values.tolist(), embList, average='macro')
    #embF1 = metrics.f1_score(groundtruth['type'].values.tolist(), embList)
    logger.info('[GROUNDTRUTH] {}'.format(groundtruth['type'].values.tolist()))
    logger.info('[PREDICTIONS] {}'.format(predictionList))
    report = metrics.classification_report(groundtruth['type'].values.tolist(), predictionList, zero_division=0.0)
    logger.info(report)
    with open(os.path.join(outputPath, 'performance.txt'), 'w') as file:
        file.write('[MODE] {}\n'.format(mode.upper()))
        file.write('[GROUNDTRUTH] {}\n'.format(groundtruth['type'].values.tolist()))
        file.write('[PREDICTIONS] {}\n'.format(predictionList))
        file.write('[REPORT]\n{}'.format(report))
    #logger.info('[{}]\nAverage Accuracy: {}\nAverage Precision: {}\nF1-score: {}\nF1-micro: {}\nF1-macro: {}'.format(mode.upper(), accuracy, precision, f1, f1micro, f1macro))
    #logger.info('[BERT]\nAverage Accuracy: {}\nAverage Precision: {}\nF1-score: {}\nF1-micro: {}\nF1-macro: {}'.format(embAccuracy, embPrecision, embF1, embF1micro, embF1macro))

        


   

    



    
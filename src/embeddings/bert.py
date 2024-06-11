# This code takes inspiration from https://github.com/dbamman/latin-bert

import argparse, sys
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from numpy import linalg as LA
import os
import pickle
import logging
import re
import latinTokenizer
import hashlib

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class LatinBERT():

    def __init__(self, tokenizerPath=None, bertPath=None):
        self.model = latinTokenizer.BertLatin(bertPath)
        self.model.to(device)
        encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
        self.tokenizer = latinTokenizer.LatinTokenizer(encoder)
        self.sentTokenizer = SentenceTokenizer()
        self.wordTokenizer = WordTokenizer()
        self.targetIDs = {}

    def getSentenceHash(self, sentence: str) -> str:
        splitSentence = sentence.split(' ', 5)
        spaceJoin = ' '.join(splitSentence[:5]).encode('utf-8')
        sentenceHash = int(hashlib.md5(spaceJoin).hexdigest(), 16)

        return sentenceHash

    def convertToTokens(self, sentences: list[str]) -> list[list[str]]:
        '''
        sentences: list of raw sentences
        sentencesList: list of tokenized sentences       
        '''
        sentencesList = []

        for sentence in sentences:            
            sentenceHash = self.getSentenceHash(sentence)
            tokenizedSent = bert.sentTokenizer.tokenize(re.sub(r'<[^>]+>', '', sentence).lower()) # remove </line><line> and </section><section> tokens 
            for s in tokenizedSent:
                tokenizedWords = self.wordTokenizer.tokenize(s)
                tokens = [t for t in tokenizedWords if t != '']
                for i, t in enumerate(tokens):
                    if t == 'target':
                        tokens.remove('target')
                        tokens.insert(i, '[TARGET]')
                tokens.insert(0, '[CLS]')
                tokens.append('[SEP]')
                sentencesList.append((sentenceHash, tokens))

        return sentencesList

    def generateTensors(self, tokenizedSentences, targetWord):

        batchHash = []
        batchTokens = []
        batchIndexes = []
        batchMasks = []
        batchTransforms = []
        ids = []
        roots = []

        for sentenceHash, sentence in tokenizedSentences:
            tokens = []
            indexes = []
            mask = []
            transforms = []
            numberTokens = 0
            isTarget = False
            for word in sentence:                
                t = self.tokenizer.tokenize(word)
                numberTokens += len(t)
                tokens.append(t)
                indexedTokens = self.tokenizer.convert_tokens_to_ids(t)
                indexes.extend(indexedTokens)
                maskIDs = [1]*len(indexedTokens)
                mask.extend(maskIDs)
                if isTarget:
                    ids.append(indexedTokens[0])
                    roots.append(t[0])
                    isTarget = False
                if word == '[TARGET]':
                    isTarget = True

            currentToken = 0
            for idxT, word in enumerate(sentence):
                t = tokens[idxT]
                transform = list(np.zeros(numberTokens))
                for i in range(currentToken, currentToken+len(t)):
                    transform[i] = 1./len(t)
                currentToken += len(t)

                transforms.append(transform)    

            indexesTensor = torch.tensor([indexes]) # ids 
            attentionTensor = torch.FloatTensor([mask]) # mask
            transformTensor = torch.FloatTensor([transforms]) # transform

            self.targetIDs[targetWord] = {'index': ids, 'root': roots}
            batchHash.append(sentenceHash)
            batchTokens.append(tokens)
            batchIndexes.append(indexesTensor)
            batchMasks.append(attentionTensor)
            batchTransforms.append(transformTensor)

        return batchHash, batchTokens, batchIndexes, batchMasks, batchTransforms

    def bertEmbeddings(self, indexesTensors, attentionTensor, transformTensor):
        with torch.no_grad():
            outputs = self.model(input_ids=indexesTensors, attention_mask=attentionTensor, transforms=transformTensor)       
        tokenEmbeddings = outputs[-1]
        tokenEmbeddings = torch.squeeze(tokenEmbeddings, dim=0).to(device)
        listTokenEmbeddings = [tokenEmbed.tolist() for tokenEmbed in tokenEmbeddings]

        return listTokenEmbeddings

def serialize(file, item):
    pkl = open(file, 'wb')
    pickle.dump(item, pkl)
    pkl.close

# python3 ML/src/bert.py --bertPath latin-bert/models/latin_bert/ --tokenizerPath latin-bert/models/subword_tokenizer_latin/latin.subword.encoder --sentencesPath  --outputPath 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
    parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
    args = vars(parser.parse_args())

    bertPath=args['bertPath']
    tokenizerPath=args['tokenizerPath']
    
    paths = [('../data/fragments/target/BC', '../output/BC'), 
            ('../data/fragments/target/AD', '../output/AD')]
    
    bert = LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)
    
    for (sentencesPath, outputPath) in paths:
        period = outputPath.rsplit('/', 1)[0]

        for file in os.listdir(sentencesPath):
            sentsFile = os.fsdecode(file)  

            targetWord = sentsFile.split('_')[0]
            logger.info('[TARGET WORD] {}'.format(targetWord))

            # OUTPUT FILES
            outputFolder = os.path.join(outputPath, targetWord)
            if not os.path.exists(outputFolder):
                os.mkdir(outputFolder) # create a folder for each lemma       
            predictionFile = os.path.join(outputFolder, targetWord+'_corpus.bert') # embedded tokens
            targetFile = os.path.join(outputFolder, targetWord+'.bert')

            corpusTextFile = open(os.path.join(outputFolder, targetWord+'_corpus.csv'), 'w', encoding='utf-8')
            targetTextFile = open(os.path.join(outputFolder, targetWord+'.csv'), 'w', encoding='utf-8')

            # DATASET FILE
            f = open(os.path.join(sentencesPath, sentsFile), 'r')
            sentences = f.readlines()

            predictionsList = [] # all sentences
            targetDictionary = {} # target sentences

            wordsList = []
            leftEmbList = []
            wordEmbList = []
            productEmbList = []
            sumEmbList = []
            hashList = []

            tokenizedSentences = bert.convertToTokens(sentences)
            batchHash, batchTokens, batchIndexes, batchMasks, batchTransforms = bert.generateTensors(tokenizedSentences, targetWord)
            x = 0    
            for b in range(len(batchTokens)):
                listTokenEmbeddings = bert.bertEmbeddings(batchIndexes[b], batchMasks[b], batchTransforms[b])
                predictionsList.append(listTokenEmbeddings)
                for idx, e in enumerate(listTokenEmbeddings):
                    corpusTextFile.write('{} \t {}\n'.format(batchTokens[b][idx], e))
                corpusTextFile.write('\n')
                for i, t in enumerate(batchTokens[b]):           
                    if '[TARGET]' in t:
                        x = x+1
                        wordIndex = i+1
                        
                        leftEmb = listTokenEmbeddings[0]/LA.norm(listTokenEmbeddings[0])
                        leftCentroid = np.mean(a = np.array(leftEmb), axis = 0)
                        leftEmbList.append(leftCentroid) # [CLS]

                        wordEmbedding = listTokenEmbeddings[wordIndex] / LA.norm(listTokenEmbeddings[wordIndex])
                        wordEmbList.append(wordEmbedding)
                        wordsList.append(batchTokens[b][wordIndex])
                        
                        targetTextFile.write('{} \t {} \t {}\n'.format(batchHash[b], batchTokens[b][wordIndex], wordEmbedding))

                        productEmbList.append(np.dot(leftCentroid, wordEmbedding))
                        sumEmbList.append([sum(x) for x in zip(leftEmb, wordEmbedding)])

                        hashList.append(batchHash[b])
                
            logger.info('[TOTAL] {}'.format(x))
            targetDictionary = {'words' : wordsList, 'cls': leftEmbList, 'target': wordEmbList, 'hash': hashList, 'product' : productEmbList, 'sum' : sumEmbList}

            # SERIALIZATION    
            serialize(predictionFile, predictionsList)
            serialize(targetFile, targetDictionary)
            corpusTextFile.close()
            targetTextFile.close()


    serialize(os.path.join(outputPath, period+'.dct'), bert.targetIDs)








    
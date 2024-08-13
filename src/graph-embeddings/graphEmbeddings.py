from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import pandas as pd
import os
import torch
from termcolor import colored
import pickle
from pykeen.hpo import hpo_pipeline
import graphQuery
import re
from numpy import linalg as LA

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def deserialize(file):
    f = open(file, 'rb')
    obj = pickle.load(f)
    return obj

def serialize(file, item):
    pkl = open(file, 'wb')
    pickle.dump(item, pkl)
    pkl.close

# not used

def gridSearch(training, testing, validation, models, epochs, dims, hpoPath):   

    for model in models:
        modelPath = os.path.join(hpoPath, model)
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)      
        for epoch in epochs:
            for dim in dims:
                print(colored('Optimizing {} with {} epochs, {} dim'.format(model, epoch, dim), 'yellow'))
                hpo_pipeline_result = hpo_pipeline(n_trials = 3,
                                                training = training,
                                                testing = testing,
                                                validation = validation,
                                                training_kwargs = {'num_epochs' : epoch, 'batch_size' : 64},
                                                model = model,
                                                model_kwargs= {'embedding_dim' : dim, 'random_seed' : 42},
                                                negative_sampler='basic',
                                                optimizer=torch.optim.AdamW,
                                                regularizer='lp',
                                                device = device)

                print(hpo_pipeline_result)
                path = os.path.join(modelPath, str(epoch)+'-'+str(dim))
                if not os.path.exists(path):
                    os.mkdir(path)
                hpo_pipeline_result.save_to_directory(path)

def queryGraph(entity_embedding_tensor, training):

    authors = graphQuery.g.subjects(predicate=graphQuery.RDF.type, object=graphQuery.SCHEMA.Person)
    books = graphQuery.g.subjects(predicate=graphQuery.RDF.type, object=graphQuery.SCHEMA.Book)
    entitiesList = []
    for s in authors:
        entitiesList.append(str(s))
    for s in books:
        entitiesList.append(str(s))
    print(entitiesList)
    entitiesDict = {}
    for k, i in training.entity_to_id.items():
        key = re.sub(r'[<>]', '', k)
        if key in entitiesList:
            entitiesDict[key] = entity_embedding_tensor[i] / LA.norm(entity_embedding_tensor[i])

    return entitiesDict

# training

def train(models, dims, training, testing, validation, srcPath, epoch):

    for model in models:
        for dim in dims:
            printline = model+' - k='+str(dim)
            print(colored('Starting ' + printline,'blue'))

            folder = os.path.join(srcPath, 'results', model+'_k='+str(dim)+'_e='+str(epoch))
            if not os.path.exists(folder):
                os.mkdir(folder)

            checkpointFile = os.path.join(srcPath, 'checkpoints', model+'_k='+str(dim)+'_e='+str(epoch))

            if os.path.isfile(folder+'/embeddings.tsv'):
                print(colored('Existing embedding in ' + folder, 'blue'))
                continue

            print(colored('Starting learning:' + folder,'blue'))
            print("Starting learning:", printline)

            result = pipeline(
                    training = training,
                    testing = testing,
                    validation = validation,
                    model = model,
                    model_kwargs={'embedding_dim' : dim, 'random_seed' : 42, 'scoring_fct_norm': 1},
                    negative_sampler='basic',
                    optimizer='adamw',
                    evaluation_fallback = True,
                    training_kwargs = {'num_epochs' : epoch,
                                        'checkpoint_name' : checkpointFile ,
                                        'checkpoint_directory' : os.path.join(srcPath, 'checkpoints'),
                                        'checkpoint_frequency' : 1,
                                        'batch_size' : 64},
                    device = device
            )

            torch.save(result, folder+'/pipeline_result.dat')

            map_ent = pd.DataFrame(data=list(training.entity_to_id.items()))
            map_ent.to_csv(folder+'/entities_to_id.tsv', sep='\t', header=False, index=False)
            map_rel = pd.DataFrame(data=list(training.relation_to_id.items()))
            map_rel.to_csv(folder+'/relations_to_id.tsv', sep='\t', header=False, index=False)

            # save mappings
            result.save_to_directory(folder, save_training=True, save_metadata=True)

            # extract embeddings with gpu
            entity_embedding_tensor = result.model.entity_representations[0](indices = None).cpu().data.numpy()
            # save entity embeddings to a .tsv file (gpu)
            df = pd.DataFrame(data=entity_embedding_tensor)

            entitiesDict = queryGraph(entity_embedding_tensor)

            outfile = folder + '/entities_embeddings.tsv'
            df.to_csv(outfile, sep='\t', header=False, index=False)

            # extract embeddings with gpu
            relation_embedding_tensor = result.model.relation_representations[0](indices = None).cpu().data.numpy()
            # save entity embeddings to a .tsv file (gpu)
            df = pd.DataFrame(data=relation_embedding_tensor)

            relationsDict = {}
            for k, i in training.relation_to_id.items():
                relationsDict[k] = relation_embedding_tensor[i] / LA.norm(relation_embedding_tensor[i])

            outfile = folder + '/relations_embeddings.tsv'
            df.to_csv(outfile, sep='\t', header=False, index=False)

            print(colored('Completed ' + printline,'green'))
            serialize(os.path.join(folder, 'entities.dict'), entitiesDict)
            serialize(os.path.join(folder, 'relations.dict'), relationsDict)

if __name__ == '__main__':

    dataset = '../data/llkg/llkg-lite-complete.tsv'

    models = ['TransE']
    dims = [2, 8, 16, 32]
    epochs = [5, 10, 20] 

    pykeenPath = '../pykeen'

    tf = TriplesFactory.from_path(dataset, entity_to_id=None, relation_to_id=None, create_inverse_triples=True)
    training, testing, validation = tf.split(ratios = [.8, .1, .1], random_state=42)
    # hyper-parameters optimization -- not used
    #hpoPath = '/home/ghizzotae/machine-learning/ML/pykeen/hpo'
    #gridSearch(training, testing, validation, models, epochs, dims, hpoPath)

    train(models, dims, training, testing, validation, pykeenPath, 20)


        
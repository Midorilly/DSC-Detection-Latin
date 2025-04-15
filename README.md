# Diachronic Semantic Change Detection in Latin

For Linked Linguistic Knowlege Graph, check [Midorilly/STKG-LLKG](https://github.com/Midorilly/STKG-LLKG)  
For downloading Latin BERT, check [dbamman/latin-bert](https://github.com/dbamman/latin-bert)

Project Organization
------------

    ├── data 
    |   ├── embeddings          <- Latin BERT target word embeddings.  
    |   ├── fragment
    |   |    └── target         <- Latin BERT input fragments.           
    |   └── llkg                <- Knowledge graph and schema.
    ├── evaluation              <- Clustering best hyper-parameters for evaluation.
    ├── pykeen                  <- Knowledge graph embeddings dictionaries.
    ├── src  
    |   ├── clustering          <- Code for clustering and downstream evaluation.
    |   ├── embeddings          <- Code for Latin BERT embeddings.
    |   └── graph-embeddings    <- Code for knowledge graph embeddings.
    ├── .gitignore
    ├── README.md
    ├── bert-requirements.txt
    ├── clustering-requirements.txt
    └── graph-requirements.txt

Citation
------------
```
Ghizzota, E., Basile, P., d'Amato, C., & Fanizzi, N. (2025). 
Linked Linguistic Knowledge Graph (1.0.0) [Data set]. 
Global WordNet Conference 2025 (GWC2025), Pavia, Italy. Zenodo. 
```
LLKG can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14623212).
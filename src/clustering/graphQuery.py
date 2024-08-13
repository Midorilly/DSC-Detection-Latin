from rdflib import Graph, Literal
from namespaces import *
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

g = Graph(store='Oxigraph')
g.parse('../data/llkg/llkg-lite-complete.ttl', format='ttl')

g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)
g.bind("dct", DCTERMS)
g.bind("owl", OWL)
g.bind("schema", SCHEMA)
g.bind("ontolex", ONTOLEX)
g.bind("vartrans", VARTRANS)
g.bind("lexinfo", LEXINFO)
g.bind("lime", LIME)
g.bind("wn", WORDNET)
g.bind("lexvo", LEXVO)
g.bind("lvont", LVONT)
g.bind("uwn", UWN)
g.bind("lila", LILA)
g.bind("skos", SKOS08)
g.bind("wd", WIKIENTITY)
g.bind("cc", CC)
g.bind("llkg", LLKG)

textQuery = ''' SELECT ?authorName ?bookName ?date ?id
    WHERE {{
        ?text schema:text {} ;
            schema:isPartOf ?document ;
            llkg:llkgID ?id .
            schema:datePublished ?date
        ?document schema:author ?author ;
            schema:name ?bookName .
        ?author rdfs:label ?authorName
    }}
'''

dateQuery = ''' SELECT ?date
    WHERE {
            ?text rdf:type schema:Quotation;
                llkg:hashID ?id ;
                schema:datePublished ?date.
    } LIMIT 1
'''

sentenceQuery = ''' SELECT ?sentence
    WHERE {      
            ?text rdf:type schema:Quotation;
                llkg:hashID ?id ;
                schema:text ?sentence.
    } LIMIT 1
'''

authorQuery = ''' SELECT ?author ?authorLabel ?book ?bookLabel
    WHERE {
        OPTIONAL { ?text rdf:type schema:Quotation ;
                        llkg:hashID ?id ;
                        schema:isPartOf ?book .
                    ?book rdf:type schema:Book ;
                        rdfs:label ?bookLabel ;
                        schema:author ?author .
                    ?author rdf:type schema:Person ;
                        rdfs:label ?authorLabel .            
            }
    
    } LIMIT 1
'''

def executeQuery(q: str, hash: int):
    output = g.query(q, initNs = {'schema' : SCHEMA, 'llkg' : LLKG}, initBindings={'id' : Literal(hash, datatype=XSD.integer)})
    return output

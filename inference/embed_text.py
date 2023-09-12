import numpy as np
from transformers import BertTokenizer, BertModel
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import sys
sys.path.insert(1, '../mm-locate-news')
from inference.ner_utils import *
from inference.nel_utils import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model = model.to('cuda')



class DocEmbeddings(object):
    def __init__(self):
        super().__init__()

        self.document_embeddings = self.load_embedding_model()

    def load_embedding_model(self):
        # print('Loading BERT Embeddings...')
        bert_embeddings = BertEmbeddings('bert-base-uncased') # bert-base-multilingual-cased
        return DocumentPoolEmbeddings([bert_embeddings])

    def embed(self, tokens):
        x = Sentence( tokens[:500] ) # for bert we use the first 500 tokens
        self.document_embeddings.embed(x)
        return x.embedding

def bert_embed_body(text):
    # print('getting body features ...')
    doc_embeddings = DocEmbeddings()
    output = doc_embeddings.embed(text).unsqueeze(0)

    return output


def get_bert_body_feature(text_input):
    return bert_embed_body(text_input)


def get_named_entities(input_text):
    cfg_nl = {  
                "wikifier": {
                "lang": {"deu":"de", "eng":"en", "cro":"hr"},
                "params": {
                        "extraVocabularies": "",
                        "wikiDataClasses": "false",
                        "wikiDataClassIds": "false",
                        "support": "true",
                        "ranges": "true",
                        "includeCosines": "true",
                        "maxMentionEntropy": "-1",
                        "maxTargetsPerMention": "20",
                        "minLinkFrequency": "1",
                        "pageRankSqThreshold": "-1",
                        "applyPageRankSqThreshold": "true",
                        "partsOfSpeech": "true",
                        "verbs": "true",
                        "nTopDfValuesToIgnore": "0"},
                "api_url": "http://www.wikifier.org/annotate-article",
                "user_key": "datswdkekkstrvqsekdlpjfzsuvtex"
                        }
            }

    ne_extractor = ner_spacy('eng')
    named_entities = ne_extractor.annotate( input_text )
    
    nel_extractor = wikifier()
    output = nel_extractor.annotate(
        named_entities,
        input_text,
        'eng',
        cfg_nl
        )

    return output


def get_bert_entity_feature(text_input):
    # print('getting named entities ...')
    nes = get_named_entities(text_input)
    doc_embeddings = DocEmbeddings()
    output = []

    for ne_ in nes:
        ne = ne_['uri'].split('/')[-1].replace('_', ' ')
        output.append ( doc_embeddings.embed(ne) )
    output = torch.stack(output, 0)
    output = torch.mean(output, 0)
    return output.unsqueeze(0)



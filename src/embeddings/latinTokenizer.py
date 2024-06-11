# Except for some minor changes, this code comes from https://github.com/dbamman/latin-bert

import argparse, sys
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
import os
import pickle
import logging
import re

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4
		self.vocab["[TARGET]"]=5

		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+6
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+6]=key

	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)
			elif token == "[TARGET]":
				wp_tokens.append(5)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text):
		tokens=text.split(" ")
		wp_tokens=[]
		for token in tokens:

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[TARGET]"}:
				wp_tokens.append(token)
			else:
				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+6])

		return wp_tokens

class BertLatin(nn.Module):

	def __init__(self, bertPath=None):
		super(BertLatin, self).__init__()

		self.bert = BertModel.from_pretrained(bertPath, output_hidden_states=True)
		self.bert.eval()
		
	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None):

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)
		outputs = self.bert.forward(input_ids, token_type_ids=None, attention_mask=attention_mask)
		#sequence_outputs = outputs['last_hidden_state'] 
		#pooled_outputs = outputs['pooler_output']
		
		hidden_states = outputs['hidden_states']
		cat_vec = torch.stack(hidden_states[-4:]).sum(0)
		all_layers=cat_vec
		out=torch.matmul(transforms,all_layers)
		return out
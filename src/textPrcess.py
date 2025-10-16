import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.llm_provider import LLMProvider

class TextProcessor:
    def __init__(self, text, name):
        self.text = text
        self.base_name = name
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.llm_provider = LLMProvider()
        self.embeddings = self.llm_provider.get_embedding_model()
    
    def split_sentences(self, text):
        """Support Chinese and English sentence segmentation (handling common abbreviations)"""
        # Added Chinese punctuation (references 4, 5)
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(text) if s.strip()]
        return sentences
    
    def generate_id(self, index):
        """Generate ID according to requirements"""
        return f"{self.base_name}{index+1}"  # Start numbering from 1
    
    def process(self):
        # Step 1: Convert PDF to text
        text = self.text
        
        # Step 2: Sentence segmentation and ID mapping
        sentences = self.split_sentences(text)
        for idx, sent in enumerate(sentences):
            sent_id = self.generate_id(idx)
            self.sentence_to_id[sent] = sent_id
            self.id_to_sentence[sent_id] = sent
        
        # Step 3: Vector storage
        vector = self.embeddings.embed_query(sentences[0])
        print(vector[:3])
        vectors = self.embeddings.embed_documents(sentences)
        return {
            "sentences": sentences,
            "vectors": vectors,
            "sentence_to_id": self.sentence_to_id,
            "id_to_sentence": self.id_to_sentence
        }
    
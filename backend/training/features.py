import nltk
import numpy as np
import spacy
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class FeatureExtractor:
    def __init__(self):
        # Load small spacy model for NLP features
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Pre-install it in your container image with: "
                "python -m spacy download en_core_web_sm"
            )
        
        # Load GPT-2 for perplexity scoring
        self.ppl_model_name = "gpt2"
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_name)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_name)
        self.ppl_model.eval()

    def get_stylometry(self, text):
        """
        Extracts stylometric features:
        - Sentence length variance
        - Lexical diversity (Type-Token Ratio)
        - Repetition score (based on bigrams)
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return { "sent_len_var": 0, "lexical_diversity": 0, "repetition_score": 0 }

        # 1. Sentence length variance
        sent_lens = [len(sent) for sent in sentences]
        sent_len_var = np.var(sent_lens) if len(sent_lens) > 1 else 0

        # 2. Lexical Diversity (TTR)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0

        # 3. Repetition Score (Bigrams)
        if len(words) > 1:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            repetition_score = (len(bigrams) - len(set(bigrams))) / len(bigrams)
        else:
            repetition_score = 0

        return {
            "sent_len_var": float(sent_len_var),
            "lexical_diversity": float(lexical_diversity),
            "repetition_score": float(repetition_score)
        }

    def get_perplexity(self, text):
        """
        Computes perplexity of the text using GPT-2.
        AI generated text often has lower perplexity.
        """
        encodings = self.ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
        return float(perplexity)

    def extract_all(self, text):
        features = self.get_stylometry(text)
        features["perplexity"] = self.get_perplexity(text)
        return features

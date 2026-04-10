import requests
from sentence_transformers import SentenceTransformer, util
import torch

class TrustAgent:
    def __init__(self):
        # Load a sentence transformer model for semantic similarity
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Simplified trust database / news sources (in production these would be live API searches)
        self.known_sources = {
            "reuters.com": 0.95,
            "apnews.com": 0.95,
            "bbc.com": 0.92,
            "nytimes.com": 0.90,
            "infowars.com": 0.10,
            "theonion.com": 0.05
        }

    def search_news(self, claim):
        """
        Mocks a news search for the claim and returns similar actual news headlines.
        """
        # For demo purposes, we return some static similar-looking "reality"
        # In production: search GDELT / NewsAPI / Google Search
        mock_headlines = [
            "Official reports confirm no evidence of widespread election fraud.",
            "Scientists reveal new study on climate change impact in the Arctic.",
            "Economic indicators show steady growth in the tech sector for Q3.",
            "Government spokesperson denies rumors of a secret moon base."
        ]
        
        # Calculate semantic similarity
        claim_embedding = self.similarity_model.encode(claim, convert_to_tensor=True)
        headline_embeddings = self.similarity_model.encode(mock_headlines, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(claim_embedding, headline_embeddings)[0]
        
        results = []
        for i in range(len(mock_headlines)):
            results.append({
                "headline": mock_headlines[i],
                "score": float(cosine_scores[i])
            })
            
        # Sort by similarity
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def verify_claim(self, claim):
        """
        External verification logic.
        """
        top_matches = self.search_news(claim)
        # If the highest match score is low, the claim might be fringe
        # If the highest match is high and the headline supports the claim, trust goes up.
        
        best_match = top_matches[0]
        credibility = 0.5 # default neutral
        
        if best_match['score'] > 0.8:
            credibility = 0.9
        elif best_match['score'] > 0.6:
            credibility = 0.7
        elif best_match['score'] < 0.3:
            credibility = 0.2 # likely unsubstantiated
            
        return {
            "claim": claim,
            "top_match": best_match['headline'],
            "match_score": best_match['score'],
            "agent_credibility": credibility
        }

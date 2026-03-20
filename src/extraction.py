

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Claim:
    claim_id: str
    source_id: str
    claim_text: str
    supporting_snippet: str
    confidence: float = 0.7
    keywords: List[str] = field(default_factory=list)
    position: int = 0


class ClaimExtractor:
    def __init__(self, max_claims_per_source: int = 15, min_claim_length: int = 20):
        self.max_claims = max_claims_per_source
        self.min_claim_length = min_claim_length
        
        self.claim_patterns = [
            r'.*(?:found|shows|reveals|indicates|suggests|demonstrates|concluded|reported|estimates|predicts|expects|believes|states|announced).*',
            r'.*\d+(?:\.\d+)?%.*(?:of|increase|decrease|growth|decline|rise|drop).*',
            r'.*(?:according to|based on|as reported by|data shows|study finds).*',
            r'.*(?:significant|substantial|notable|considerable|major|key|important|critical).*',
            r'.*(?:will|is expected to|is projected to|is likely to|may|could|should).*',
        ]
        
        self.claim_indicators = [
            'found', 'shows', 'reveals', 'indicates', 'suggests',
            'demonstrates', 'concluded', 'reported', 'estimates',
            'predicts', 'states', 'announced', 'according to',
            'significant', 'substantial', 'important', 'key',
            'increase', 'decrease', 'growth', 'decline'
        ]
        
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
            'as', 'if', 'when', 'where', 'which', 'who', 'whom', 'whose',
            'what', 'how', 'why', 'than', 'so', 'such', 'no', 'not', 'only',
            'also', 'just', 'more', 'most', 'some', 'any', 'all', 'each',
            'other', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'very', 'too', 'both', 'same', 'new'
        }
    
    def extract_claims(self, source_id: str, content: str, title: Optional[str] = None) -> List[Claim]:
        if not content or len(content.strip()) < self.min_claim_length:
            return []
        
        claims = []
        
        sentences = self._split_sentences(content)
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence)
            if score > 0:
                scored_sentences.append((sentence, i, score))
        
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        claim_counter = 0
        for sentence, position, score in scored_sentences:
            if claim_counter >= self.max_claims:
                break
            
            snippet = self._get_snippet(sentences, position, window=1)
            keywords = self._extract_keywords(sentence)
            
            claim = Claim(
                claim_id=f"{source_id}_c{claim_counter}",
                source_id=source_id,
                claim_text=sentence,
                supporting_snippet=snippet,
                confidence=min(score / 10.0, 1.0),
                keywords=keywords,
                position=position
            )
            claims.append(claim)
            claim_counter += 1
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s*', lambda m: m.group(0).replace(' ', '‡'), text)
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        sentences = [s.replace('‡', ' ').strip() for s in sentences]
        sentences = [s for s in sentences if len(s) >= self.min_claim_length]
        
        return sentences
    
    def _score_sentence(self, sentence: str) -> float:
        score = 0.0
        sentence_lower = sentence.lower()
        
        for pattern in self.claim_patterns:
            if re.match(pattern, sentence_lower, re.IGNORECASE):
                score += 2.0
                break
        
        indicator_count = sum(1 for ind in self.claim_indicators if ind in sentence_lower)
        score += indicator_count * 0.5
        
        if re.search(r'\d+(?:\.\d+)?%?', sentence):
            score += 1.5
        
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        if len(capitalized) > 1:
            score += 0.5
        
        if '?' in sentence:
            score -= 1.0
        
        if len(sentence.split()) < 5:
            score -= 0.5
        
        return max(score, 0)
    
    def _get_snippet(self, sentences: List[str], position: int, window: int = 1) -> str:
        start = max(0, position - window)
        end = min(len(sentences), position + window + 1)
        return ' '.join(sentences[start:end])
    
    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        words = [w for w in words if w not in self.stop_words]
        
        try:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [w for w, _ in sorted_words[:5]]
        except:
            keywords = list(set(words))[:5]
        
        return keywords
    
    def extract_from_sources(self, sources_content: List[dict]) -> Dict[str, List[Claim]]:
        all_claims = {}
        
        for source_data in sources_content:
            source_id = source_data.get('source_id')
            content = source_data.get('content', '')
            title = source_data.get('title')
            
            claims = self.extract_claims(source_id, content, title)
            all_claims[source_id] = claims
        
        return all_claims

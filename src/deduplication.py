

import re
import math
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ClaimGroup:
    group_id: str
    theme: str
    claims: List[dict]
    supporting_sources: List[str]
    is_conflicting: bool = False
    confidence: float = 0.7


class ClaimDeduplicator:
    def __init__(self, similarity_threshold: float = 0.65, min_shared_keywords: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_shared_keywords = min_shared_keywords
        
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'cannot', "can't", "won't", "doesn't", "isn't", "aren't"}
        
        self.uncertainty_words = {'may', 'might', 'could', 'possibly', 'perhaps', 'maybe', 'potential', 'likely', 'unlikely', 'uncertain'}
        
        self.semantic_groups = [
            {'increase', 'growth', 'rise', 'gain', 'boost', 'expand', 'higher', 'up'},
            {'decrease', 'decline', 'drop', 'fall', 'reduce', 'lower', 'down', 'shrink'},
            {'important', 'significant', 'key', 'critical', 'crucial', 'major', 'essential'},
            {'study', 'research', 'analysis', 'report', 'survey', 'investigation', 'findings'},
            {'show', 'indicate', 'reveal', 'demonstrate', 'suggest', 'find', 'report'},
            {'technology', 'tech', 'digital', 'ai', 'artificial intelligence', 'automation'},
            {'environment', 'climate', 'sustainable', 'green', 'carbon', 'emission'},
        ]
    
    def deduplicate_claims(self, claims_by_source: Dict[str, List[dict]]) -> List[ClaimGroup]:
        all_claims = []
        for source_id, claims in claims_by_source.items():
            for claim in claims:
                all_claims.append({
                    **claim,
                    'source_id': source_id
                })
        
        if not all_claims:
            return []
        
        groups = self._group_similar_claims(all_claims)
        
        processed_groups = []
        for i, group_claims in enumerate(groups):
            claim_group = self._create_claim_group(group_claims, i)
            processed_groups.append(claim_group)
        
        return processed_groups
    
    def _group_similar_claims(self, claims: List[dict]) -> List[List[dict]]:
        if not claims:
            return []
        
        groups = []
        assigned = set()
        
        for i, claim1 in enumerate(claims):
            if i in assigned:
                continue
            
            group = [claim1]
            assigned.add(i)
            
            for j, claim2 in enumerate(claims):
                if j in assigned or i == j:
                    continue
                
                similarity = self._calculate_similarity(claim1, claim2)
                
                if similarity >= self.similarity_threshold:
                    group.append(claim2)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, claim1: dict, claim2: dict) -> float:
        text1 = claim1.get('claim_text', '').lower()
        text2 = claim2.get('claim_text', '').lower()
        
        keywords1 = set(claim1.get('keywords', []))
        keywords2 = set(claim2.get('keywords', []))
        
        if not keywords1 or not keywords2:
            words1 = set(re.findall(r'\b\w{3,}\b', text1))
            words2 = set(re.findall(r'\b\w{3,}\b', text2))
            keywords1 = words1
            keywords2 = words2
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        jaccard = intersection / union if union > 0 else 0
        
        semantic_boost = 0
        for sem_group in self.semantic_groups:
            if keywords1 & sem_group and keywords2 & sem_group:
                semantic_boost += 0.1
        
        nums1 = re.findall(r'\d+(?:\.\d+)?%?', text1)
        nums2 = re.findall(r'\d+(?:\.\d+)?%?', text2)
        if nums1 and nums2 and set(nums1) & set(nums2):
            semantic_boost += 0.15
        
        similarity = jaccard + semantic_boost
        
        has_negation1 = bool(self.negation_words & set(text1.split()))
        has_negation2 = bool(self.negation_words & set(text2.split()))
        
        if has_negation1 != has_negation2:
            similarity -= 0.2
        
        return similarity
    
    def _create_claim_group(self, claims: List[dict], group_index: int) -> ClaimGroup:
        source_ids = list(set(c.get('source_id') for c in claims if c.get('source_id')))
        
        theme = self._determine_theme(claims)
        
        is_conflicting = self._check_conflicts(claims)
        
        confidences = [c.get('confidence', 0.7) for c in claims]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7
        
        return ClaimGroup(
            group_id=f"group_{group_index}",
            theme=theme,
            claims=claims,
            supporting_sources=source_ids,
            is_conflicting=is_conflicting,
            confidence=avg_confidence
        )
    
    def _determine_theme(self, claims: List[dict]) -> str:
        if len(claims) == 1:
            return claims[0].get('claim_text', '')[:200]
        
        best_claim = max(claims, key=lambda c: (c.get('confidence', 0), len(c.get('keywords', []))))
        return best_claim.get('claim_text', '')[:200]
    
    def _check_conflicts(self, claims: List[dict]) -> bool:
        if len(claims) < 2:
            return False
        
        texts = [c.get('claim_text', '').lower() for c in claims]
        
        has_increase = any(any(word in t for word in ['increase', 'growth', 'rise', 'higher', 'up', 'more', 'gain', 'boost']) for t in texts)
        has_decrease = any(any(word in t for word in ['decrease', 'decline', 'drop', 'fall', 'lower', 'down', 'less', 'reduce']) for t in texts)
        
        negations = [bool(self.negation_words & set(t.split())) for t in texts]
        
        if (has_increase and has_decrease) or (True in negations and False in negations):
            return True
        
        return False
    
    def find_near_duplicates(self, claims: List[dict], threshold: float = 0.9) -> List[Tuple[int, int]]:
        duplicates = []
        
        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims):
                if i >= j:
                    continue
                
                similarity = self._calculate_similarity(claim1, claim2)
                if similarity >= threshold:
                    duplicates.append((i, j))
        
        return duplicates

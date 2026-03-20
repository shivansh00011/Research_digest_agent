

import os
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class Claim:
    claim_id: str
    source_id: str
    claim_text: str
    supporting_snippet: str
    confidence: float = 0.7
    keywords: List[str] = field(default_factory=list)
    position: int = 0


class GeminiClaimExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_claims_per_source: int = 15,
        min_claim_length: int = 20
    ):
        self.max_claims = max_claims_per_source
        self.min_claim_length = min_claim_length
        self.model_name = model_name
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai is not installed. Run: pip install google-genai")
        
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def extract_claims(self, source_id: str, content: str, title: Optional[str] = None) -> List[Claim]:
        if not content or len(content.strip()) < self.min_claim_length:
            return []
        
        max_content_length = 30000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = self._build_extraction_prompt(content, title)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )
            
            claims_data = self._parse_llm_response(response.text)
            
            claims = []
            for i, claim_data in enumerate(claims_data[:self.max_claims]):
                claim = Claim(
                    claim_id=f"{source_id}_c{i}",
                    source_id=source_id,
                    claim_text=claim_data.get("claim_text", ""),
                    supporting_snippet=claim_data.get("supporting_snippet", ""),
                    confidence=claim_data.get("confidence", 0.8),
                    keywords=claim_data.get("keywords", []),
                    position=i
                )
                claims.append(claim)
            
            return claims
            
        except Exception as e:
            print(f"Error extracting claims with Gemini: {e}")
            return []
    
    def _build_extraction_prompt(self, content: str, title: Optional[str] = None) -> str:
        title_info = f"Title: {title}\n\n" if title else ""
        
        prompt = f"""You are a meticulous research analyst. Extract ALL factual claims from this document.

{title_info}DOCUMENT CONTENT:
{content}

INSTRUCTIONS:
1. Extract EVERY distinct factual claim, statement, or insight from the document
2. Aim for {self.max_claims} claims minimum - be thorough and comprehensive
3. Each claim should be a single complete thought or fact
4. Include claims with statistics, percentages, dates, and quantitative data
5. Include claims about trends, predictions, and findings
6. Include claims about causes, effects, and relationships
7. Preserve the original wording as much as possible

FOR EACH CLAIM:
- claim_text: The exact claim as a complete sentence (preserve original wording)
- supporting_snippet: The surrounding context or quote from the document
- confidence: How well the claim is supported (0.0-1.0)
- keywords: 3-5 relevant keywords for categorization

TYPES OF CLAIMS TO EXTRACT:
- Statistical claims (e.g., "Sales increased by 40%")
- Factual statements (e.g., "The market was valued at $370 billion")
- Research findings (e.g., "Studies show that environmental concerns drive adoption")
- Predictions (e.g., "EVs will reach price parity by 2026")
- Comparisons (e.g., "China leads with 60% of global sales")
- Causal claims (e.g., "Government policies are accelerating adoption")
- Descriptive claims (e.g., "Range anxiety is the primary concern")

Return ONLY a valid JSON array. Extract as many claims as possible:
[
  {{
    "claim_text": "The exact claim text here",
    "supporting_snippet": "Supporting context from document",
    "confidence": 0.9,
    "keywords": ["keyword1", "keyword2", "keyword3"]
  }}
]"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> List[Dict]:
        try:
            cleaned = response_text.strip()
            
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1]
            elif '```' in cleaned:
                parts = cleaned.split('```')
                if len(parts) >= 2:
                    cleaned = parts[1]
            
            if '```' in cleaned:
                cleaned = cleaned.split('```')[0]
            
            cleaned = cleaned.strip()
            
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                
                try:
                    claims_data = json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    claims_data = json.loads(json_str)
                
                valid_claims = []
                for claim in claims_data:
                    if isinstance(claim, dict) and claim.get("claim_text"):
                        valid_claims.append({
                            "claim_text": str(claim.get("claim_text", ""))[:500],
                            "supporting_snippet": str(claim.get("supporting_snippet", ""))[:500],
                            "confidence": min(1.0, max(0.0, float(claim.get("confidence", 0.7)))),
                            "keywords": claim.get("keywords", [])[:5] if isinstance(claim.get("keywords"), list) else []
                        })
                
                return valid_claims
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._regex_extract_claims(response_text)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._regex_extract_claims(response_text)
        
        return []
    
    def _regex_extract_claims(self, text: str) -> List[Dict]:
        claims = []
        
        claim_texts = re.findall(r'"claim_text"\s*:\s*"([^"]*)"', text)
        snippets = re.findall(r'"supporting_snippet"\s*:\s*"([^"]*)"', text)
        confidences = re.findall(r'"confidence"\s*:\s*([\d.]+)', text)
        
        for i, claim_text in enumerate(claim_texts[:self.max_claims]):
            claim = {
                "claim_text": claim_text,
                "supporting_snippet": snippets[i] if i < len(snippets) else "",
                "confidence": float(confidences[i]) if i < len(confidences) else 0.7,
                "keywords": []
            }
            claims.append(claim)
        
        return claims
    
    def extract_from_sources(self, sources_content: List[dict]) -> Dict[str, List[Claim]]:
        all_claims = {}
        
        for source_data in sources_content:
            source_id = source_data.get('source_id')
            if not source_id:
                continue
                
            content = source_data.get('content', '')
            title = source_data.get('title')
            
            claims = self.extract_claims(source_id, content, title)
            all_claims[source_id] = claims
        
        return all_claims


class HybridClaimExtractor:
    def __init__(
        self,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_claims_per_source: int = 15
    ):
        self.use_llm = use_llm
        self.max_claims = max_claims_per_source
        
        from .extraction import ClaimExtractor
        self.rule_extractor = ClaimExtractor(max_claims_per_source=max_claims_per_source)
        
        self.llm_extractor = None
        if use_llm:
            try:
                self.llm_extractor = GeminiClaimExtractor(
                    api_key=api_key,
                    model_name=model_name,
                    max_claims_per_source=max_claims_per_source
                )
            except Exception as e:
                print(f"Warning: Could not initialize Gemini extractor: {e}")
                print("Falling back to rule-based extraction.")
                self.llm_extractor = None
    
    def extract_claims(self, source_id: str, content: str, title: Optional[str] = None) -> List[Claim]:
        if self.llm_extractor:
            try:
                return self.llm_extractor.extract_claims(source_id, content, title)
            except Exception as e:
                print(f"LLM extraction failed, falling back to rules: {e}")
        
        return self.rule_extractor.extract_claims(source_id, content, title)



import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class DigestGenerator:
    def __init__(self):
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.source_names = {}
    
    def _build_source_names(self, sources: List[dict]):
        self.source_names = {}
        for source in sources:
            source_id = source.get('source_id', '')
            title = source.get('title', '')
            location = source.get('location', '')
            
            if title and title != 'Untitled':
                name = title
            elif location:
                name = os.path.basename(location)
            else:
                name = source_id
            
            self.source_names[source_id] = name
    
    def get_source_name(self, source_id: str) -> str:
        return self.source_names.get(source_id, source_id)
    
    def generate_digest(
        self,
        claim_groups: List,
        sources: List[dict],
        topic: str = "Research Summary"
    ) -> str:
        self._build_source_names(sources)
        
        sections = []
        sections.append(self._generate_header(topic, len(sources), len(claim_groups)))
        sections.append(self._generate_toc(claim_groups))
        sections.append(self._generate_summary(claim_groups, sources))
        sections.append(self._generate_main_content(claim_groups))
        sections.append(self._generate_source_references(sources))
        
        return '\n\n'.join(sections)
    
    def _generate_header(self, topic: str, num_sources: int, num_groups: int) -> str:
        return f"""# Research Digest: {topic}

**Generated:** {datetime.now().strftime(self.date_format)}  
**Sources Processed:** {num_sources}  
**Claim Groups Identified:** {num_groups}

---"""
    
    def _generate_toc(self, claim_groups: List) -> str:
        toc = "## Table of Contents\n"
        themes = self._categorize_groups(claim_groups)
        for i, (theme_name, groups) in enumerate(themes.items(), 1):
            toc += f"{i}. {theme_name}\n"
        toc += f"{len(themes) + 1}. Source References\n"
        return toc
    
    def _categorize_groups(self, claim_groups: List) -> Dict[str, List]:
        themes = defaultdict(list)
        for group in claim_groups:
            theme = self._determine_category(group)
            themes[theme].append(group)
        return dict(themes)
    
    def _determine_category(self, group) -> str:
        all_text = ' '.join([c.get('claim_text', '') for c in group.claims])
        all_text_lower = all_text.lower()
        
        if any(kw in all_text_lower for kw in ['market', 'sales', 'revenue', 'growth', 'industry', 'business', 'company', 'companies']):
            return "Market & Industry Trends"
        elif any(kw in all_text_lower for kw in ['technology', 'ai', 'digital', 'software', 'platform', 'system', 'tool']):
            return "Technology & Innovation"
        elif any(kw in all_text_lower for kw in ['environment', 'climate', 'carbon', 'sustainable', 'green', 'emission', 'energy']):
            return "Environmental Impact"
        elif any(kw in all_text_lower for kw in ['policy', 'regulation', 'government', 'law', 'standard', 'compliance']):
            return "Policy & Regulation"
        elif any(kw in all_text_lower for kw in ['consumer', 'user', 'customer', 'people', 'survey', 'respondent']):
            return "Consumer & User Insights"
        elif any(kw in all_text_lower for kw in ['future', 'predict', 'forecast', 'expect', 'will', 'project']):
            return "Future Outlook & Predictions"
        else:
            return "Key Findings"
    
    def _generate_summary(self, claim_groups: List, sources: List[dict]) -> str:
        summary = f"""## Executive Summary

This digest synthesizes findings from {len(sources)} sources, identifying {sum(len(g.claims) for g in claim_groups)} key claims grouped into {len(claim_groups)} thematic clusters.

### Key Highlights

"""
        
        sorted_groups = sorted(claim_groups, key=lambda g: (len(g.supporting_sources), g.confidence), reverse=True)
        
        highlights = []
        for group in sorted_groups[:5]:
            highlight_text = f"**{group.theme[:100]}{'...' if len(group.theme) > 100 else ''}**"
            if len(group.supporting_sources) > 1:
                source_names = [self.get_source_name(sid) for sid in group.supporting_sources[:3]]
                sources_str = ', '.join(source_names)
                if len(group.supporting_sources) > 3:
                    sources_str += f" and {len(group.supporting_sources) - 3} more"
                highlight_text += f" (supported by {sources_str})"
            if group.is_conflicting:
                highlight_text += " *Conflicting viewpoints*"
            highlights.append(f"- {highlight_text}")
        
        summary += '\n'.join(highlights[:5])
        
        conflicts = [g for g in claim_groups if g.is_conflicting]
        if conflicts:
            summary += f"\n\n**Note:** {len(conflicts)} claim group(s) contain conflicting viewpoints."
        
        return summary
    
    def _generate_main_content(self, claim_groups: List) -> str:
        content = "## Detailed Findings\n"
        themes = self._categorize_groups(claim_groups)
        
        for theme_name, groups in themes.items():
            content += f"\n### {theme_name}\n\n"
            for group in groups:
                content += self._format_claim_group(group)
        
        return content
    
    def _format_claim_group(self, group) -> str:
        text = f"#### {group.theme[:150]}{'...' if len(group.theme) > 150 else ''}\n\n"
        
        if group.is_conflicting:
            text += "> **Conflicting Viewpoints:** Sources disagree on this topic.\n\n"
        
        source_names = [self.get_source_name(sid) for sid in group.supporting_sources]
        sources_str = ', '.join(source_names[:3])
        if len(group.supporting_sources) > 3:
            sources_str += f" and {len(group.supporting_sources) - 3} more"
        
        text += f"**Confidence Score:** {group.confidence:.0%}  \n"
        text += f"**Supporting Sources:** {sources_str}\n\n"
        text += "**Evidence:**\n"
        
        for i, claim in enumerate(group.claims, 1):
            source_id = claim.get('source_id', 'unknown')
            source_name = self.get_source_name(source_id)
            claim_text = claim.get('claim_text', '')
            snippet = claim.get('supporting_snippet', '')
            
            text += f"\n{i}. *{claim_text[:300]}{'...' if len(claim_text) > 300 else ''}*\n"
            text += f"   - Source: {source_name}\n"
            if snippet:
                text += f"   - Context: \"{snippet[:200]}{'...' if len(snippet) > 200 else ''}\"\n"
        
        text += "\n---\n"
        return text
    
    def _generate_source_references(self, sources: List[dict]) -> str:
        references = "## Source References\n\n"
        
        for i, source in enumerate(sources, 1):
            source_id = source.get('source_id', f'source_{i}')
            source_type = source.get('source_type', 'unknown')
            location = source.get('location', 'Unknown location')
            title = source.get('title', 'Untitled')
            length = source.get('length', 0)
            error = source.get('error')
            
            display_name = title if title and title != 'Untitled' else os.path.basename(location) if location else source_id
            
            if error:
                references += f"### {display_name} \n"
                references += f"- **Error:** {error}\n"
                references += f"- **Location:** {location}\n\n"
            else:
                references += f"### {display_name}\n"
                references += f"- **Type:** {source_type}\n"
                references += f"- **Location:** {location}\n"
                references += f"- **Content Length:** {length:,} characters\n\n"
        
        return references
    
    def generate_sources_json(
        self,
        claim_groups: List,
        sources: List[dict],
        output: Optional[dict] = None
    ) -> dict:
        self._build_source_names(sources)
        
        if output is None:
            output = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_sources": len(sources),
                    "total_claim_groups": len(claim_groups),
                    "agent_version": "1.0"
                },
                "sources": {},
                "claims": [],
                "claim_groups": []
            }
        
        for source in sources:
            source_id = source.get('source_id')
            if source_id:
                output['sources'][source_id] = {
                    "source_id": source_id,
                    "source_name": self.get_source_name(source_id),
                    "source_type": source.get('source_type'),
                    "location": source.get('location'),
                    "title": source.get('title'),
                    "length": source.get('length'),
                    "error": source.get('error')
                }
        
        all_claims = []
        for group in claim_groups:
            for claim in group.claims:
                source_id = claim.get('source_id')
                claim_entry = {
                    "claim_id": claim.get('claim_id'),
                    "source_id": source_id,
                    "source_name": self.get_source_name(source_id) if source_id else "Unknown",
                    "claim_text": claim.get('claim_text'),
                    "supporting_snippet": claim.get('supporting_snippet'),
                    "confidence": claim.get('confidence', 0.7),
                    "keywords": claim.get('keywords', []),
                    "group_id": group.group_id
                }
                all_claims.append(claim_entry)
        
        output['claims'] = all_claims
        
        groups_data = []
        for group in claim_groups:
            group_entry = {
                "group_id": group.group_id,
                "theme": group.theme,
                "confidence": group.confidence,
                "is_conflicting": group.is_conflicting,
                "supporting_sources": [
                    {"source_id": sid, "source_name": self.get_source_name(sid)}
                    for sid in group.supporting_sources
                ],
                "claim_count": len(group.claims)
            }
            groups_data.append(group_entry)
        
        output['claim_groups'] = groups_data
        return output
    
    def save_digest(self, digest: str, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(digest)
    
    def save_json(self, data: dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

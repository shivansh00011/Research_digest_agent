

import os
import json
import argparse
from typing import List, Dict, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from .ingestion import ContentIngester, SourceContent
from .extraction import ClaimExtractor, Claim
from .llm_extraction import GeminiClaimExtractor, HybridClaimExtractor
from .deduplication import ClaimDeduplicator, ClaimGroup
from .generation import DigestGenerator


class AgentState(TypedDict):
    sources: List[str]
    topic: str
    verbose: bool
    use_llm: bool
    source_contents: List[dict]
    valid_sources: List[dict]
    failed_sources: List[dict]
    duplicate_hashes: Dict[str, str]
    claims_by_source: Dict[str, List[dict]]
    claim_groups: List[dict]
    digest_markdown: str
    sources_json: dict
    results: dict
    errors: List[str]
    metadata: dict


def ingest_node(state: AgentState) -> AgentState:
    sources = state['sources']
    verbose = state.get('verbose', False)
    
    if verbose:
        print("Step 1: Ingesting content...")
    
    ingester = ContentIngester()
    source_contents = ingester.ingest_sources(sources)
    ingester.close()
    
    valid_sources = []
    failed_sources = []
    
    for s in source_contents:
        source_dict = {
            'source_id': s.source_id,
            'source_type': s.source_type,
            'location': s.location,
            'title': s.title,
            'content': s.content,
            'content_hash': s.content_hash,
            'length': s.length,
            'error': s.error,
            'metadata': s.metadata
        }
        
        if s.error:
            failed_sources.append(source_dict)
        else:
            valid_sources.append(source_dict)
    
    if verbose:
        print(f"  - Successfully ingested: {len(valid_sources)}")
        print(f"  - Failed: {len(failed_sources)}")
        for s in failed_sources:
            print(f"    * {s['location']}: {s['error']}")
    
    return {
        **state,
        'source_contents': valid_sources + failed_sources,
        'valid_sources': valid_sources,
        'failed_sources': failed_sources,
        'errors': state.get('errors', []) + [f"{s['location']}: {s['error']}" for s in failed_sources]
    }


def deduplicate_content_node(state: AgentState) -> AgentState:
    valid_sources = state['valid_sources']
    verbose = state.get('verbose', False)
    
    if verbose:
        print("\nStep 1b: Checking for duplicate content...")
    
    content_hashes = {}
    unique_sources = []
    duplicates_found = 0
    
    for source in valid_sources:
        content_hash = source.get('content_hash', '')
        if content_hash in content_hashes:
            if verbose:
                print(f"  - Skipping duplicate content: {source['location']}")
            duplicates_found += 1
        else:
            content_hashes[content_hash] = source['source_id']
            unique_sources.append(source)
    
    if verbose and duplicates_found > 0:
        print(f"  - Found {duplicates_found} duplicate(s)")
    
    return {
        **state,
        'valid_sources': unique_sources,
        'duplicate_hashes': content_hashes,
        'metadata': {**state.get('metadata', {}), 'duplicates_removed': duplicates_found}
    }


def extract_claims_node(state: AgentState) -> AgentState:
    valid_sources = state['valid_sources']
    use_llm = state.get('use_llm', False)
    max_claims = state.get('metadata', {}).get('max_claims_per_source', 15)
    verbose = state.get('verbose', False)
    
    if verbose:
        extraction_method = "Gemini LLM" if use_llm else "rule-based heuristics"
        print(f"\nStep 2: Extracting claims (using {extraction_method})...")
    
    if use_llm:
        try:
            extractor = HybridClaimExtractor(
                use_llm=True,
                max_claims_per_source=max_claims
            )
        except Exception as e:
            if verbose:
                print(f"  - LLM initialization failed, falling back to rules: {e}")
            extractor = ClaimExtractor(max_claims_per_source=max_claims)
    else:
        extractor = ClaimExtractor(max_claims_per_source=max_claims)
    
    claims_by_source = {}
    total_claims = 0
    
    for source in valid_sources:
        source_id = source.get('source_id')
        if not source_id:
            continue
            
        claims = extractor.extract_claims(
            source_id,
            source.get('content', ''),
            source.get('title')
        )
        
        claims_data = [
            {
                'claim_id': c.claim_id,
                'source_id': c.source_id,
                'claim_text': c.claim_text,
                'supporting_snippet': c.supporting_snippet,
                'confidence': c.confidence,
                'keywords': c.keywords,
                'position': c.position
            }
            for c in claims
        ]
        
        claims_by_source[source_id] = claims_data
        total_claims += len(claims)
        
        if verbose:
            print(f"  - {source_id}: {len(claims)} claims extracted")
    
    return {
        **state,
        'claims_by_source': claims_by_source,
        'metadata': {**state.get('metadata', {}), 'total_claims': total_claims}
    }


def deduplicate_claims_node(state: AgentState) -> AgentState:
    claims_by_source = state['claims_by_source']
    similarity_threshold = state.get('metadata', {}).get('similarity_threshold', 0.65)
    verbose = state.get('verbose', False)
    
    if verbose:
        print("\nStep 3: Deduplicating and grouping claims...")
    
    deduplicator = ClaimDeduplicator(similarity_threshold=similarity_threshold)
    claim_groups = deduplicator.deduplicate_claims(claims_by_source)
    
    groups_data = []
    for group in claim_groups:
        group_dict = {
            'group_id': group.group_id,
            'theme': group.theme,
            'claims': group.claims,
            'supporting_sources': group.supporting_sources,
            'is_conflicting': group.is_conflicting,
            'confidence': group.confidence
        }
        groups_data.append(group_dict)
    
    if verbose:
        print(f"  - Created {len(groups_data)} claim groups")
        conflicts = sum(1 for g in groups_data if g.get('is_conflicting'))
        if conflicts > 0:
            print(f"  - Found {conflicts} groups with conflicting viewpoints")
    
    return {
        **state,
        'claim_groups': groups_data,
        'metadata': {**state.get('metadata', {}), 'claim_groups': len(groups_data)}
    }


def generate_digest_node(state: AgentState) -> AgentState:
    claim_groups_data = state['claim_groups']
    source_contents = state['source_contents']
    topic = state['topic']
    output_dir = state.get('metadata', {}).get('output_dir', 'output')
    verbose = state.get('verbose', False)
    
    if verbose:
        print("\nStep 4: Generating digest outputs...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = DigestGenerator()
    
    claim_groups = []
    for g in claim_groups_data:
        cg = ClaimGroup(
            group_id=g['group_id'],
            theme=g['theme'],
            claims=g['claims'],
            supporting_sources=g['supporting_sources'],
            is_conflicting=g['is_conflicting'],
            confidence=g['confidence']
        )
        claim_groups.append(cg)
    
    sources_metadata = state['source_contents']
    
    digest_md = generator.generate_digest(claim_groups, sources_metadata, topic)
    sources_json = generator.generate_sources_json(claim_groups, sources_metadata)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"digest_{timestamp}.md")
    json_path = os.path.join(output_dir, f"sources_{timestamp}.json")
    
    generator.save_digest(digest_md, md_path)
    generator.save_json(sources_json, json_path)
    
    if verbose:
        print(f"  - Saved digest: {md_path}")
        print(f"  - Saved JSON: {json_path}")
    
    return {
        **state,
        'digest_markdown': digest_md,
        'sources_json': sources_json,
        'metadata': {
            **state.get('metadata', {}),
            'digest_path': md_path,
            'json_path': json_path
        }
    }


def finalize_node(state: AgentState) -> AgentState:
    valid_sources = state['valid_sources']
    failed_sources = state['failed_sources']
    claim_groups = state['claim_groups']
    metadata = state.get('metadata', {})
    errors = state.get('errors', [])
    
    total_claims = sum(len(claims) for claims in state.get('claims_by_source', {}).values())
    
    results = {
        'success': len(valid_sources) > 0,
        'sources_processed': len(valid_sources),
        'sources_failed': len(failed_sources),
        'claims_extracted': total_claims,
        'claim_groups': len(claim_groups),
        'digest_path': metadata.get('digest_path'),
        'json_path': metadata.get('json_path'),
        'errors': errors
    }
    
    return {
        **state,
        'results': results
    }


def check_sources(state: AgentState) -> str:
    valid_sources = state.get('valid_sources', [])
    
    if not valid_sources:
        return "finalize"
    
    return "deduplicate_content"


def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("deduplicate_content", deduplicate_content_node)
    workflow.add_node("extract_claims", extract_claims_node)
    workflow.add_node("deduplicate_claims", deduplicate_claims_node)
    workflow.add_node("generate_digest", generate_digest_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.set_entry_point("ingest")
    
    workflow.add_conditional_edges(
        "ingest",
        check_sources,
        {
            "deduplicate_content": "deduplicate_content",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("deduplicate_content", "extract_claims")
    workflow.add_edge("extract_claims", "deduplicate_claims")
    workflow.add_edge("deduplicate_claims", "generate_digest")
    workflow.add_edge("generate_digest", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


class ResearchDigestAgent:
    def __init__(
        self,
        max_claims_per_source: int = 15,
        similarity_threshold: float = 0.65,
        output_dir: str = "output",
        use_llm: bool = False
    ):
        self.max_claims = max_claims_per_source
        self.similarity_threshold = similarity_threshold
        self.output_dir = output_dir
        self.use_llm = use_llm
        self.graph = build_graph()
    
    def process_sources(
        self,
        sources: List[str],
        topic: str = "Research Summary",
        verbose: bool = False
    ) -> Dict:
        if verbose:
            print(f"Starting research digest for topic: {topic}")
            print(f"Processing {len(sources)} sources...")
            if self.use_llm:
                print("Using Gemini LLM for claim extraction")
            print()
        
        initial_state = {
            'sources': sources,
            'topic': topic,
            'verbose': verbose,
            'use_llm': self.use_llm,
            'source_contents': [],
            'valid_sources': [],
            'failed_sources': [],
            'duplicate_hashes': {},
            'claims_by_source': {},
            'claim_groups': [],
            'digest_markdown': '',
            'sources_json': {},
            'results': {},
            'errors': [],
            'metadata': {
                'max_claims_per_source': self.max_claims,
                'similarity_threshold': self.similarity_threshold,
                'output_dir': self.output_dir
            }
        }
        
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print(f"\nDone! Processed {final_state['results']['sources_processed']} sources.")
            print(f"Extracted {final_state['results']['claims_extracted']} claims in {final_state['results']['claim_groups']} groups.")
        
        return final_state['results']


def main():
    parser = argparse.ArgumentParser(description='Research Digest Agent')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--urls', nargs='+', help='List of URLs to process')
    group.add_argument('--file-list', help='Path to file containing URLs')
    group.add_argument('--folder', help='Path to folder containing files')
    
    parser.add_argument('--topic', default='Research Summary', help='Topic title')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--max-claims', type=int, default=15, help='Max claims per source')
    parser.add_argument('--similarity', type=float, default=0.65, help='Similarity threshold')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument('--use-llm', action='store_true', help='Use Gemini LLM')
    
    args = parser.parse_args()
    
    sources = []
    
    if args.urls:
        sources = args.urls
    elif args.file_list:
        with open(args.file_list, 'r') as f:
            sources = [line.strip() for line in f if line.strip()]
    elif args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith(('.txt', '.html', '.htm', '.md')):
                sources.append(os.path.join(args.folder, filename))
    
    if not sources:
        print("Error: No sources found")
        return 1
    
    agent = ResearchDigestAgent(
        max_claims_per_source=args.max_claims,
        similarity_threshold=args.similarity,
        output_dir=args.output_dir,
        use_llm=args.use_llm
    )
    
    results = agent.process_sources(
        sources=sources,
        topic=args.topic,
        verbose=args.verbose
    )
    
    if not results['success']:
        print(f"Error: {results['errors']}")
        return 1
    
    if not args.verbose:
        print(f"Digest generated: {results['digest_path']}")
        print(f"JSON output: {results['json_path']}")
    
    return 0


if __name__ == '__main__':
    exit(main())

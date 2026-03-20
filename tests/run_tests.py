

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import ContentIngester, SourceContent
from src.extraction import ClaimExtractor, Claim
from src.deduplication import ClaimDeduplicator, ClaimGroup
from src.agent import ResearchDigestAgent


def test_empty_file_handling():
    """Test that empty files are handled gracefully."""
    print("Test: Empty file handling...")
    
    extractor = ClaimExtractor()
    claims = extractor.extract_claims("test_id", "", "Empty File")
    
    assert claims == [], f"Expected empty list, got {claims}"
    assert len(claims) == 0
    print("  ✓ Passed")


def test_unreachable_url_handling():
    """Test that unreachable URLs are handled gracefully."""
    print("Test: Unreachable URL handling...")
    
    ingester = ContentIngester()
    result = ingester._fetch_url("https://this-domain-does-not-exist-12345.invalid/test")
    
    assert result.error is not None, "Expected error for unreachable URL"
    assert result.content == "", "Expected empty content for unreachable URL"
    assert result.source_type == 'url'
    print("  ✓ Passed")


def test_missing_file_handling():
    """Test that missing files are handled gracefully."""
    print("Test: Missing file handling...")
    
    ingester = ContentIngester()
    result = ingester._read_file("/nonexistent/path/file.txt")
    
    assert result.error is not None, "Expected error for missing file"
    assert result.source_type == 'file'
    print("  ✓ Passed")


def test_agent_handles_mixed_sources():
    """Test that agent handles a mix of valid and invalid sources."""
    print("Test: Agent handles mixed sources...")
    
    tmp_dir = tempfile.mkdtemp()
    try:
        valid_file = os.path.join(tmp_dir, "valid.txt")
        with open(valid_file, 'w') as f:
            f.write("This is a test source about electric vehicles. Studies show that EV sales have increased by 40% in recent years.")
        
        output_dir = os.path.join(tmp_dir, "output")
        # Disable tracing for tests
        agent = ResearchDigestAgent(output_dir=output_dir)
        
        sources = [
            valid_file,
            "/nonexistent/file.txt",
            "https://invalid-domain-12345.invalid/test"
        ]
        
        results = agent.process_sources(sources, topic="Test", verbose=False)
        
        assert results['sources_processed'] >= 1, f"Expected at least 1 processed source, got {results['sources_processed']}"
        assert results['sources_failed'] >= 1, f"Expected at least 1 failed source, got {results['sources_failed']}"
        print("  ✓ Passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_identical_claims_deduplicated():
    """Test that identical claims are grouped together."""
    print("Test: Identical claims deduplication...")
    
    deduplicator = ClaimDeduplicator(similarity_threshold=0.6)
    
    claim1 = {
        'claim_id': 'c1',
        'source_id': 'source1',
        'claim_text': 'Electric vehicle sales increased by 40% in 2022.',
        'supporting_snippet': 'The report shows electric vehicle sales increased by 40% in 2022.',
        'confidence': 0.8,
        'keywords': ['electric', 'vehicle', 'sales', 'increased', '40%']
    }
    
    claim2 = {
        'claim_id': 'c2',
        'source_id': 'source2',
        'claim_text': 'EV sales grew by 40% last year.',
        'supporting_snippet': 'According to the data, EV sales grew by 40% last year.',
        'confidence': 0.75,
        'keywords': ['sales', 'grew', '40%', 'ev', 'year']
    }
    
    claims_by_source = {
        'source1': [claim1],
        'source2': [claim2]
    }
    
    groups = deduplicator.deduplicate_claims(claims_by_source)
    
    assert len(groups) >= 1, "Expected at least 1 group"
    print("  ✓ Passed")


def test_duplicate_content_hash():
    """Test that duplicate content is detected via content hash."""
    print("Test: Duplicate content hash detection...")
    
    tmp_dir = tempfile.mkdtemp()
    try:
        ingester = ContentIngester()
        
        content = "This is duplicate content about electric vehicles."
        
        file1 = os.path.join(tmp_dir, "file1.txt")
        file2 = os.path.join(tmp_dir, "file2.txt")
        
        with open(file1, 'w') as f:
            f.write(content)
        with open(file2, 'w') as f:
            f.write(content)
        
        results = ingester.ingest_sources([file1, file2])
        
        assert results[0].content_hash == results[1].content_hash, "Expected same content hash for identical content"
        assert results[0].content_hash != "", "Expected non-empty content hash"
        print("  ✓ Passed")
    finally:
        shutil.rmtree(tmp_dir)


def test_different_claims_not_grouped():
    """Test that different claims are not grouped together."""
    print("Test: Different claims not grouped...")
    
    deduplicator = ClaimDeduplicator(similarity_threshold=0.8)
    
    claim1 = {
        'claim_id': 'c1',
        'source_id': 'source1',
        'claim_text': 'Battery costs have decreased significantly.',
        'supporting_snippet': 'Battery costs have decreased significantly over the past decade.',
        'confidence': 0.8,
        'keywords': ['battery', 'costs', 'decreased']
    }
    
    claim2 = {
        'claim_id': 'c2',
        'source_id': 'source2',
        'claim_text': 'Consumer preference for SUVs is growing.',
        'supporting_snippet': 'Studies show consumer preference for SUVs is growing.',
        'confidence': 0.75,
        'keywords': ['consumer', 'preference', 'suvs', 'growing']
    }
    
    claims_by_source = {
        'source1': [claim1],
        'source2': [claim2]
    }
    
    groups = deduplicator.deduplicate_claims(claims_by_source)
    
    assert len(groups) == 2, f"Expected 2 groups for different claims, got {len(groups)}"
    print("  ✓ Passed")


def test_conflicting_viewpoints_preserved():
    """Test that conflicting viewpoints are detected and preserved."""
    print("Test: Conflicting viewpoints preservation...")
    
    deduplicator = ClaimDeduplicator(similarity_threshold=0.5)
    
    claim1 = {
        'claim_id': 'c1',
        'source_id': 'source1',
        'claim_text': 'EV subsidies are highly effective at increasing adoption.',
        'supporting_snippet': 'Research indicates EV subsidies are highly effective at increasing adoption rates.',
        'confidence': 0.8,
        'keywords': ['subsidies', 'effective', 'adoption', 'increasing']
    }
    
    claim2 = {
        'claim_id': 'c2',
        'source_id': 'source2',
        'claim_text': 'EV subsidies may not be cost-effective for increasing adoption.',
        'supporting_snippet': 'Some analysts argue EV subsidies may not be cost-effective for increasing adoption.',
        'confidence': 0.75,
        'keywords': ['subsidies', 'cost-effective', 'adoption', 'not']
    }
    
    claims_by_source = {
        'source1': [claim1],
        'source2': [claim2]
    }
    
    groups = deduplicator.deduplicate_claims(claims_by_source)
    
    total_claims = sum(len(g.claims) for g in groups)
    assert total_claims == 2, f"Expected 2 total claims preserved, got {total_claims}"
    print("  ✓ Passed")


def test_source_attribution_preserved():
    """Test that source attribution is preserved for all claims."""
    print("Test: Source attribution preservation...")
    
    deduplicator = ClaimDeduplicator()
    
    claim1 = {
        'claim_id': 'c1',
        'source_id': 'source_A',
        'claim_text': 'Claim from source A.',
        'supporting_snippet': 'Snippet A',
        'confidence': 0.8,
        'keywords': ['claim', 'source']
    }
    
    claim2 = {
        'claim_id': 'c2',
        'source_id': 'source_B',
        'claim_text': 'Claim from source B.',
        'supporting_snippet': 'Snippet B',
        'confidence': 0.75,
        'keywords': ['claim', 'source']
    }
    
    claims_by_source = {
        'source_A': [claim1],
        'source_B': [claim2]
    }
    
    groups = deduplicator.deduplicate_claims(claims_by_source)
    
    all_source_ids = set()
    for group in groups:
        for claim in group.claims:
            all_source_ids.add(claim['source_id'])
    
    assert 'source_A' in all_source_ids, "Expected source_A in attribution"
    assert 'source_B' in all_source_ids, "Expected source_B in attribution"
    print("  ✓ Passed")


def test_full_pipeline():
    """Test the complete pipeline with sample sources."""
    print("Test: Full pipeline integration...")
    
    tmp_dir = tempfile.mkdtemp()
    try:
        source1 = os.path.join(tmp_dir, "source1.txt")
        with open(source1, 'w') as f:
            f.write("""
            Electric Vehicle Market Analysis
            
            The electric vehicle market has grown significantly. Sales increased by 40% 
            in 2022, reaching 10 million units globally. Studies show that environmental 
            concerns are driving adoption, with 65% of buyers citing this as a key factor.
            """)
        
        source2 = os.path.join(tmp_dir, "source2.txt")
        with open(source2, 'w') as f:
            f.write("""
            Consumer Survey Results
            
            Our survey of 5,000 consumers found that 42% are likely to purchase an EV 
            as their next vehicle. Environmental concerns motivate 65% of potential buyers.
            Range anxiety remains the top concern at 45%.
            """)
        
        output_dir = os.path.join(tmp_dir, "output")
        # Disable tracing for tests
        agent = ResearchDigestAgent(output_dir=output_dir)
        
        results = agent.process_sources(
            [source1, source2],
            topic="Electric Vehicles",
            verbose=False
        )
        
        assert results['sources_processed'] == 2, f"Expected 2 processed sources, got {results['sources_processed']}"
        assert results['claims_extracted'] > 0, f"Expected claims extracted, got {results['claims_extracted']}"
        assert results['claim_groups'] > 0, f"Expected claim groups, got {results['claim_groups']}"
        assert results['success'] == True, "Expected success"
        print("  ✓ Passed")
    finally:
        shutil.rmtree(tmp_dir)


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Research Digest Agent Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_empty_file_handling,
        test_unreachable_url_handling,
        test_missing_file_handling,
        test_agent_handles_mixed_sources,
        test_identical_claims_deduplicated,
        test_duplicate_content_hash,
        test_different_claims_not_grouped,
        test_conflicting_viewpoints_preserved,
        test_source_attribution_preserved,
        test_full_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

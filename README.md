# Research Digest Agent

An autonomous research digest agent built with LangGraph that ingests multiple sources, extracts key information using Gemini LLM, removes redundancy, and produces structured, evidence-backed briefs.

## Features

- **LangGraph Pipeline**: State machine orchestration with traceable nodes
- **Gemini LLM Integration**: Intelligent claim extraction with fallback to rule-based
- **Streamlit UI**: Web interface for easy interaction
- **Multiple Input Types**: URLs, local files, or folder processing

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### CLI Usage

```bash
# Rule-based extraction
python -m src.agent --folder ./sample_sources --topic "Electric Vehicles"

# With Gemini LLM (set API key first)
export GOOGLE_API_KEY="your-api-key"
python -m src.agent --folder ./sample_sources --topic "Electric Vehicles" --use-llm --verbose
```

### Streamlit App

```bash
streamlit run app.py
```

### Running Tests

```bash
python tests/run_tests.py
```

## Project Structure

```
research_digest/
├── README.md
├── requirements.txt
├── app.py                    # Streamlit entry point
├── src/
│   ├── agent.py              # LangGraph orchestration
│   ├── ingestion.py          # Content fetching
│   ├── extraction.py         # Rule-based claim extraction
│   ├── llm_extraction.py     # Gemini LLM extraction
│   ├── deduplication.py      # Claim grouping
│   └── generation.py         # Output generation
├── app/
│   └── streamlit_app.py      # Web UI
├── tests/
│   └── run_tests.py          # Test suite (10 tests)
└── sample_sources/           # 6 example input files
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_claims_per_source` | 15 | Max claims to extract per source |
| `similarity_threshold` | 0.65 | Threshold for grouping similar claims |
| `use_llm` | False | Use Gemini for extraction |
| `model_name` | gemini-2.5-flash | Gemini model to use |

## API Keys

### Gemini (for LLM extraction)
```bash
export GOOGLE_API_KEY="your-key"
```
Get from: https://aistudio.google.com/app/apikey

## How the Agent Processes Sources (Step by Step)

### Step 1: Content Ingestion
The `ingest_node` in LangGraph handles content ingestion:
1. **Input Detection**: Determines if each source is a URL or file path
2. **Content Fetching**: 
   - URLs: HTTP requests with timeout handling, HTML parsing
   - Files: Direct read with encoding detection
3. **Text Cleaning**: Removes excessive whitespace, control characters
4. **Metadata Storage**: Captures source_id, title, content_hash, length
5. **Error Handling**: Records failures for unreachable URLs, missing files

### Step 2: Claim Extraction
The `extract_claims_node` processes each valid source:

**LLM Mode (Gemini)**:
1. Builds extraction prompt with document content
2. Sends to Gemini API with temperature=0.1 for consistency
3. Parses JSON response with claim_text, supporting_snippet, confidence, keywords
4. Validates claims are grounded in source text
5. Falls back to regex extraction if JSON parsing fails

**Rule-based Mode**:
1. Splits content into sentences
2. Scores each sentence based on claim indicators (found, shows, reveals, etc.)
3. Extracts top-scoring sentences as claims
4. Captures surrounding context as supporting snippet
5. Calculates confidence from sentence score

### Step 3: Deduplication & Grouping
The `deduplicate_claims_node` groups similar claims:
1. **Similarity Calculation**:
   - Jaccard similarity on keyword sets
   - Semantic boosting for related words (increase/growth/rise)
   - Numerical matching (same statistics boost similarity)
   - Negation penalty (opposite meanings reduce similarity)
2. **Grouping Algorithm**: Greedy first-match approach
3. **Conflict Detection**: Identifies opposing sentiment (increase vs. decrease)
4. **Source Tracking**: Maintains list of supporting sources per group

### Step 4: Output Generation
The `generate_digest_node` creates outputs:
1. Builds source name mapping (title or filename)
2. Categorizes claim groups into themes
3. Generates `digest.md` with:
   - Executive summary with top findings
   - Detailed findings by theme
   - Source references with full names
4. Generates `sources.json` with:
   - All claims with source names
   - Claim groups with source attribution
   - Complete metadata

## How Claims Are Grounded

Every claim includes:
- **claim_text**: Exact text from source (or faithful LLM summary)
- **supporting_snippet**: Surrounding context (±1 sentence window)
- **source_name**: Human-readable source identifier
- **confidence**: Score indicating how well-supported the claim is

Claims are never invented:
- LLM is prompted to extract only from document
- Rule-based extracts verbatim sentences
- Empty/unclear content is safely skipped

## How Deduplication Works

1. **Keyword Extraction**: Each claim gets 3-5 keywords
2. **Similarity Score**: 
   - Jaccard overlap on keyword sets (0-1)
   - Semantic boost (+0.1) for related word groups
   - Numerical match (+0.15) for same statistics
   - Negation penalty (-0.2) for opposing sentiment
3. **Grouping**: Claims with similarity ≥ threshold (default 0.65) are grouped
4. **Conflict Detection**: Checks for opposing words (increase/decrease) or mixed negations

## One Limitation

**Rule-based extraction lacks semantic understanding**: Without LLM, the agent relies on keyword patterns and indicators. This means:
- Complex sentence structures may be missed
- Context-dependent claims might not be extracted
- Nuanced insights requiring inference are not captured

## One Improvement With More Time

**Integrate LangChain's Gemini wrapper for full LLM tracing**: Currently, LangGraph sees node execution but not the actual LLM prompts/responses. Using `ChatGoogleGenerativeAI` from LangChain would enable:
- Full prompt/response tracing in one view
- Token usage tracking per claim
- Better debugging of extraction failures
- Cost analysis for LLM usage

## Tests

The test suite includes 10 tests covering:

1. **Empty/unreachable source handling**:
   - Empty file handling
   - Unreachable URL handling
   - Missing file handling
   - Mixed valid/invalid sources

2. **Deduplication of duplicate content**:
   - Identical claims grouped
   - Duplicate content hash detection
   - Different claims not grouped

3. **Preservation of conflicting claims**:
   - Conflicting viewpoints detected
   - Source attribution preserved
   - Full pipeline integration

Run with: `python tests/run_tests.py`



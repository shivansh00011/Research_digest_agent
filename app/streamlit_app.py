

import os
import sys
import json
import tempfile
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ResearchDigestAgent


def main():
    st.set_page_config(
        page_title="Research Digest Agent",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .llm-badge {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Research Digest Agent</div>', unsafe_allow_html=True)
    st.markdown("**An autonomous agent that ingests multiple sources, extracts key claims, and produces structured briefs.**")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Extraction Method")
        use_llm = st.checkbox(
            "Use Gemini LLM",
            value=False,
            help="Use Google Gemini for better claim extraction"
        )
        
        if use_llm:
            existing_key = os.environ.get("GOOGLE_API_KEY", "")
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=existing_key,
                help="Get your API key from https://aistudio.google.com/app/apikey"
            )
            if gemini_api_key:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
            
            
        else:
            st.info("Using rule-based extraction")
        
        st.markdown("---")
        
        topic = st.text_input(
            "Topic Title",
            value="Research Summary",
            help="Title for the generated digest"
        )
        
        max_claims = st.slider(
            "Max Claims per Source",
            min_value=5,
            max_value=25,
            value=15,
            help="Maximum claims to extract from each source"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.65,
            step=0.05,
            help="Threshold for grouping similar claims"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This agent:
        1. Ingests content
        2. Extracts claims
        3. Deduplicates claims
        4. Generates structured outputs
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Sources")
        
        tab1, tab2, tab3 = st.tabs(["URLs", "Upload Files", "File List"])
        
        urls_input = ""
        uploaded_files = None
        file_list_text = ""
        
        with tab1:
            urls_input = st.text_area(
                "Enter URLs (one per line)",
                height=200,
                placeholder="https://example.com/article1\nhttps://example.com/article2\n..."
            )
        
        with tab2:
            uploaded_files = st.file_uploader(
                "Upload text or HTML files",
                type=['txt', 'html', 'htm', 'md'],
                accept_multiple_files=True
            )
        
        with tab3:
            file_list_text = st.text_area(
                "Enter file paths (one per line)",
                height=200,
                placeholder="/path/to/file1.txt\n/path/to/file2.html\n..."
            )
        
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            process_btn = st.button("Process Sources", type="primary", use_container_width=True)
    
    with col2:
        st.header("Status")
        status_placeholder = st.empty()
        
        with st.expander("Current Configuration", expanded=False):
            config = {
                "topic": topic,
                "max_claims_per_source": max_claims,
                "similarity_threshold": similarity_threshold,
                "extraction_method": "Gemini LLM" if use_llm else "Rule-based"
            }
            st.json(config)
    
    results_container = st.container()
    
    if process_btn:
        sources = []
        temp_dir = None
        
        if urls_input.strip():
            urls = [u.strip() for u in urls_input.strip().split('\n') if u.strip()]
            sources.extend(urls)
        
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                sources.append(file_path)
        
        if file_list_text.strip():
            file_paths = [p.strip() for p in file_list_text.strip().split('\n') if p.strip()]
            sources.extend(file_paths)
        
        if not sources:
            st.error("Please provide at least one source")
            return
        
        if use_llm and not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please enter your Gemini API key in the sidebar")
            return
        
        output_dir = tempfile.mkdtemp()
        
        agent = ResearchDigestAgent(
            max_claims_per_source=max_claims,
            similarity_threshold=similarity_threshold,
            output_dir=output_dir,
            use_llm=use_llm
        )
        
        progress_bar = st.progress(0)
        status_text = status_placeholder.empty()
        
        try:
            status_text.info("Step 1/4: Ingesting content...")
            progress_bar.progress(25)
            
            results = agent.process_sources(
                sources=sources,
                topic=topic,
                verbose=False
            )
            
            progress_bar.progress(100)
            status_text.success("Processing complete!")
            
            with results_container:
                st.markdown("---")
                st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Sources Processed", results['sources_processed'])
                with metric_col2:
                    st.metric("Sources Failed", results['sources_failed'])
                with metric_col3:
                    st.metric("Claims Extracted", results['claims_extracted'])
                with metric_col4:
                    st.metric("Claim Groups", results['claim_groups'])
                
                
            
                if results['errors']:
                    with st.expander("Errors & Warnings", expanded=False):
                        for error in results['errors']:
                            st.warning(error)
                
                if results.get('digest_path') and os.path.exists(results['digest_path']):
                    with open(results['digest_path'], 'r') as f:
                        digest_content = f.read()
                    
                    result_tab1, result_tab2, result_tab3 = st.tabs(["Digest", "JSON Data", "Downloads"])
                    
                    with result_tab1:
                        st.markdown(digest_content)
                    
                    with result_tab2:
                        if results.get('json_path') and os.path.exists(results['json_path']):
                            with open(results['json_path'], 'r') as f:
                                json_data = json.load(f)
                            st.json(json_data)
                    
                    with result_tab3:
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "Download Digest (MD)",
                                digest_content,
                                file_name=f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        with col_dl2:
                            if results.get('json_path'):
                                with open(results['json_path'], 'r') as f:
                                    json_content = f.read()
                                st.download_button(
                                    "Download JSON",
                                    json_content,
                                    file_name=f"sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            progress_bar.empty()
            st.exception(e)
        
        finally:
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()

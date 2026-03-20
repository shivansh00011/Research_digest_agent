

import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
from bs4 import BeautifulSoup


@dataclass
class SourceContent:
    source_id: str
    source_type: str
    location: str
    title: Optional[str] = None
    content: str = ""
    content_hash: str = ""
    length: int = 0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class ContentIngester:
    def __init__(self, timeout: int = 30, max_content_length: int = 500000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchDigestAgent/1.0)'
        })
    
    def ingest_sources(self, sources: List[str]) -> List[SourceContent]:
        results = []
        for source in sources:
            source = source.strip()
            if not source:
                continue
                
            if source.startswith(('http://', 'https://')):
                content = self._fetch_url(source)
            elif os.path.exists(source):
                content = self._read_file(source)
            else:
                content = SourceContent(
                    source_id=self._generate_id(source),
                    source_type='unknown',
                    location=source,
                    error=f"Source not found or invalid: {source}"
                )
            results.append(content)
        
        return results
    
    def _fetch_url(self, url: str) -> SourceContent:
        source_id = self._generate_id(url)
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            if 'text/html' in content_type:
                content, title = self._parse_html(response.text)
            else:
                content = response.text
                title = None
            
            content = self._clean_text(content)
            content = content[:self.max_content_length]
            
            return SourceContent(
                source_id=source_id,
                source_type='url',
                location=url,
                title=title,
                content=content,
                content_hash=self._hash_content(content),
                length=len(content),
                metadata={
                    'content_type': content_type,
                    'status_code': response.status_code
                }
            )
            
        except requests.exceptions.Timeout:
            return SourceContent(
                source_id=source_id,
                source_type='url',
                location=url,
                error="Request timed out"
            )
        except requests.exceptions.ConnectionError:
            return SourceContent(
                source_id=source_id,
                source_type='url',
                location=url,
                error="Connection failed"
            )
        except requests.exceptions.HTTPError as e:
            return SourceContent(
                source_id=source_id,
                source_type='url',
                location=url,
                error=f"HTTP error: {e}"
            )
        except Exception as e:
            return SourceContent(
                source_id=source_id,
                source_type='url',
                location=url,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _read_file(self, filepath: str) -> SourceContent:
        source_id = self._generate_id(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            if filepath.endswith(('.html', '.htm')):
                content, title = self._parse_html(raw_content)
            else:
                content = raw_content
                title = None
            
            content = self._clean_text(content)
            content = content[:self.max_content_length]
            
            filename = os.path.basename(filepath)
            
            return SourceContent(
                source_id=source_id,
                source_type='file',
                location=filepath,
                title=title or filename,
                content=content,
                content_hash=self._hash_content(content),
                length=len(content),
                metadata={
                    'filename': filename,
                    'extension': os.path.splitext(filepath)[1]
                }
            )
            
        except FileNotFoundError:
            return SourceContent(
                source_id=source_id,
                source_type='file',
                location=filepath,
                error="File not found"
            )
        except PermissionError:
            return SourceContent(
                source_id=source_id,
                source_type='file',
                location=filepath,
                error="Permission denied"
            )
        except UnicodeDecodeError:
            return SourceContent(
                source_id=source_id,
                source_type='file',
                location=filepath,
                error="Unable to decode file (invalid encoding)"
            )
        except Exception as e:
            return SourceContent(
                source_id=source_id,
                source_type='file',
                location=filepath,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _parse_html(self, html: str) -> Tuple[str, Optional[str]]:
        soup = BeautifulSoup(html, 'lxml')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        title = None
        if soup.title:
            title = soup.title.get_text(strip=True)
        
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        return text, title
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
    
    def _generate_id(self, source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:12]
    
    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def close(self):
        self.session.close()

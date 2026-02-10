"""
Document Processor - Handles various file formats and extracts code
Supports: .c, .cpp, .h, .txt, .pdf, .docx, .md
"""

import os
import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

# Document parsing libraries
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes various document formats and extracts code"""
    
    def __init__(self):
        """Initialize document processor"""
        self.supported_extensions = {
            '.c', '.cpp', '.h', '.hpp',  # C/C++ source files
            '.py',  # Python files
            '.txt', '.md',  # Text files
            '.pdf',  # PDF files
            '.docx',  # Word documents
        }
        
        # Code block patterns for markdown/text files
        self.code_block_patterns = [
            r'```(?:c|cpp|c\+\+|c\+\+)?\s*\n(.*?)```',  # Markdown code blocks
            r'```(?:python)?\s*\n(.*?)```',  # Python code blocks
            r'<code>(.*?)</code>',  # HTML code tags
            r'<pre>(.*?)</pre>',  # HTML pre tags
        ]
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file and extract code
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with extracted code and metadata
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Route to appropriate processor
        if file_extension in ['.c', '.cpp', '.h', '.hpp', '.py']:
            return await self._process_source_file(file_path)
        elif file_extension in ['.txt', '.md']:
            return await self._process_text_file(file_path)
        elif file_extension == '.pdf':
            return await self._process_pdf(file_path)
        elif file_extension == '.docx':
            return await self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    async def _process_source_file(self, file_path: str) -> Dict[str, Any]:
        """Process C/C++/Python source files directly"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        return {
            "code": code,
            "file_type": Path(file_path).suffix,
            "language": self._detect_language(code),
            "lines": len(code.split('\n')),
            "extraction_method": "direct_read"
        }
    
    async def _process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process text/markdown files and extract code blocks"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Try to extract code blocks
        extracted_code = []
        
        for pattern in self.code_block_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            extracted_code.extend(matches)
        
        # If no code blocks found, check if entire file is code
        if not extracted_code:
            # Check if content looks like code (has common code patterns)
            if self._looks_like_code(content):
                extracted_code = [content]
            else:
                # Try to find code-like sections
                code_sections = self._extract_code_sections(content)
                extracted_code.extend(code_sections)
        
        code = '\n\n'.join(extracted_code) if extracted_code else content
        
        return {
            "code": code,
            "file_type": Path(file_path).suffix,
            "language": self._detect_language(code),
            "lines": len(code.split('\n')),
            "extraction_method": "code_block_extraction",
            "code_blocks_found": len(extracted_code)
        }
    
    async def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF files and extract code"""
        if not pypdf:
            raise ImportError("pypdf library required for PDF processing")
        
        text_content = []
        code_blocks = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    text_content.append(page_text)
                    
                    # Look for code blocks in PDF text
                    for pattern in self.code_block_patterns:
                        matches = re.findall(pattern, page_text, re.DOTALL | re.IGNORECASE)
                        code_blocks.extend(matches)
            
            full_text = '\n'.join(text_content)
            
            # If no code blocks found, try to extract code-like sections
            if not code_blocks:
                code_blocks = self._extract_code_sections(full_text)
            
            code = '\n\n'.join(code_blocks) if code_blocks else full_text
            
            return {
                "code": code,
                "file_type": ".pdf",
                "language": self._detect_language(code),
                "lines": len(code.split('\n')),
                "extraction_method": "pdf_text_extraction",
                "pages": len(pdf_reader.pages),
                "code_blocks_found": len(code_blocks)
            }
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    async def _process_docx(self, file_path: str) -> Dict[str, Any]:
        """Process Word documents and extract code"""
        if not DocxDocument:
            raise ImportError("python-docx library required for DOCX processing")
        
        try:
            doc = DocxDocument(file_path)
            text_content = []
            code_blocks = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text
                text_content.append(text)
                
                # Look for code blocks
                for pattern in self.code_block_patterns:
                    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    code_blocks.extend(matches)
            
            full_text = '\n'.join(text_content)
            
            # If no code blocks found, try to extract code-like sections
            if not code_blocks:
                code_blocks = self._extract_code_sections(full_text)
            
            code = '\n\n'.join(code_blocks) if code_blocks else full_text
            
            return {
                "code": code,
                "file_type": ".docx",
                "language": self._detect_language(code),
                "lines": len(code.split('\n')),
                "extraction_method": "docx_text_extraction",
                "code_blocks_found": len(code_blocks)
            }
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            raise
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        code_lower = code.lower()
        
        # C/C++ indicators
        if '#include' in code or 'int main(' in code:
            if 'class ' in code or 'cout' in code or 'std::' in code:
                return 'cpp'
            if 'volatile' in code or '__interrupt' in code or '0x' in code:
                return 'embedded_c'
            return 'c'
        
        # Python indicators
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        
        # Default to embedded C for embedded systems context
        return 'embedded_c'
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = [
            r'#include', r'int\s+main\s*\(', r'void\s+\w+\s*\(',  # C/C++
            r'def\s+\w+\s*\(', r'import\s+\w+',  # Python
            r'\{', r'\}', r'\(', r'\)',  # Common code symbols
            r'=\s*[0-9]', r'=\s*["\']',  # Assignments
        ]
        
        matches = sum(1 for pattern in code_indicators if re.search(pattern, text))
        return matches >= 3
    
    def _extract_code_sections(self, text: str) -> list:
        """Extract code-like sections from text"""
        code_sections = []
        
        # Look for lines that look like code
        lines = text.split('\n')
        current_section = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comments-only lines
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                if current_section and len(current_section) > 3:
                    code_sections.append('\n'.join(current_section))
                current_section = []
                continue
            
            # Check if line looks like code
            if self._looks_like_code_line(line):
                current_section.append(line)
            else:
                if current_section and len(current_section) > 3:
                    code_sections.append('\n'.join(current_section))
                current_section = []
        
        # Add final section if exists
        if current_section and len(current_section) > 3:
            code_sections.append('\n'.join(current_section))
        
        return code_sections
    
    def _looks_like_code_line(self, line: str) -> bool:
        """Check if a single line looks like code"""
        code_patterns = [
            r'^\s*\w+\s+\w+\s*\(',  # Function declaration
            r'^\s*\w+\s*=\s*',  # Assignment
            r'^\s*if\s*\(',  # If statement
            r'^\s*for\s*\(',  # For loop
            r'^\s*while\s*\(',  # While loop
            r'^\s*return\s+',  # Return statement
            r'^\s*#include',  # Include directive
            r'^\s*#define',  # Define directive
            r'^\s*\{',  # Opening brace
            r'^\s*\}',  # Closing brace
        ]
        
        return any(re.search(pattern, line) for pattern in code_patterns)

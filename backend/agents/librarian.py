"""
Librarian Agent - Knowledge Base Specialist
Responsible for fetching and retrieving relevant documentation
Uses Groq (Llama 3) for document understanding and semantic search
"""

import os
import json
import chromadb
import time
from typing import List, Dict, Any, Optional
from utils.llm_client import LLMClient
from utils.document_processor import DocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
import asyncio
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from pathlib import Path 

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocSearchResult:
    """Data class for document search results"""
    text: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any]

class LibrarianAgent:
    """Agent responsible for retrieving relevant documentation and context"""
    
    def __init__(self, chroma_persist_path: str = "data/chroma_db"):
        """Initialize the Librarian Agent with Groq (Context Specialist)"""
        
        # 1. Initialize Unified LLM Client (Switched to Groq for reliability)
        self.llm = LLMClient(provider="groq")
        self.use_mock = False 
        
        # 2. Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # 3. Pre-defined Infineon keywords
        self.infineon_keywords = [
            "AURIX", "TC3xx", "microcontroller", "embedded", "safety",
            "sensor", "radar", "calibration", "MISRA-C", "ISO 26262",
            "ASIL", "interrupt", "DMA", "ADC", "PWM", "CAN", "Ethernet",
            "flash", "RAM", "watchdog", "crypto", "HSM", "real-time",
            "RTOS", "FreeRTOS", "AUTOSAR", "SPI", "I2C", "UART"
        ]

        # 4. Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
        self.collection_name = "infineon_documents"
        
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            if self.collection.count() == 0:
                self._preload_documents()
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            self._preload_documents()

    async def process_input(self, input_data: str, input_type: str = "code") -> Dict[str, Any]:
        """
        Process various input types (code, file path, document content)
        and extract code for analysis
        
        Args:
            input_data: Code string, file path, or document content
            input_type: "code", "file", or "document"
            
        Returns:
            Dictionary with extracted code and metadata
        """
        if input_type == "code":
            # Direct code input - use as-is
            return {
                "code": input_data,
                "language": self._detect_language_from_code(input_data),
                "extraction_method": "direct_code"
            }
        elif input_type == "file":
            # File path - process file
            try:
                result = await self.document_processor.process_file(input_data)
                logger.info(f"Processed file: {input_data}, extracted {len(result.get('code', ''))} chars")
                return result
            except Exception as e:
                logger.error(f"File processing failed: {e}")
                raise
        else:
            # Document content - extract code
            try:
                # Try to extract code from document content
                code_sections = self._extract_code_from_document(input_data)
                code = '\n\n'.join(code_sections) if code_sections else input_data
                return {
                    "code": code,
                    "language": self._detect_language_from_code(code),
                    "extraction_method": "document_extraction",
                    "code_blocks_found": len(code_sections)
                }
            except Exception as e:
                logger.error(f"Document extraction failed: {e}")
                # Fallback: treat entire content as code
                return {
                    "code": input_data,
                    "language": "embedded_c",
                    "extraction_method": "fallback"
                }
    
    def _detect_language_from_code(self, code: str) -> str:
        """Detect programming language from code"""
        code_lower = code.lower()
        if '#include' in code or 'int main(' in code:
            if 'class ' in code or 'cout' in code or 'std::' in code:
                return 'cpp'
            if 'volatile' in code or '__interrupt' in code or '0x' in code:
                return 'embedded_c'
            return 'c'
        elif 'def ' in code or 'import ' in code:
            return 'python'
        return 'embedded_c'
    
    def _extract_code_from_document(self, document_content: str) -> list:
        """Extract code blocks from document content"""
        import re
        code_blocks = []
        
        # Markdown code blocks
        markdown_pattern = r'```(?:c|cpp|c\+\+|python)?\s*\n(.*?)```'
        matches = re.findall(markdown_pattern, document_content, re.DOTALL)
        code_blocks.extend(matches)
        
        # HTML code tags
        html_patterns = [
            r'<code>(.*?)</code>',
            r'<pre>(.*?)</pre>',
            r'<pre><code>(.*?)</code></pre>'
        ]
        for pattern in html_patterns:
            matches = re.findall(pattern, document_content, re.DOTALL)
            code_blocks.extend(matches)
        
        return code_blocks
    
    async def analyze_context(self, code: str, context_type: str = "embedded") -> Dict[str, Any]:
        """
        Main method: Analyze code context and retrieve relevant documentation
        """
        logger.info(f"Librarian analyzing {len(code)} chars of {context_type} code")
        
        # Extract keywords from code
        keywords = self.extract_code_keywords(code)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Build search query from keywords
        search_query = f"{context_type} programming: " + ", ".join(keywords[:5])
        
        # Search for relevant documents
        relevant_docs = self.search_documents(search_query, n_results=3)
        
        # Use LLM to summarize the context
        context_summary = ""
        if relevant_docs:
            docs_text = "\n\n".join([f"Source: {doc.source}\n{doc.text}" 
                                   for doc in relevant_docs[:2]])
            
            prompt = f"""
            As an Infineon embedded systems expert, analyze this code context:
            Code Snippet:
            {code[:300]}...
            Relevant Documentation:
            {docs_text}
            Provide a brief context summary (2-3 sentences) explaining:
            1. What type of embedded system this appears to be
            2. Key safety or performance considerations
            """
            
            # UNIFIED CALL (Groq/Llama 3)
            response = self.llm.generate(prompt)
            if response:
                context_summary = response
            else:
                context_summary = "Context summary unavailable (LLM Error)."
        
        # Format the response
        return {
            "agent": "librarian",
            "status": "completed",
            "timestamp": asyncio.get_event_loop().time(),
            "context_type": context_type,
            "keywords_found": keywords,
            "relevant_documents": [
                {
                    "source": doc.source,
                    "content": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    "relevance_score": round(doc.relevance_score, 3),
                    "type": doc.metadata.get("type", "unknown")
                }
                for doc in relevant_docs
            ],
            "context_summary": context_summary,
            "search_query_used": search_query,
            "stats": {
                "code_length": len(code),
                "keywords_count": len(keywords),
                "documents_found": len(relevant_docs),
                "using_groq": True
            }
        }
    
    def _load_rag_documents(self) -> Optional[List[Dict]]:
        """Load RAG documents from data directory"""
        data_dir = Path("data")
        if not data_dir.exists():
            return None
        
        rag_docs = []
        for md_file in data_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    rag_docs.append({
                        "id": f"rag_{md_file.stem}",
                        "text": content,
                        "metadata": {
                            "source": f"RAG_{md_file.name}",
                            "type": "rag_guide",
                            "file": str(md_file)
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not load RAG file {md_file}: {e}")
        
        return rag_docs if rag_docs else None
    
    def _get_default_documents(self) -> List[Dict]:
        """Get default Infineon documentation"""
        return [
            {
                "id": "infineon_aurix_overview",
                "text": """AURIX™ TC3xx Microcontroller Family
Infineon's AURIX™ TC3xx 32-bit microcontrollers are designed for automotive and industrial applications requiring high performance and safety.
Key features:
- TriCore™ 1.6.2P architecture with up to 300 MHz
- 8 MB flash memory with ECC protection
- Hardware Security Module (HSM) for cryptographic operations
- ASIL-D capable for functional safety (ISO 26262)
- Multiple CAN-FD, Ethernet, and FlexRay interfaces
- Advanced PWM modules for motor control applications

Safety Requirements:
1. All safety-critical functions must be ASIL-D compliant
2. Memory protection units (MPU) must be configured
3. Watchdog timers must be properly initialized
4. Error Correction Codes (ECC) must be enabled for flash/ram""",
                "metadata": {"source": "Infineon_AURIX_TC3xx_Datasheet_v2.1", "type": "datasheet"}
            },
            {
                "id": "misra_c_guidelines",
                "text": """MISRA-C:2012 Guidelines for Embedded Systems
Essential rules for Infineon automotive projects:

Rule 1.3: No occurrence of undefined or critical unspecified behavior
Rule 8.4: A compatible declaration shall be visible when an object or function with external linkage is defined
Rule 10.3: The value of a complex expression of floating type may only be cast to a narrower floating type
Rule 11.4: A conversion should not be performed between a pointer to object and an integer type
Rule 14.4: The controlling expression of an if statement and the controlling expression of an iteration-statement shall have essentially Boolean type
Rule 17.2: Functions shall not call themselves, either directly or indirectly (no recursion in safety-critical code)

Critical for AURIX:
- Rule 8.13: Pointers should be used to access arrays
- Rule 18.1: All structure and union types shall be complete at the end of a translation unit""",
                "metadata": {"source": "MISRA-C:2012_Guidelines", "type": "coding_standard"}
            },
            {
                "id": "sensor_calibration_guide",
                "text": """Infineon Sensor Calibration Protocol
For accurate sensor readings (DPS310, TLI493D, etc.), follow this 3-step calibration:

1. Initialization Phase:
   - Power on sensor with stable voltage (3.3V ±5%)
   - Wait 100ms for internal oscillator stabilization
   - Send initialization command 0x01

2. Calibration Phase:
   - Apply known reference values (temperature, pressure)
   - Read raw sensor values for 10 samples
   - Calculate offset and gain coefficients
   - Store coefficients in non-volatile memory

3. Runtime Phase:
   - Apply calibration coefficients to raw readings
   - Validate readings against expected ranges
   - Implement temperature compensation if needed

Common Issues:
- Missing volatile qualifier for sensor registers
- Buffer overflow in calibration data arrays
- Incorrect timing between commands
- Missing error checking for I2C/SPI communication""",
                "metadata": {"source": "Infineon_Sensor_Calibration_Guide_v1.2", "type": "application_note"}
            },
            {
                "id": "interrupt_best_practices",
                "text": """Interrupt Service Routine Guidelines for AURIX
Critical timing requirements:
1. ISR must complete within 50μs for real-time control loops
2. Minimum 10μs between consecutive interrupts
3. Interrupt nesting depth should not exceed 3 levels

Best Practices:
- Keep ISR code minimal (move processing to main loop)
- Use DMA for data transfers when possible
- Clear interrupt flags at the beginning of ISR
- Avoid floating-point operations in ISR
- Disable interrupts only for critical sections

Safety Considerations:
- Always validate interrupt sources
- Implement timeout mechanisms
- Use watchdog to recover from stuck ISRs
- Test interrupt latency under worst-case conditions

Example of problematic code:
void __attribute__((interrupt)) Timer_ISR(void) {
    // Complex calculations here - AVOID!
    // Should only set flags and clear interrupts
}""",
                "metadata": {"source": "AURIX_Interrupt_Programming_Guide", "type": "programming_guide"}
            },
            {
                "id": "memory_management_rules",
                "text": """Memory Management for Safety-Critical Systems
Infineon Automotive Memory Allocation Rules:

1. Stack Usage:
   - Maximum stack depth: 2KB per task
   - Enable stack overflow detection in debug builds
   - Use static analysis to verify stack usage

2. Heap Usage:
   - AVOID dynamic memory allocation (malloc/free) in safety-critical code
   - Use pool allocators with fixed block sizes
   - Implement memory leak detection

3. Buffer Management:
   - All arrays must have explicit size declarations
   - Use sizeof() operator for buffer calculations
   - Implement boundary checks for all array accesses
   - Initialize all variables before use

Common Memory Bugs:
- Buffer overflow (off-by-one errors)
- Uninitialized variables
- Memory leaks in error paths
- Use-after-free in multi-threaded contexts
- Stack overflow from deep recursion

Detection Methods:
- Static analysis (MISRA-C checker)
- Runtime bounds checking
- Memory protection units (MPU)
- Watchdog for stack overflow""",
                "metadata": {"source": "Infineon_Memory_Safety_Guidelines", "type": "safety_manual"}
            }
        ]
    
    def _preload_documents(self):
        """Pre-load Infineon-specific documentation into vector database"""
        logger.info("Pre-loading Infineon documentation...")
        
        # Load RAG data files if they exist
        rag_docs = self._load_rag_documents()
        default_docs = self._get_default_documents()
        
        # Combine RAG docs with default docs
        documents = default_docs
        if rag_docs:
            documents.extend(rag_docs)
            logger.info(f"Loaded {len(rag_docs)} RAG documents")
        
        # Process and add documents to ChromaDB
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc["text"])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    **doc["metadata"],
                    "chunk_index": i,
                    "original_id": doc["id"]
                })
                all_ids.append(chunk_id)
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
        
        logger.info(f"Loaded {len(all_chunks)} document chunks into ChromaDB")
    
    def extract_code_keywords(self, code: str) -> List[str]:
        """
        Extract relevant keywords from code for document search
        """
        keywords = set()
        
        # Add Infineon-specific keywords found in code
        code_lower = code.lower()
        for keyword in self.infineon_keywords:
            if keyword.lower() in code_lower:
                keywords.add(keyword)
        
        # Extract function names and variables (simple regex approach)
        import re
        
        # Find function definitions
        functions = re.findall(r'\b(\w+)\s*\([^)]*\)\s*\{', code)
        keywords.update(functions)
        
        # Find common embedded patterns
        patterns = {
            "interrupt": r'__attribute__\s*\(\(interrupt\)\)|interrupt\s+void',
            "volatile": r'volatile\s+\w+',
            "register": r'#define\s+(PORT|DDR|PIN)\w+|register\s+\w+',
            "sensor": r'sensor|adc|read_adc|calibrate',
            "memory": r'malloc|free|buffer|array\[',
            "timer": r'timer|delay|sleep|wait',
        }
        
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                keywords.add(pattern_name)
        
        # If using LLM, get more intelligent keywords
        if not self.use_mock and self.llm:
            try:
                prompt = f"""
                Analyze this embedded C code and extract 5-10 keywords for documentation search.
                Focus on: microcontroller features, safety requirements, coding standards, and hardware interfaces.
                
                Code:
                {code[:500]}  # Limit code length
                
                Return keywords as a comma-separated list.
                """
                
                # UNIFIED CALL (Groq/Llama 3)
                response_text = self.llm.generate(prompt)
                
                if response_text:
                    # Clean potential markdown or explanation text from response
                    clean_text = response_text.replace("```", "").strip()
                    gemini_keywords = [k.strip() for k in clean_text.split(',')]
                    keywords.update(gemini_keywords[:10])
            except Exception as e:
                logger.warning(f"LLM keyword extraction failed: {e}")
        
        return list(keywords)[:15]  # Return top 15 keywords
    
    def search_documents(self, query: str, n_results: int = 5) -> List[DocSearchResult]:
        """
        Search for relevant documents using semantic search
        """
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results['documents']:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    # Convert distance to relevance score (0-1, higher is better)
                    relevance_score = 1.0 / (1.0 + distance)
                    
                    search_results.append(DocSearchResult(
                        text=doc,
                        source=metadata.get('source', 'Unknown'),
                        relevance_score=relevance_score,
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            # Return mock results for demo
            return self._get_mock_results(query)
    
    def _get_mock_results(self, query: str) -> List[DocSearchResult]:
        """Return mock results when ChromaDB or API fails"""
        mock_docs = [
            DocSearchResult(
                text="AURIX TC3xx: Interrupt Service Routines must complete within 50μs for real-time automotive applications. ISR code should be minimal and avoid complex calculations.",
                source="AURIX_Programming_Manual",
                relevance_score=0.85,
                metadata={"type": "programming_guide"}
            ),
            DocSearchResult(
                text="MISRA-C Rule 17.2: Functions shall not call themselves, either directly or indirectly. Recursion is prohibited in safety-critical embedded systems.",
                source="MISRA-C:2012",
                relevance_score=0.78,
                metadata={"type": "coding_standard"}
            ),
            DocSearchResult(
                text="Sensor calibration requires 3-step initialization: power stabilization, coefficient calculation, and runtime application. Missing any step leads to inaccurate readings.",
                source="Infineon_Sensor_Calibration_Guide",
                relevance_score=0.72,
                metadata={"type": "application_note"}
            ),
        ]
        
        # Filter based on query keywords
        query_lower = query.lower()
        filtered_results = []
        for result in mock_docs:
            if any(keyword in query_lower for keyword in ['interrupt', 'isr', 'timer']):
                if 'interrupt' in result.text.lower():
                    filtered_results.append(result)
            elif any(keyword in query_lower for keyword in ['sensor', 'calibrate', 'adc']):
                if 'sensor' in result.text.lower():
                    filtered_results.append(result)
            else:
                filtered_results.append(result)
        
        return filtered_results[:3]
    
    def add_document(self, text: str, source: str, metadata: Dict = None):
        """
        Add a new document to the knowledge base
        """
        try:
            # Split document
            chunks = self.text_splitter.split_text(text)
            
            # Generate IDs
            doc_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{source}_{doc_hash}_chunk_{i}"
                
                # Add to collection
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "source": source,
                        "chunk_index": i,
                        **(metadata or {})
                    }],
                    ids=[chunk_id]
                )
            
            logger.info(f"Added document '{source}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            # Simple count for ChromaDB 0.4.x
            total_chunks = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_document_chunks": total_chunks,
                "using_groq": not self.use_mock,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": f"Could not retrieve collection stats: {str(e)}"}


# Quick test function
def test_librarian():
    """Test the librarian agent"""
    import asyncio
    
    # Initialize
    librarian = LibrarianAgent()
    
    # Test code snippet
    test_code = """
    #include <stdint.h>
    
    volatile uint32_t* SENSOR_REG = (uint32_t*)0x40021000;
    
    void __attribute__((interrupt)) timer_isr(void) {
        static uint32_t counter = 0;
        counter++;
        
        // Read sensor value
        uint32_t sensor_value = *SENSOR_REG;
        
        // Process sensor data
        if(counter >= 1000) {
            calibrate_sensor();
            counter = 0;
        }
    }
    
    void calibrate_sensor() {
        float calibration_data[10];
        for(int i = 0; i <= 10; i++) {
            calibration_data[i] = read_adc() * 0.1;
        }
    }
    """
    
    # Run analysis
    result = asyncio.run(librarian.analyze_context(test_code, "embedded"))
    
    # Print results
    print("\n" + "="*60)
    print("LIBRARIAN AGENT TEST RESULTS")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Keywords found: {result['keywords_found']}")
    print(f"Documents found: {len(result['relevant_documents'])}")
    print("\nRelevant Documents:")
    for i, doc in enumerate(result['relevant_documents'], 1):
        print(f"\n{i}. {doc['source']} (Relevance: {doc['relevance_score']})")
        print(f"   {doc['content'][:150]}...")
    
    print("\nContext Summary:")
    print(result['context_summary'])
    
    print("\nCollection Stats:")
    print(json.dumps(librarian.get_collection_stats(), indent=2))


if __name__ == "__main__":
    test_librarian()
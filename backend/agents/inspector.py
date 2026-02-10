"""
Inspector Agent - Static Analysis Specialist
Responsible for detecting bugs, coding violations, and security issues in code
Uses pattern matching, static analysis, and Gemini API for intelligent bug detection
"""

import re
import ast
import time
import asyncio
import logging
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# # UPDATED: Import new library
# from google import genai as google_genai
from utils.llm_client import LLMClient

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeFinding:
    """Data class for code analysis findings"""
    line_number: int
    column: int
    bug_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    description: str
    code_snippet: str
    confidence: float  # 0.0 to 1.0
    rule_id: Optional[str] = None  # e.g., "MISRA-C:2012 Rule 17.2"
    suggested_fix: Optional[str] = None

class InspectorAgent:
    """Agent responsible for static code analysis and bug detection"""
    
    def __init__(self):
        """Initialize the Inspector Agent with HuggingFace (Mistral)"""
        
        # 1. Initialize Unified LLM Client
        self.llm = LLMClient(provider="groq") 
        self.use_llm = True
        # Initialize bug patterns and rules
        self._initialize_rules()
        
        # Language-specific patterns
        self.language_patterns = {
            "c": self._get_c_patterns(),
            "cpp": self._get_cpp_patterns(),
            "python": self._get_python_patterns(),
            "embedded_c": self._get_embedded_c_patterns(),
        }

    async def llm_analysis(self, code: str, language: str, rule_findings: List[CodeFinding]) -> List[CodeFinding]:
        """Use LLM for advanced code analysis"""
        if not self.use_llm:
            return rule_findings
        
        prompt = f"""
        Analyze this {language} code for embedded system bugs (Infineon/MISRA context).
        Focus on safety-critical issues, buffer overflows, and race conditions.
        
        Code:
        {code[:2000]}
        
        Return ONLY a JSON array with objects: {{ "line_number": int, "bug_type": str, "severity": str, "description": str }}.
        """
        
        # UNIFIED CALL
        response_text = await asyncio.to_thread(self.llm.generate, prompt)
        
        if response_text:
            try:
                # Clean up response (Mistral sometimes adds text before/after JSON)
                text = response_text.replace("```json", "").replace("```", "").strip()
                start = text.find('[')
                end = text.rfind(']') + 1
                if start != -1 and end != 0:
                    data = json.loads(text[start:end])
                    new_findings = []
                    for item in data:
                        new_findings.append(CodeFinding(
                            line_number=item.get("line_number", 0),
                            column=0,
                            bug_type=item.get("bug_type", "AI_DETECTED"),
                            severity=item.get("severity", "MEDIUM"),
                            description=item.get("description", ""),
                            code_snippet="AI Analysis",
                            confidence=0.8,
                            suggested_fix=item.get("suggested_fix")
                        ))
                    return rule_findings + new_findings
            except Exception as e:
                logger.error(f"LLM parsing error: {e}")
        
        return rule_findings

    def _initialize_rules(self):
        """Initialize bug detection rules and patterns"""
        
        # Common bug patterns (language-agnostic)
        self.common_patterns = [
            # Buffer overflow / off-by-one
            {
                "name": "BUFFER_OVERFLOW",
                "pattern": r'for\s*\(\s*(?:int|uint\d*_t)?\s*\w+\s*=\s*\d*\s*;\s*\w+\s*<=\s*\w+\s*;\s*\w+\+\+',
                "description": "Potential off-by-one error (<= instead of <) leading to buffer overflow",
                "severity": "CRITICAL",
                "confidence": 0.85
            },
            {
                "name": "UNINITIALIZED_VAR",
                "pattern": r'(?:int|float|double|char|void)\s+\w+\s*;\s*(?!=\s*)(?:\w+\s*=)',
                "description": "Variable declared but not initialized before use",
                "severity": "HIGH",
                "confidence": 0.70
            },
            {
                "name": "MEMORY_LEAK",
                "pattern": r'malloc\s*\([^)]+\)\s*(?!free)',
                "description": "Memory allocated but not freed, potential memory leak",
                "severity": "HIGH",
                "confidence": 0.65
            },
            {
                "name": "DIVISION_BY_ZERO",
                "pattern": r'\/\s*(?:0|0\.0|\w+\s*\*\s*0)',
                "description": "Potential division by zero",
                "severity": "CRITICAL",
                "confidence": 0.90
            }
        ]
        
        # MISRA-C rules for embedded systems
        self.misra_rules = [
            {
                "id": "MISRA-C:2012 Rule 17.2",
                "pattern": r'\w+\s*\([^)]*\)\s*{[^}]*\w+\s*\([^)]*\)\s*;',
                "description": "Functions shall not call themselves (no recursion)",
                "severity": "HIGH",
                "confidence": 0.80
            },
            {
                "id": "MISRA-C:2012 Rule 11.4",
                "pattern": r'\(\s*(?:int|long|short)\s*\)\s*\w+',
                "description": "Avoid conversion between pointer and integer",
                "severity": "MEDIUM",
                "confidence": 0.70
            }
        ]
        
        # Infineon-specific rules
        self.infineon_rules = [
            {
                "id": "INFINEON-VOLATILE-001",
                "pattern": r'(?:register|PORT|DDR|PIN|ADC|TIMER)\s+\w+\s*[=;]',
                "description": "Hardware register missing volatile qualifier",
                "severity": "HIGH",
                "confidence": 0.85
            },
            {
                "id": "INFINEON-ISR-001",
                "pattern": r'void\s+__attribute__\s*\(\(interrupt\)\)\s+\w+\s*\([^)]*\)\s*{[^}]*float\s+\w+',
                "description": "Floating point operations in ISR - timing violation",
                "severity": "MEDIUM",
                "confidence": 0.80
            }
        ]
    
    def _get_c_patterns(self) -> List[Dict]:
        return [
            {"name": "C-UNSAFE_FUNCTION", "pattern": r'gets\s*\(', "description": "Unsafe function 'gets()'", "severity": "CRITICAL", "confidence": 0.95},
            {"name": "C-STRCPY", "pattern": r'strcpy\s*\(', "description": "Unsafe strcpy() - use strncpy()", "severity": "HIGH", "confidence": 0.85},
        ]
    
    def _get_cpp_patterns(self) -> List[Dict]:
        return [{"name": "CPP-EXCEPTION", "pattern": r'throw\s+\w+', "description": "Avoid exceptions in embedded code", "severity": "MEDIUM", "confidence": 0.70}]
    
    def _get_python_patterns(self) -> List[Dict]:
        return [{"name": "PYTHON-BROAD_EXCEPT", "pattern": r'except\s*:', "description": "Bare except clause", "severity": "MEDIUM", "confidence": 0.80}]
    
    def _get_embedded_c_patterns(self) -> List[Dict]:
        return [
            {"name": "EMBEDDED-NO_RETURN", "pattern": r'void\s+main\s*\(\s*\)\s*{', "description": "main() should return int", "severity": "MEDIUM", "confidence": 0.75},
            {"name": "EMBEDDED-HEAP_USAGE", "pattern": r'malloc\s*\(', "description": "Dynamic memory in embedded system", "severity": "MEDIUM", "confidence": 0.70},
        ]
    
    def detect_language(self, code: str) -> str:
        if "#include" in code or re.search(r'int\s+main\s*\(', code):
            if "class " in code or "cout " in code: return "cpp"
            if "volatile" in code or "__interrupt" in code: return "embedded_c"
            return "c"
        elif "def " in code or "import " in code: return "python"
        return "embedded_c"
    
    def rule_based_analysis(self, code: str, language: str) -> List[CodeFinding]:
        findings = []
        lines = code.split('\n')
        all_patterns = self.common_patterns.copy()
        
        if language in self.language_patterns: all_patterns.extend(self.language_patterns[language])
        if language in ["c", "cpp", "embedded_c"]: all_patterns.extend(self.misra_rules)
        if language == "embedded_c": all_patterns.extend(self.infineon_rules)
        
        for line_num, line in enumerate(lines, start=1):
            for pattern in all_patterns:
                if re.search(pattern["pattern"], line, re.IGNORECASE):
                    match = re.search(pattern["pattern"], line, re.IGNORECASE)
                    findings.append(CodeFinding(
                        line_number=line_num,
                        column=match.start() if match else 0,
                        bug_type=pattern.get("name", pattern.get("id", "UNKNOWN")),
                        severity=pattern["severity"],
                        description=pattern["description"],
                        code_snippet=line.strip(),
                        confidence=pattern["confidence"],
                        rule_id=pattern.get("id")
                    ))
        return findings
    
    def _get_line_from_code(self, code: str, line_number: int) -> str:
        """Get specific line from code"""
        lines = code.split('\n')
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()
        return ""
    
    def calculate_metrics(self, findings: List[CodeFinding]) -> Dict[str, Any]:
        """Calculate analysis metrics"""
        severities = [f.severity for f in findings]
        
        return {
            "total_findings": len(findings),
            "critical_count": severities.count("CRITICAL"),
            "high_count": severities.count("HIGH"),
            "medium_count": severities.count("MEDIUM"),
            "low_count": severities.count("LOW"),
            "info_count": severities.count("INFO"),
            "unique_bug_types": len(set(f.bug_type for f in findings)),
            "avg_confidence": sum(f.confidence for f in findings) / len(findings) if findings else 0,
        }
    
    async def analyze_code(self, code: str, librarian_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main method: Analyze code for bugs and issues
        """
        logger.info(f"Inspector analyzing {len(code)} chars of code")
        
        # Detect language
        language = self.detect_language(code)
        logger.info(f"Detected language: {language}")
        
        # Rule-based analysis
        rule_findings = self.rule_based_analysis(code, language)
        logger.info(f"Rule-based analysis found {len(rule_findings)} issues")
        
        if self.use_llm:
            # Note: We also renamed gemini_analysis to llm_analysis earlier
            findings = await self.llm_analysis(code, language, rule_findings)
            logger.info(f"LLM-enhanced analysis found {len(findings)} total issues")
        else:
            findings = rule_findings
        
        # Apply librarian context if available
        if librarian_context:
            findings = self._apply_librarian_context(findings, librarian_context)
        
        # Calculate metrics
        metrics = self.calculate_metrics(findings)
        
        # Sort findings by severity (critical first)
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        findings.sort(key=lambda f: severity_order.get(f.severity, 5))
        
        # Format response
        return {
            "agent": "inspector",
            "status": "completed",
            "timestamp": asyncio.get_event_loop().time(),
            "language_detected": language,
            "total_findings": len(findings),
            "metrics": metrics,
            "findings": [
                {
                    "line": f.line_number,
                    "column": f.column,
                    "bug_type": f.bug_type,
                    "severity": f.severity,
                    "description": f.description,
                    "code_snippet": f.code_snippet,
                    "confidence": round(f.confidence, 2),
                    "rule_id": f.rule_id,
                    "suggested_fix": f.suggested_fix
                }
                for f in findings
            ],
            # --- FIX: Changed status string ---
            "analysis_method": "llm_enhanced" if self.use_llm else "rule_based",
            "stats": {
                "code_length": len(code),
                "lines_of_code": len(code.split('\n')),
                "using_llm": self.use_llm,
                "language": language
            }
        }
    
    def _apply_librarian_context(self, findings: List[CodeFinding], context: Dict) -> List[CodeFinding]:
        """Enhance findings with librarian context"""
        if not context.get("relevant_documents"):
            return findings
        
        # Extract key terms from documents
        doc_terms = set()
        for doc in context.get("relevant_documents", []):
            content = doc.get("content", "").lower()
            # Add relevant terms
            if "misra" in content:
                doc_terms.add("MISRA")
            if "interrupt" in content:
                doc_terms.add("INTERRUPT")
            if "volatile" in content:
                doc_terms.add("VOLATILE")
            if "safety" in content:
                doc_terms.add("SAFETY")
        
        # Enhance findings based on context
        enhanced_findings = []
        for finding in findings:
            # Check if finding relates to context
            finding_text = (finding.description + " " + finding.bug_type).upper()
            context_relevant = any(term in finding_text for term in doc_terms)
            
            if context_relevant:
                # Increase confidence for context-relevant findings
                finding.confidence = min(finding.confidence * 1.2, 1.0)
            
            enhanced_findings.append(finding)
        
        return enhanced_findings

# Test function
if __name__ == "__main__":
    inspector = InspectorAgent()
    test_code = "void main() { int i; i = i + 1; }"
    result = asyncio.run(inspector.analyze_code(test_code))
    print(json.dumps(result, indent=2))
"""
Fixer Agent - Code Repair Specialist
Responsible for generating fixes, patches, and corrected code for detected bugs
Uses Gemini API for intelligent code generation and rule-based templates
"""

import asyncio
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from dotenv import load_dotenv

# Updated Gemini import
# from google import genai as google_genai
from utils.llm_client import LLMClient

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeFix:
    """Data class for code fix suggestions"""
    bug_id: str
    line_number: int
    bug_type: str
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    test_case: Optional[str] = None
    diff_output: Optional[str] = None
    safety_notes: Optional[str] = None

class FixerAgent:
    """Agent responsible for generating code fixes and patches"""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the Fixer Agent with Groq (Code Specialist)"""
        
        # 1. Initialize Unified LLM Client
        self.llm = LLMClient(provider="groq")
        self.use_llm = True
        
        # Initialize fix templates
        self._initialize_fix_templates()
        
        # Infineon-specific patterns
        self.infineon_patterns = {
            "volatile": {
                "before": r'(\w+\s*\*\s*\w+\s*=\s*\(\w+\s*\*\)\s*0x[0-9A-F]+)',
                "after": r'volatile \1',
                "explanation": "Added volatile qualifier for hardware register access"
            },
            "isr_simple": {
                "before": r'void\s+__attribute__\s*\(\(interrupt\)\)\s+\w+\s*\([^)]*\)\s*{[^}]*float\s+\w+',
                "after": "// Moved complex calculations out of ISR",
                "explanation": "Avoid floating-point operations in Interrupt Service Routines"
            }
        }

    async def _generate_llm_fix(self, diagnosis: Dict, code_context: str, 
                                  original_code: str, librarian_context: Dict) -> Optional[CodeFix]:
        """Use LLM to generate intelligent fix"""
        if not self.use_llm:
            return None
        
        try:
            # Prepare context from librarian
            context_summary = ""
            if librarian_context and librarian_context.get("relevant_documents"):
                docs = librarian_context["relevant_documents"][:2]
                context_summary = "\n".join([
                    f"Document {i+1}: {doc.get('source', 'Unknown')} - {doc.get('content', '')[:200]}"
                    for i, doc in enumerate(docs)
                ])
            
            # Prepare prompt for LLM
            prompt = f"""
            As an Infineon embedded systems expert, generate a code fix for this bug.
            
            BUG DIAGNOSIS:
            - Type: {diagnosis.get('bug_type')}
            - Line: {diagnosis.get('line')}
            - Severity: {diagnosis.get('severity')}
            - Root Cause: {diagnosis.get('root_cause', 'Unknown')}
            - Impact: {diagnosis.get('impact_analysis', 'Unknown')[:200]}
            
            CODE CONTEXT (Line {diagnosis.get('line')}):
            ```c
            {code_context}
            ```
            
            FULL FUNCTION (for context):
            ```c
            {self._extract_function(original_code, diagnosis.get('line', 0))}
            ```
            
            RELEVANT DOCUMENTATION:
            {context_summary}
            
            Generate a COMPLETE fix with:
            1. Fixed code snippet (show only the changed lines)
            2. Brief explanation of the fix
            3. A simple test case to verify the fix
            4. Safety notes for Infineon hardware
            
            Format response as JSON with these exact keys:
            - "fixed_code": string (the corrected code)
            - "explanation": string
            - "test_case": string (C test case)
            - "safety_notes": string (Infineon-specific considerations)
            
            Only return the JSON object, no other text.
            """
            
            # UNIFIED CALL
            response_text = await asyncio.to_thread(self.llm.generate, prompt)
            
            # Parse response
            if response_text:
                # Find JSON in response
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_str = response_text[json_start:json_end]
                    gemini_result = json.loads(json_str)
                    
                    # Generate diff
                    diff_output = self._generate_diff(code_context, gemini_result.get('fixed_code', ''))
                    
                    return CodeFix(
                        bug_id=diagnosis.get("bug_id", "UNKNOWN"),
                        line_number=diagnosis.get("line", 0),
                        bug_type=diagnosis.get("bug_type", "UNKNOWN"),
                        original_code=code_context.strip(),
                        fixed_code=gemini_result.get('fixed_code', ''),
                        explanation=gemini_result.get('explanation', 'Gemini-generated fix'),
                        confidence=min(diagnosis.get('confidence', 0.7) * 1.2, 1.0),
                        test_case=gemini_result.get('test_case'),
                        diff_output=diff_output,
                        safety_notes=gemini_result.get('safety_notes')
                    )
            
        except Exception as e:
            logger.error(f"LLM fix generation failed: {e}")
        
        return None
    
    def _initialize_fix_templates(self):
        """Initialize rule-based fix templates"""
        
        self.fix_templates = {
            # Buffer overflow fixes
            "BUFFER_OVERFLOW": {
                "pattern": r'for\s*\(\s*(?:int|uint\d*_t)?\s*(\w+)\s*=\s*(\d*)\s*;\s*\w+\s*<=\s*(\w+)\s*;\s*\w+\+\+',
                "template": "for({type} {var} = {start}; {var} < {limit}; {var}++)",
                "explanation": "Changed loop condition from <= to < to prevent off-by-one buffer overflow",
                "test_case": "TEST(test_buffer_bounds) {{\n    // Test with boundary values\n    int buffer[{limit}];\n    fill_buffer(buffer, {limit});\n    // Verify no overflow\n}}"
            },
            
            # Memory leak fixes
            "MEMORY_LEAK": {
                "pattern": r'(\w+\s*\*\s*\w+\s*=\s*malloc\s*\([^)]+\))',
                "template": "{allocation}\nif({var}) {{\n    // ... use memory ...\n    free({var});\n}}",
                "explanation": "Added proper memory deallocation with error checking",
                "safety_notes": "Always check malloc return value and free memory in all execution paths"
            },
            
            # Uninitialized variable fixes
            "UNINITIALIZED_VAR": {
                "pattern": r'(int|float|double|char|void)\s+(\w+)\s*;',
                "template": "{type} {var} = 0;",
                "explanation": "Initialized variable to prevent undefined behavior",
                "test_case": "TEST(test_initialization) {{\n    // Variable should have defined value\n    {type} test_var = 0;\n    ASSERT_EQ(test_var, 0);\n}}"
            },
            
            # Missing volatile fixes
            "MISSING_VOLATILE": {
                "pattern": r'(\w+\s*\*\s*\w+\s*=\s*\(\w+\s*\*\)\s*0x[0-9A-F]+)',
                "template": "volatile {line}",
                "explanation": "Added volatile qualifier for hardware register access",
                "safety_notes": "Hardware registers must be volatile to prevent compiler optimization issues"
            },
            
            # Division by zero fixes
            "DIVISION_BY_ZERO": {
                "pattern": r'(\w+)\s*/\s*(\w+)',
                "template": "({denominator} != 0) ? ({numerator} / {denominator}) : 0",
                "explanation": "Added safety check for division by zero",
                "test_case": "TEST(test_division_safety) {{\n    // Test with zero denominator\n    ASSERT_EQ(safe_divide(10, 0), 0);\n    ASSERT_EQ(safe_divide(10, 2), 5);\n}}"
            },
            
            # ISR timing fixes
            "ISR_TIMING": {
                "pattern": r'void\s+__attribute__\s*\(\(interrupt\)\)\s+(\w+)\s*\([^)]*\)\s*{',
                "template": "void __attribute__((interrupt)) {isr_name}(void) {{\n    // Minimal ISR - set flag and exit\n    g_{isr_name}_flag = 1;\n    // Clear interrupt\n}}",
                "explanation": "Simplified ISR to meet real-time constraints (max 50μs)",
                "safety_notes": "ISR must complete within 50μs for AURIX real-time requirements"
            }
        }
    
    def _extract_variables(self, code: str, pattern: str) -> Dict[str, str]:
        """Extract variables from code using regex pattern"""
        match = re.search(pattern, code)
        if not match:
            return {}
        
        # For buffer overflow pattern
        if "BUFFER_OVERFLOW" in pattern:
            return {
                "var": match.group(1) if match.group(1) else "i",
                "start": match.group(2) if match.group(2) else "0",
                "limit": match.group(3) if match.group(3) else "size"
            }
        
        # For memory leak pattern
        if "MEMORY_LEAK" in pattern:
            var_match = re.search(r'\*\s*(\w+)\s*=', match.group(0))
            return {
                "var": var_match.group(1) if var_match else "ptr",
                "allocation": match.group(0)
            }
        
        return {}
    
    def _generate_rule_based_fix(self, diagnosis: Dict, code_context: str) -> CodeFix:
        """Generate fix using rule-based templates"""
        
        bug_type = diagnosis.get("bug_type", "UNKNOWN")
        line_number = diagnosis.get("line", 0)
        
        # Get template for this bug type
        template = self.fix_templates.get(bug_type, {})
        
        if not template:
            # Generic fix template
            return CodeFix(
                bug_id=diagnosis.get("bug_id", "UNKNOWN"),
                line_number=line_number,
                bug_type=bug_type,
                original_code=code_context,
                fixed_code=f"// TODO: Fix for {bug_type} - manual review required",
                explanation=f"Manual fix required for {bug_type}",
                confidence=0.5
            )
        
        # Extract variables from code
        variables = self._extract_variables(code_context, template["pattern"])
        
        # Generate fixed code
        fixed_code = template["template"]
        for key, value in variables.items():
            fixed_code = fixed_code.replace(f"{{{key}}}", value)
        
        # Generate test case if template has one
        test_case = None
        if "test_case" in template:
            test_case = template["test_case"]
            for key, value in variables.items():
                test_case = test_case.replace(f"{{{key}}}", value)
        
        # Generate diff output
        diff_output = self._generate_diff(code_context, fixed_code)
        
        return CodeFix(
            bug_id=diagnosis.get("bug_id", "UNKNOWN"),
            line_number=line_number,
            bug_type=bug_type,
            original_code=code_context.strip(),
            fixed_code=fixed_code,
            explanation=template.get("explanation", "Fixed code issue"),
            confidence=0.8,
            test_case=test_case,
            diff_output=diff_output,
            safety_notes=template.get("safety_notes")
        )
    
    def _extract_function(self, code: str, line_number: int) -> str:
        """Extract the containing function for context"""
        lines = code.split('\n')
        if line_number <= 0 or line_number > len(lines):
            return code[:500]  # Return first 500 chars as fallback
        
        # Find function start
        start = line_number - 1
        while start > 0 and not re.search(r'^\s*\w+\s+\w+\s*\([^)]*\)\s*{', lines[start - 1]):
            start -= 1
        
        # Find function end
        end = line_number - 1
        brace_count = 0
        while end < len(lines):
            brace_count += lines[end].count('{')
            brace_count -= lines[end].count('}')
            if brace_count <= 0 and end > line_number:
                break
            end += 1
        
        # Return function (or reasonable context)
        function_lines = lines[max(0, start-2):min(len(lines), end+2)]
        return '\n'.join(function_lines)
    
    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate simple diff output"""
        # Simple diff implementation
        orig_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        diff = []
        for i, (orig, fix) in enumerate(zip(orig_lines, fixed_lines)):
            if orig != fix:
                diff.append(f"Line {i+1}:")
                diff.append(f"  - {orig}")
                diff.append(f"  + {fix}")
        
        return '\n'.join(diff) if diff else "No changes (context only)"
    
    def _get_code_context(self, code: str, line_number: int, context_lines: int = 3) -> str:
        """Get code context around a specific line"""
        lines = code.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context.append(f"{prefix}{lines[i]}")
        
        return '\n'.join(context)
    
    async def generate_fixes(self, diagnostician_result: Dict, 
                           original_code: str,
                           librarian_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main method: Generate fixes for diagnosed bugs
        
        Args:
            diagnostician_result: Results from Diagnostician agent
            original_code: Original source code
            librarian_context: Context from Librarian agent
            
        Returns:
            Dictionary with fix suggestions
        """
        logger.info(f"Fixer generating fixes for {len(diagnostician_result.get('diagnoses', []))} bugs")
        
        fixes = []
        stats = {
            "total_fixes": 0,
            "gemini_fixes": 0,
            "rule_based_fixes": 0,
            "with_test_cases": 0,
            "with_safety_notes": 0
        }
        
        for diagnosis in diagnostician_result.get('diagnoses', []):
            stats["total_fixes"] += 1
            
            # Get code context
            line_number = diagnosis.get('line', 0)
            code_context = self._get_code_context(original_code, line_number)
            
            # Try Gemini fix first
            fix = None
            if self.use_llm:
                fix = await self._generate_llm_fix(
                    diagnosis, 
                    code_context,
                    original_code,
                    librarian_context or {}
                )
                if fix:
                    stats["gemini_fixes"] += 1
            
            # Fall back to rule-based fix
            if not fix:
                fix = self._generate_rule_based_fix(diagnosis, code_context)
                stats["rule_based_fixes"] += 1
            
            # Update stats
            if fix.test_case:
                stats["with_test_cases"] += 1
            if fix.safety_notes:
                stats["with_safety_notes"] += 1
            
            fixes.append(fix)
        
        # Generate patch file if there are fixes
        patch_content = None
        if fixes:
            patch_content = self._generate_patch_file(fixes, original_code)
        
        return {
            "agent": "fixer",
            "status": "completed",
            "timestamp": asyncio.get_event_loop().time(),
            "total_fixes": len(fixes),
            "fixes": [
                {
                    "bug_id": f.bug_id,
                    "line": f.line_number,
                    "bug_type": f.bug_type,
                    "original": f.original_code,
                    "fixed": f.fixed_code,
                    "explanation": f.explanation,
                    "confidence": round(f.confidence, 2),
                    "test_case": f.test_case,
                    "diff": f.diff_output,
                    "safety_notes": f.safety_notes
                }
                for f in fixes
            ],
            "patch_file": patch_content,
            "stats": stats,
            "summary": self._generate_fix_summary(fixes, stats)
        }
    
    def _generate_patch_file(self, fixes: List[CodeFix], original_code: str) -> str:
        """Generate a unified diff patch file"""
        lines = original_code.split('\n')
        patches = []
        
        for fix in fixes:
            if fix.line_number <= 0 or fix.line_number > len(lines):
                continue
            
            # Simple patch generation
            original_lines = fix.original_code.split('\n')
            fixed_lines = fix.fixed_code.split('\n')
            
            patch = f"@@ -{fix.line_number},{len(original_lines)} +{fix.line_number},{len(fixed_lines)} @@\n"
            for orig in original_lines:
                patch += f"-{orig}\n"
            for fix_line in fixed_lines:
                patch += f"+{fix_line}\n"
            
            patches.append(patch)
        
        if patches:
            header = "--- original.c\n+++ fixed.c\n"
            return header + "\n".join(patches)
        return "No patches generated"
    
    def _generate_fix_summary(self, fixes: List[CodeFix], stats: Dict) -> Dict:
        """Generate summary of fixes"""
        if not fixes:
            return {"message": "No fixes generated", "status": "NO_FIXES"}
        
        # Group by bug type
        bug_types = {}
        for fix in fixes:
            bug_types[fix.bug_type] = bug_types.get(fix.bug_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(f.confidence for f in fixes) / len(fixes) if fixes else 0
        
        # Check if fixes include Infineon-specific patterns
        has_infineon_fixes = any(
            fix.safety_notes and "Infineon" in fix.safety_notes 
            for fix in fixes
        )
        
        return {
            "total_fixes": len(fixes),
            "unique_bug_types": len(bug_types),
            "most_common_bug": max(bug_types.items(), key=lambda x: x[1])[0] if bug_types else "NONE",
            "avg_confidence": round(avg_confidence, 2),
            "has_infineon_specific": has_infineon_fixes,
            "has_test_cases": stats["with_test_cases"] > 0,
            "has_safety_notes": stats["with_safety_notes"] > 0,
            "patch_available": bool(fixes),
            "status": "FIXES_GENERATED"
        }


# Test function
async def test_fixer():
    """Test the fixer agent"""
    from agents.librarian import LibrarianAgent
    from agents.inspector import InspectorAgent
    from agents.diagnostician import DiagnosticianAgent
    
    print("\n" + "="*80)
    print("🔧 FIXER AGENT TEST")
    print("="*80)
    
    # Initialize all agents
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    diagnostician = DiagnosticianAgent()
    fixer = FixerAgent()
    
    # Test code with bugs
    test_code = """
    // Infineon AURIX Test Code
    #include <stdint.h>
    #include <stdlib.h>
    
    // Missing volatile for hardware register
    uint32_t* SENSOR_CTRL = (uint32_t*)0xF0001000;
    
    // Buffer overflow risk
    void read_sensor_data(uint8_t* buffer, uint8_t size) {
        for(uint8_t i = 0; i <= size; i++) {  // Off-by-one
            buffer[i] = read_adc();
        }
    }
    
    // Memory leak in error handling
    void process_frame() {
        uint8_t* frame = malloc(1024);
        if (!frame) return;  // No free on error path
        
        // Process frame
        // Missing: free(frame);
    }
    
    // ISR with timing violation
    void __attribute__((interrupt)) Timer_ISR(void) {
        float readings[10];
        for(int i = 0; i < 10; i++) {
            readings[i] = read_temperature() * 1.5;  // Float ops in ISR
        }
    }
    """
    
    print("📝 Analyzing and fixing test code...")
    
    # Run through pipeline
    librarian_result = await librarian.analyze_context(test_code, "embedded")
    inspector_result = await inspector.analyze_code(test_code, librarian_result)
    diagnostician_result = await diagnostician.diagnose_bugs(
        inspector_result, 
        librarian_result, 
        test_code
    )
    
    # Generate fixes
    fixer_result = await fixer.generate_fixes(
        diagnostician_result,
        test_code,
        librarian_result
    )
    
    print(f"✅ Generated {fixer_result['total_fixes']} fixes")
    
    # Show stats
    stats = fixer_result['stats']
    print(f"\n📊 FIX STATISTICS:")
    print(f"   Gemini fixes: {stats['gemini_fixes']}")
    print(f"   Rule-based fixes: {stats['rule_based_fixes']}")
    print(f"   With test cases: {stats['with_test_cases']}")
    print(f"   With safety notes: {stats['with_safety_notes']}")
    
    # Show summary
    summary = fixer_result['summary']
    print(f"\n📈 SUMMARY:")
    print(f"   Most common bug: {summary['most_common_bug']}")
    print(f"   Average confidence: {summary['avg_confidence']}")
    print(f"   Infineon-specific: {'Yes' if summary['has_infineon_specific'] else 'No'}")
    print(f"   Patch available: {'Yes' if summary['patch_available'] else 'No'}")
    
    # Show example fixes
    print(f"\n🔧 EXAMPLE FIXES:")
    for i, fix in enumerate(fixer_result['fixes'][:2], 1):  # Show first 2
        print(f"\n{i}. {fix['bug_type']} at line {fix['line']}")
        print(f"   Original: {fix['original'][:60]}...")
        print(f"   Fixed: {fix['fixed'][:60]}...")
        print(f"   Explanation: {fix['explanation']}")
        
        if fix.get('safety_notes'):
            print(f"   Safety: {fix['safety_notes']}")
        
        if fix.get('diff'):
            print(f"   Diff:\n{fix['diff'][:100]}...")
    
    # Show patch preview
    if fixer_result.get('patch_file'):
        print(f"\n📁 PATCH FILE PREVIEW:")
        print("-"*40)
        patch_lines = fixer_result['patch_file'].split('\n')[:10]
        for line in patch_lines:
            print(line)
        print("...")
    
    print(f"\n{'='*80}")
    print("✅ FIXER AGENT READY!")
    print(f"{'='*80}")

async def quick_test():
    """Quick test without external dependencies"""
    print("🚀 Quick Fixer Test")
    
    fixer = FixerAgent()
    
    # Mock diagnosis
    mock_diagnosis = {
        "diagnoses": [
            {
                "bug_id": "BUG-BUFF-0007-123456",
                "line": 7,
                "bug_type": "BUFFER_OVERFLOW",
                "severity": "CRITICAL",
                "root_cause": "Off-by-one error in loop condition",
                "impact_analysis": "Buffer overflow can corrupt memory",
                "confidence": 0.85
            }
        ]
    }
    
    mock_code = """
    void test() {
        int buffer[10];
        for(int i = 0; i <= 10; i++) {
            buffer[i] = i * 2;
        }
    }
    """
    
    result = await fixer.generate_fixes(mock_diagnosis, mock_code)
    
    print(f"Generated {result['total_fixes']} fixes")
    
    for fix in result['fixes']:
        print(f"\n• {fix['bug_type']} at line {fix['line']}")
        print(f"  Original: {fix['original']}")
        print(f"  Fixed: {fix['fixed']}")
        print(f"  Explanation: {fix['explanation']}")

if __name__ == "__main__":
    # Run quick test by default
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        asyncio.run(test_fixer())
    else:
        # Quick test without external dependencies
        asyncio.run(quick_test())
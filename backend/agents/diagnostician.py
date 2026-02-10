"""
Diagnostician Agent - Root Cause Analyst
Responsible for explaining why bugs are problematic, their impact, and root causes
Uses Gemini API for intelligent analysis and contextual understanding
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional
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
class BugDiagnosis:
    """Data class for bug diagnosis results"""
    bug_id: str
    line_number: int
    bug_type: str
    root_cause: str
    impact_analysis: str
    severity: str
    confidence: float
    domain_context: Optional[str] = None  # Infineon-specific context
    related_standards: List[str] = None
    mitigation_strategy: Optional[str] = None

class DiagnosticianAgent:
    """Agent responsible for root cause analysis and impact assessment"""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the Diagnostician Agent with Groq (Fast Reasoning)"""
        
        # 1. Initialize Unified LLM Client
        self.llm = LLMClient(provider="groq")
        self.use_llm = True
        
        # Initialize domain knowledge for Infineon
        self._initialize_domain_knowledge()
        
        # Severity impact mapping
        self.severity_impacts = {
            "CRITICAL": {
                "business_impact": "Can cause system failure, safety hazards, or security breaches",
                "time_to_fix": "Immediate",
                "testing_priority": "Highest"
            },
            "HIGH": {
                "business_impact": "Can cause functional failures or data corruption",
                "time_to_fix": "Within 24 hours",
                "testing_priority": "High"
            },
            "MEDIUM": {
                "business_impact": "May cause performance issues or unexpected behavior",
                "time_to_fix": "Within 1 week",
                "testing_priority": "Medium"
            },
            "LOW": {
                "business_impact": "Minor issues, code quality improvements",
                "time_to_fix": "Next release",
                "testing_priority": "Low"
            }
        }
        
        # Root cause patterns for common bugs
        self.root_cause_patterns = {
            "BUFFER_OVERFLOW": {
                "common_causes": [
                    "Off-by-one errors in loop conditions",
                    "Incorrect boundary calculations",
                    "Missing bounds checking",
                    "Using <= instead of < in loop conditions"
                ],
                "infineon_impact": "Can corrupt adjacent memory in AURIX microcontrollers, potentially affecting safety-critical data"
            },
            "MEMORY_LEAK": {
                "common_causes": [
                    "Missing free() for malloc() calls",
                    "Error paths without cleanup",
                    "Circular references in complex data structures"
                ],
                "infineon_impact": "In embedded systems, memory leaks can cause system crashes over time, violating real-time constraints"
            },
            "UNINITIALIZED_VAR": {
                "common_causes": [
                    "Variable declared but not initialized",
                    "Using stack variables without initialization",
                    "Missing default values in struct initialization"
                ],
                "infineon_impact": "Can lead to unpredictable sensor readings and calibration errors in Infineon sensor applications"
            },
            "MISSING_VOLATILE": {
                "common_causes": [
                    "Hardware register access without volatile qualifier",
                    "Compiler optimizations removing necessary reads/writes"
                ],
                "infineon_impact": "Critical for Infineon AURIX microcontrollers - can cause communication failures with peripherals"
            },
            "ISR_TIMING": {
                "common_causes": [
                    "Complex calculations in interrupt service routines",
                    "Floating-point operations in ISR",
                    "Blocking calls in ISR"
                ],
                "infineon_impact": "Violates AURIX real-time constraints (must complete within 50μs), can cause missed deadlines"
            }
        }

    async def _get_llm_diagnosis(self, finding: Dict, code_snippet: str, 
                                   librarian_context: Dict) -> Optional[BugDiagnosis]:
        """Use LLM for advanced diagnosis"""
        if not self.use_llm:
            return None
        
        try:
            # Prepare context from librarian
            context_summary = ""
            if librarian_context and librarian_context.get("relevant_documents"):
                docs = librarian_context["relevant_documents"][:2]  # Top 2 docs
                context_summary = "\n".join([
                    f"Document {i+1}: {doc.get('source', 'Unknown')} - {doc.get('content', '')[:200]}"
                    for i, doc in enumerate(docs)
                ])
            
            # Prepare prompt
            prompt = f"""
            As an Infineon embedded systems expert, diagnose this bug with detailed root cause analysis.
            
            BUG DETAILS:
            - Type: {finding.get('bug_type')}
            - Line: {finding.get('line')}
            - Severity: {finding.get('severity')}
            - Description: {finding.get('description')}
            
            CODE SNIPPET:
            ```c
            {code_snippet}
            ```
            
            RELEVANT DOCUMENTATION:
            {context_summary}
            
            Provide a comprehensive diagnosis including:
            1. Root Cause: Technical explanation of why this bug occurs
            2. Impact Analysis: Business and technical impact, especially for Infineon automotive systems
            3. Domain Context: How this relates to Infineon's AURIX microcontrollers or sensors
            4. Related Standards: Which coding standards or safety standards are violated
            5. Mitigation Strategy: Recommended approach to fix and prevent
            
            Format response as JSON with these exact keys:
            - "root_cause": string
            - "impact_analysis": string (include severity justification)
            - "domain_context": string (Infineon-specific context)
            - "related_standards": array of strings
            - "mitigation_strategy": string
            
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
                    
                    return BugDiagnosis(
                        bug_id=self._generate_bug_id(finding.get('bug_type'), finding.get('line')),
                        line_number=finding.get('line'),
                        bug_type=finding.get('bug_type'),
                        root_cause=gemini_result.get('root_cause', 'Unknown root cause'),
                        impact_analysis=gemini_result.get('impact_analysis', 'Unknown impact'),
                        severity=finding.get('severity'),
                        confidence=min(finding.get('confidence', 0.7) * 1.1, 1.0),
                        domain_context=gemini_result.get('domain_context'),
                        related_standards=gemini_result.get('related_standards', []),
                        mitigation_strategy=gemini_result.get('mitigation_strategy')
                    )
            
        except Exception as e:
            logger.error(f"LLM diagnosis failed: {e}")
        
        return None
    
    def _generate_with_retry(self, prompt: str, max_retries: int = 3):
        """
        Helper to call Gemini with automatic rate-limit handling.
        Designed to be run in a thread to avoid blocking the main event loop.
        """
        if not self.client:
            return None
            
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt
                )
                return response
            except Exception as e:
                error_str = str(e)
                # Check for Rate Limit (429) errors
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 20s, 40s, 60s
                        wait_time = (attempt + 1) * 20  
                        logger.warning(f"Diagnostician Rate Limit (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                
                # If non-retryable error, log and return None
                logger.error(f"Gemini diagnosis error: {e}")
                return None
        return None

    def _initialize_domain_knowledge(self):
        """Initialize Infineon-specific domain knowledge"""
        self.infineon_standards = {
            "MISRA-C:2012": "Mandatory for Infineon automotive projects",
            "ISO 26262": "Functional safety standard for automotive",
            "ASIL-D": "Highest Automotive Safety Integrity Level",
            "AUTOSAR": "Automotive software architecture",
            "SPICE": "Software process improvement and capability determination"
        }
        
        self.infineon_hardware = {
            "AURIX": "Infineon's 32-bit microcontroller family for automotive",
            "TC3xx": "Latest AURIX generation with safety features",
            "DPS310": "Digital pressure sensor",
            "TLI493D": "Magnetic sensor",
            "Radar": "Infineon's radar sensors for ADAS"
        }
        
        self.safety_critical_areas = [
            "Braking systems",
            "Steering control",
            "Airbag deployment",
            "Battery management",
            "ADAS (Advanced Driver Assistance Systems)"
        ]
    
    def _generate_bug_id(self, bug_type: str, line_number: int) -> str:
        """Generate a unique bug ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"BUG-{bug_type[:4]}-{line_number:04d}-{timestamp[-6:]}"
    
    def _get_rule_based_diagnosis(self, finding: Dict, librarian_context: Dict) -> BugDiagnosis:
        """Generate diagnosis using rule-based patterns"""
        
        bug_type = finding.get("bug_type", "UNKNOWN")
        line_number = finding.get("line", 0)
        
        # Get patterns for this bug type
        patterns = self.root_cause_patterns.get(bug_type, {})
        
        # Default values
        root_cause = patterns.get("common_causes", ["Unknown cause"])[0]
        infineon_impact = patterns.get("infineon_impact", "General software issue")
        
        # Enhance with librarian context if available
        domain_context = ""
        if librarian_context and librarian_context.get("relevant_documents"):
            docs = librarian_context["relevant_documents"]
            # Extract key terms from documents
            doc_keywords = set()
            for doc in docs[:2]:  # Use top 2 documents
                content = doc.get("content", "").lower()
                if "safety" in content:
                    doc_keywords.add("Safety-critical")
                if "embedded" in content:
                    doc_keywords.add("Embedded system")
                if "real-time" in content:
                    doc_keywords.add("Real-time constraints")
            
            if doc_keywords:
                domain_context = f"Context: This affects {' and '.join(doc_keywords)} systems."
        
        # Generate impact analysis based on severity
        severity = finding.get("severity", "MEDIUM")
        impact_info = self.severity_impacts.get(severity, {})
        
        impact_analysis = f"""
        Severity: {severity}
        Business Impact: {impact_info.get('business_impact', 'Unknown impact')}
        Time to Fix: {impact_info.get('time_to_fix', 'When possible')}
        Testing Priority: {impact_info.get('testing_priority', 'Medium')}
        
        Infineon-Specific Impact: {infineon_impact}
        """
        
        # Determine related standards
        related_standards = []
        if "MISRA" in finding.get("description", ""):
            related_standards.append("MISRA-C:2012")
        if bug_type in ["MISSING_VOLATILE", "ISR_TIMING"]:
            related_standards.append("Infineon Coding Guidelines")
        
        # Add general standards for embedded
        if bug_type in ["BUFFER_OVERFLOW", "MEMORY_LEAK", "UNINITIALIZED_VAR"]:
            related_standards.append("ISO 26262 (Functional Safety)")
        
        return BugDiagnosis(
            bug_id=self._generate_bug_id(bug_type, line_number),
            line_number=line_number,
            bug_type=bug_type,
            root_cause=root_cause,
            impact_analysis=impact_analysis.strip(),
            severity=severity,
            confidence=finding.get("confidence", 0.7),
            domain_context=domain_context,
            related_standards=related_standards,
            mitigation_strategy=self._get_mitigation_strategy(bug_type)
        )
    
    def _get_mitigation_strategy(self, bug_type: str) -> str:
        """Get mitigation strategy for bug type"""
        strategies = {
            "BUFFER_OVERFLOW": "Use bounds checking, validate array indices, prefer fixed-size arrays with sizeof()",
            "MEMORY_LEAK": "Always pair malloc/free, use RAII patterns in C++, consider static allocation for embedded",
            "UNINITIALIZED_VAR": "Initialize all variables at declaration, use -Wuninitialized compiler flag",
            "MISSING_VOLATILE": "Always use volatile for hardware registers, follow Infineon HAL guidelines",
            "ISR_TIMING": "Keep ISR code minimal, move processing to main loop, avoid complex operations",
            "DIVISION_BY_ZERO": "Validate divisor before division, add boundary checks",
            "NULL_DEREFERENCE": "Check pointers before dereferencing, use defensive programming"
        }
        return strategies.get(bug_type, "Review code and apply best practices for the specific bug type")
    
    def _get_code_context(self, code: str, line_number: int, context_lines: int = 3) -> str:
        """Get code context around a specific line"""
        lines = code.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context_lines.append(f"{prefix}Line {i+1}: {lines[i]}")
        
        return "\n".join(context_lines)
    
    async def diagnose_bugs(self, inspector_findings: Dict, 
                          librarian_context: Optional[Dict] = None,
                          original_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method: Diagnose bugs with root cause analysis
        
        Args:
            inspector_findings: Findings from Inspector agent
            librarian_context: Context from Librarian agent
            original_code: Original source code for context
            
        Returns:
            Dictionary with diagnosis results
        """
        logger.info(f"Diagnostician analyzing {len(inspector_findings.get('findings', []))} bugs")
        
        diagnoses = []
        stats = {
            "total_bugs": 0,
            "critical_diagnosed": 0,
            "high_diagnosed": 0,
            "infineon_context_added": 0,
            "using_gemini": self.use_llm
        }
        
        for finding in inspector_findings.get('findings', []):
            stats["total_bugs"] += 1
            
            # Get code context if available
            code_context = ""
            if original_code and finding.get('line'):
                code_context = self._get_code_context(original_code, finding['line'])
            
            # Try Gemini diagnosis first
            diagnosis = None
            if self.use_llm:
                diagnosis = await self._get_llm_diagnosis(
                    finding, 
                    code_context or finding.get('code_snippet', ''),
                    librarian_context or {}
                )
            
            # Fall back to rule-based diagnosis if Gemini failed (returned None)
            if not diagnosis:
                diagnosis = self._get_rule_based_diagnosis(
                    finding, 
                    librarian_context or {}
                )
            
            # Track stats
            if diagnosis.severity == "CRITICAL":
                stats["critical_diagnosed"] += 1
            elif diagnosis.severity == "HIGH":
                stats["high_diagnosed"] += 1
            
            if diagnosis.domain_context and "Infineon" in diagnosis.domain_context:
                stats["infineon_context_added"] += 1
            
            diagnoses.append(diagnosis)
        
        # Generate overall risk assessment
        risk_assessment = self._generate_risk_assessment(diagnoses, stats)
        
        return {
            "agent": "diagnostician",
            "status": "completed",
            "timestamp": asyncio.get_event_loop().time(),
            "total_diagnoses": len(diagnoses),
            "diagnoses": [
                {
                    "bug_id": d.bug_id,
                    "line": d.line_number,
                    "bug_type": d.bug_type,
                    "severity": d.severity,
                    "root_cause": d.root_cause,
                    "impact_analysis": d.impact_analysis,
                    "confidence": round(d.confidence, 2),
                    "domain_context": d.domain_context,
                    "related_standards": d.related_standards,
                    "mitigation_strategy": d.mitigation_strategy
                }
                for d in diagnoses
            ],
            "risk_assessment": risk_assessment,
            "stats": stats,
            "summary": self._generate_summary(diagnoses)
        }
    
    def _generate_risk_assessment(self, diagnoses: List[BugDiagnosis], stats: Dict) -> Dict:
        """Generate overall risk assessment"""
        if not diagnoses:
            return {"overall_risk": "LOW", "recommendations": ["No bugs found"]}
        
        # Calculate risk score
        severity_scores = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        total_score = sum(severity_scores.get(d.severity, 1) for d in diagnoses)
        avg_score = total_score / len(diagnoses)
        
        # Determine overall risk
        if avg_score >= 3.5:
            overall_risk = "CRITICAL"
        elif avg_score >= 2.5:
            overall_risk = "HIGH"
        elif avg_score >= 1.5:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        # Generate recommendations
        recommendations = []
        
        if stats["critical_diagnosed"] > 0:
            recommendations.append(f"🚨 IMMEDIATE ACTION: {stats['critical_diagnosed']} critical bugs require immediate fixing")
        
        if stats["high_diagnosed"] > 0:
            recommendations.append(f"⚠️ PRIORITY: {stats['high_diagnosed']} high severity bugs should be addressed within 24 hours")
        
        # Check for safety-critical issues
        safety_issues = [d for d in diagnoses if "safety" in d.impact_analysis.lower()]
        if safety_issues:
            recommendations.append("🔒 SAFETY CRITICAL: Bugs affecting functional safety - review with safety team")
        
        # Check for Infineon-specific issues
        infineon_issues = [d for d in diagnoses if d.domain_context and "Infineon" in d.domain_context]
        if infineon_issues:
            recommendations.append(f"🏭 INFINEON-SPECIFIC: {len(infineon_issues)} bugs have Infineon hardware implications")
        
        # General recommendation
        recommendations.append("✅ Run suggested fixes and retest with Infineon hardware if available")
        
        return {
            "overall_risk": overall_risk,
            "risk_score": round(avg_score, 2),
            "critical_count": stats["critical_diagnosed"],
            "high_count": stats["high_diagnosed"],
            "recommendations": recommendations,
            "next_steps": [
                "1. Address critical bugs immediately",
                "2. Review high severity bugs within 24 hours",
                "3. Update test cases for found bugs",
                "4. Consider code review for similar patterns"
            ]
        }
    
    def _generate_summary(self, diagnoses: List[BugDiagnosis]) -> Dict:
        """Generate executive summary"""
        if not diagnoses:
            return {"message": "No bugs diagnosed", "status": "CLEAN"}
        
        # Group by bug type
        bug_types = {}
        for d in diagnoses:
            bug_types[d.bug_type] = bug_types.get(d.bug_type, 0) + 1
        
        # Get most common bug type
        most_common = max(bug_types.items(), key=lambda x: x[1]) if bug_types else ("NONE", 0)
        
        # Check for patterns
        patterns = []
        if any("buffer" in d.bug_type.lower() for d in diagnoses):
            patterns.append("Memory safety issues")
        if any("memory" in d.bug_type.lower() for d in diagnoses):
            patterns.append("Memory management problems")
        if any("isr" in d.bug_type.lower() for d in diagnoses):
            patterns.append("Interrupt handling issues")
        
        return {
            "total_bugs": len(diagnoses),
            "most_common_bug": most_common[0],
            "occurrences": most_common[1],
            "detected_patterns": patterns,
            "has_safety_issues": any("safety" in str(d.impact_analysis).lower() for d in diagnoses),
            "has_infineon_context": any(d.domain_context and "Infineon" in d.domain_context for d in diagnoses),
            "status": "NEEDS_ATTENTION" if diagnoses else "CLEAN",
            "timestamp": datetime.now().isoformat()
        }


# Test function
async def test_diagnostician():
    """Test the diagnostician agent"""
    from agents.inspector import InspectorAgent
    from agents.librarian import LibrarianAgent
    
    print("\n" + "="*80)
    print("🔬 DIAGNOSTICIAN AGENT TEST")
    print("="*80)
    
    # Initialize agents
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    diagnostician = DiagnosticianAgent()
    
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
    
    print("📝 Analyzing test code...")
    
    # Run through pipeline
    librarian_result = await librarian.analyze_context(test_code, "embedded")
    inspector_result = await inspector.analyze_code(test_code, librarian_result)
    diagnostician_result = await diagnostician.diagnose_bugs(
        inspector_result, 
        librarian_result, 
        test_code
    )
    
    print(f"✅ Diagnosed {diagnostician_result['total_diagnoses']} bugs")
    
    # Show risk assessment
    risk = diagnostician_result['risk_assessment']
    print(f"\n📊 RISK ASSESSMENT: {risk['overall_risk']} (Score: {risk['risk_score']})")
    print(f"   Critical: {risk['critical_count']}, High: {risk['high_count']}")
    
    print("\n🎯 RECOMMENDATIONS:")
    for rec in risk['recommendations']:
        print(f"   • {rec}")
    
    # Show detailed diagnoses
    print("\n🔍 DETAILED DIAGNOSES:")
    for i, diagnosis in enumerate(diagnostician_result['diagnoses'][:3], 1):  # Show first 3
        print(f"\n{i}. {diagnosis['bug_id']} - Line {diagnosis['line']}")
        print(f"   Type: {diagnosis['bug_type']} [{diagnosis['severity']}]")
        print(f"   Root Cause: {diagnosis['root_cause'][:100]}...")
        
        if diagnosis.get('domain_context'):
            print(f"   Infineon Context: {diagnosis['domain_context'][:120]}...")
        
        if diagnosis.get('related_standards'):
            print(f"   Standards: {', '.join(diagnosis['related_standards'])}")
    
    # Show summary
    summary = diagnostician_result['summary']
    print(f"\n📈 SUMMARY:")
    print(f"   Total Bugs: {summary['total_bugs']}")
    print(f"   Most Common: {summary['most_common_bug']} ({summary['occurrences']}x)")
    print(f"   Safety Issues: {'Yes' if summary['has_safety_issues'] else 'No'}")
    print(f"   Status: {summary['status']}")
    
    print(f"\n{'='*80}")
    print("✅ DIAGNOSTICIAN AGENT READY!")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Run quick test by default
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        asyncio.run(test_diagnostician())
    else:
        # Quick test without external dependencies
        asyncio.run(test_diagnostician())
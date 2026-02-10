#!/usr/bin/env python3
"""
Test the complete 4-agent pipeline: Librarian → Inspector → Diagnostician → Fixer
Includes Rate Limit Safety Delays for BOTH Demo and Full modes.
"""

import asyncio
import sys
import os
import time
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.librarian import LibrarianAgent
from agents.inspector import InspectorAgent
from agents.diagnostician import DiagnosticianAgent
from agents.fixer import FixerAgent

# Force reload of .env
load_dotenv(override=True)

def safe_sleep(seconds, reason="Rate Limit Safety"):
    """Sleep with a visual countdown to prevent API blocking"""
    print(f"\n⏳ {reason}: Waiting {seconds}s...", end="", flush=True)
    for _ in range(seconds):
        time.sleep(1)
        print(".", end="", flush=True)
    print(" Done.\n")

def check_keys():
    """Verify keys are loaded (prints first 4 chars only)"""
    # Map agent names to actual environment variable names
    key_map = {
        "LIBRARIAN_KEY": "GOOGLE_API_KEY",
        "INSPECTOR_KEY": "HUGGINGFACE_API_KEY", 
        "DIAGNOSTICIAN_KEY": "GROQ_API_KEY",
        "FIXER_KEY": "GROQ_API_KEY"
    }
    print("\n🔑 KEY CHECK:")
    for agent_key, env_key in key_map.items():
        val = os.getenv(env_key)
        if val:
            print(f"  {agent_key}: ✅ FOUND ({env_key}: {val[:4]}...)")
        else:
            print(f"  {agent_key}: ❌ NOT FOUND ({env_key} missing - Using Backup)")
    print("-" * 40)

# ==========================================
# 1. FULL PIPELINE TEST (Verbose/Debug Mode)
# ==========================================
async def test_complete_pipeline():
    """Run the full pipeline with detailed logging"""
    print("\n" + "="*100)
    print("🚀 COMPLETE 4-AGENT PIPELINE TEST (DEBUG MODE)")
    print("="*100)
    
    check_keys()
    
    # Initialize all agents
    print("🔧 Initializing agents...")
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    diagnostician = DiagnosticianAgent()
    fixer = FixerAgent()
    
    # Real Infineon-style code with multiple bugs
    test_code = """
    // Infineon AURIX TC3xx Radar Processing
    #include "Ifx_Types.h"
    #include "IfxCpu.h"
    #include <stdlib.h>
    
    // BUG 1: Missing volatile for hardware register
    uint32_t* g_radarControl = (uint32_t*)0xF0000000;
    
    // BUG 2: Buffer overflow
    void read_radar_data(uint8_t* buffer, uint8_t count) {
        for(uint8_t i = 0; i <= count; i++) { // Off-by-one
            buffer[i] = read_adc_channel(i);
        }
    }
    
    // BUG 3: Memory leak
    void* create_frame(void) {
        void* frame = malloc(1024);
        return frame; // No error check, potential leak if not freed
    }
    
    // BUG 4: ISR with timing violation
    void __interrupt(0x100) void Radar_ISR(void) {
        float avg = 0.0f;
        for(int i = 0; i < 100; i++) {
            avg += read_sensor() * 0.1f; // Float in ISR
        }
    }
    """
    
    print("📊 PIPELINE EXECUTION")
    print("-"*100)
    
    # Step 1: Librarian
    print("\n1. 📚 LIBRARIAN AGENT")
    print("-"*40)
    librarian_result = await librarian.analyze_context(test_code, "embedded")
    print(f"   ✓ Keywords: {', '.join(librarian_result['keywords_found'][:5])}")
    print(f"   ✓ Documents: {len(librarian_result['relevant_documents'])} relevant")
    
    safe_sleep(5, "Cooling down API (Librarian)")

    # Step 2: Inspector
    print("\n2. 🔍 INSPECTOR AGENT")
    print("-"*40)
    inspector_result = await inspector.analyze_code(test_code, librarian_result)
    print(f"   ✓ Bugs found: {inspector_result['total_findings']}")
    print(f"   ✓ Critical: {inspector_result['metrics']['critical_count']}")
    
    safe_sleep(5, "Cooling down API (Inspector)")

    # Step 3: Diagnostician
    print("\n3. 🔬 DIAGNOSTICIAN AGENT")
    print("-"*40)
    diagnostician_result = await diagnostician.diagnose_bugs(
        inspector_result, 
        librarian_result, 
        test_code
    )
    print(f"   ✓ Diagnoses: {diagnostician_result['total_diagnoses']}")
    risk = diagnostician_result['risk_assessment']
    print(f"   ✓ Risk level: {risk['overall_risk']}")
    
    # LONGER PAUSE: Diagnostician loops through bugs, using more quota
    safe_sleep(15, "Cooling down API (Diagnostician Loop)")

    # Step 4: Fixer
    print("\n4. 🔧 FIXER AGENT")
    print("-"*40)
    fixer_result = await fixer.generate_fixes(
        diagnostician_result,
        test_code,
        librarian_result
    )
    print(f"   ✓ Fixes generated: {fixer_result['total_fixes']}")
    
    # Summary Output
    print("\n" + "="*100)
    print("🎯 RESULTS SUMMARY")
    print("="*100)
    
    if fixer_result['fixes']:
        for i, fix in enumerate(fixer_result['fixes'][:3], 1):
            print(f"\n{i}. {fix['bug_type']}")
            print(f"   Original: {fix['original'][:40]}...")
            print(f"   Fixed:    {fix['fixed'][:40]}...")
            if fix.get('safety_notes'):
                print(f"   ⚠️ Safety: {fix['safety_notes'][:80]}...")
    else:
        print("❌ No fixes generated.")

    print("\n" + "="*100)

# ==========================================
# 2. DEMO MODE (Presentation Mode)
# ==========================================
async def demo_mode():
    """Run the clean demo for hackathon presentation"""
    print("\n" + "="*80)
    print("🎭 HACKATHON DEMO: Complete Agentic Bug Hunter")
    print("="*80)
    
    check_keys()
    
    demo_code = """
    // Infineon Sensor Demo - Common Bugs
    volatile uint32_t* SENSOR = (uint32_t*)0x40021000;
    uint32_t* TIMER = (uint32_t*)0x40022000;  // Missing volatile!
    
    void read_sensor() {
        int buffer[10];
        for(int i=0; i<=10; i++) {  // Buffer overflow!
            buffer[i] = *SENSOR;
        }
    }
    
    void process_data() {
        int* data = malloc(100);
        // No free! Memory leak
    }
    """
    
    print("\n📋 INPUT CODE:")
    print("-" * 40)
    print(demo_code.strip())
    print("-" * 40)
    
    print("\n⚙️  Initializing Agents...")
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    diagnostician = DiagnosticianAgent()
    fixer = FixerAgent()
    
    print("\n1. 📚 LIBRARIAN: Context Retrieval")
    lib_result = await librarian.analyze_context(demo_code, "embedded")
    print(f"   ✓ Extracted {len(lib_result['keywords_found'])} keywords")
    
    safe_sleep(5, "API Cooldown")

    print("2. 🔍 INSPECTOR: Bug Detection")
    insp_result = await inspector.analyze_code(demo_code, lib_result)
    print(f"   ✓ Found {insp_result['total_findings']} issues")
    
    safe_sleep(5, "API Cooldown")

    print("3. 🔬 DIAGNOSTICIAN: Root Cause Analysis")
    diag_result = await diagnostician.diagnose_bugs(insp_result, lib_result, demo_code)
    print(f"   ✓ Generated {diag_result['total_diagnoses']} diagnoses")
    
    safe_sleep(15, "Processing Analysis (Heavy Load)")

    print("4. 🔧 FIXER: Automated Remediation")
    fix_result = await fixer.generate_fixes(diag_result, demo_code, lib_result)
    
    print("\n" + "="*80)
    print("✅ FINAL SOLUTIONS")
    print("="*80)
    
    if fix_result.get('fixes'):
        for i, fix in enumerate(fix_result['fixes'], 1):
            print(f"\n🔹 ISSUE #{i}: {fix['bug_type']}")
            print(f"   Explanation: {fix['explanation']}")
            print(f"   🔴 Before: {fix['original'].strip()[:50]}...")
            print(f"   🟢 After:  {fix['fixed'].strip()[:50]}...")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--full", action="store_true", help="Run full pipeline test")
    
    args = parser.parse_args()
    
    if args.full:
        asyncio.run(test_complete_pipeline())
    else:
        # Default to demo mode if no flag provided
        asyncio.run(demo_mode())
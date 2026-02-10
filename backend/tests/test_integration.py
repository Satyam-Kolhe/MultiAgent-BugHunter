#!/usr/bin/env python3
"""
Test integration between Librarian and Inspector agents
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.librarian import LibrarianAgent
from agents.inspector import InspectorAgent

async def test_full_workflow():
    """Test complete Librarian → Inspector workflow"""
    print("\n" + "="*80)
    print("🔄 AGENT INTEGRATION TEST: Librarian → Inspector")
    print("="*80)
    
    # Initialize both agents
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    
    # Realistic Infineon-style code
    test_code = """
    // AURIX TC3xx Radar Processing
    #include "Ifx_Types.h"
    #include "IfxCpu.h"
    
    // Configuration
    #define NUM_SAMPLES 256
    #define BUFFER_SIZE 512
    
    // Hardware registers
    volatile Ifx_SRC_SRCR *g_ADC_SRC;  // Interrupt source
    uint32_t* RADAR_BASE = (uint32_t*)0xF0000000;  // Missing volatile!
    
    // Data buffers
    static float range_buffer[NUM_SAMPLES];
    static float velocity_buffer[NUM_SAMPLES];
    
    // Interrupt Service Routine
    void __interrupt(0x200) void Radar_ISR(void) {
        static uint32_t isr_counter = 0;
        
        // Complex processing in ISR - violates timing constraints
        for(int i = 0; i < BUFFER_SIZE; i++) {
            // Potential buffer overflow
            range_buffer[i] = read_range_data(i);
            velocity_buffer[i] = read_velocity_data(i);
            
            // Floating point in ISR - performance issue
            float filtered = (range_buffer[i] + velocity_buffer[i]) / 2.0f;
            apply_filter(filtered);
        }
        
        isr_counter++;
        
        // Missing: Clear interrupt flag!
    }
    
    // Function with memory management issues
    RadarFrame* capture_frame(void) {
        // Allocate frame
        RadarFrame* frame = malloc(sizeof(RadarFrame));
        
        if(frame) {
            // Allocate data buffer
            frame->data = malloc(BUFFER_SIZE * sizeof(float));
            
            // No error check for frame->data!
            
            // Initialize
            for(int i = 0; i <= BUFFER_SIZE; i++) {  // Off-by-one
                frame->data[i] = 0.0f;
            }
        }
        
        // What if malloc fails? No error handling
        return frame;
    }
    
    // Recursive algorithm (MISRA violation)
    float calculate_moving_average(float* data, int size, int window) {
        if(window <= 0 || size <= 0) return 0.0f;
        
        if(window == 1) {
            return data[0];
        }
        
        // Recursive call
        float prev_avg = calculate_moving_average(data, size, window - 1);
        return (prev_avg * (window - 1) + data[window - 1]) / window;
    }
    
    // Division by zero risk
    float calculate_snr(float signal, float noise) {
        return signal / noise;  // Could divide by zero
    }
    """
    
    print("📝 Test Code Info:")
    print(f"  Lines: {len(test_code.split(chr(10)))}")
    print(f"  Characters: {len(test_code)}")
    
    print("\n" + "-"*80)
    print("📚 STEP 1: Librarian Agent (Context Analysis)")
    print("-"*80)
    
    try:
        # Run librarian
        librarian_result = await librarian.analyze_context(test_code, "embedded")
        
        print(f"✅ Librarian completed successfully")
        print(f"✅ Keywords found: {', '.join(librarian_result['keywords_found'][:5])}")
        print(f"✅ Documents retrieved: {len(librarian_result['relevant_documents'])}")
        
        # Show document types
        doc_types = {}
        for doc in librarian_result['relevant_documents']:
            doc_type = doc.get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"✅ Document types: {', '.join(f'{k}:{v}' for k, v in doc_types.items())}")
        
        if librarian_result.get('context_summary'):
            print(f"\n📋 Context Summary:")
            print(f"   {librarian_result['context_summary'][:150]}...")
    
    except Exception as e:
        print(f"❌ Librarian failed: {e}")
        return
    
    print("\n" + "-"*80)
    print("🔍 STEP 2: Inspector Agent (Bug Detection)")
    print("-"*80)
    
    try:
        # Run inspector with librarian context
        inspector_result = await inspector.analyze_code(test_code, librarian_result)
        
        print(f"✅ Inspector completed successfully")
        print(f"✅ Language detected: {inspector_result['language_detected']}")
        print(f"✅ Total findings: {inspector_result['total_findings']}")
        
        # Show metrics
        metrics = inspector_result['metrics']
        print(f"\n📊 Metrics:")
        print(f"  Critical: {metrics['critical_count']}")
        print(f"  High: {metrics['high_count']}")
        print(f"  Medium: {metrics['medium_count']}")
        print(f"  Low: {metrics['low_count']}")
        
        # Show top findings
        critical_findings = [f for f in inspector_result['findings'] 
                            if f['severity'] in ['CRITICAL', 'HIGH']]
        
        if critical_findings:
            print(f"\n🚨 CRITICAL/HIGH Findings ({len(critical_findings)}):")
            for i, finding in enumerate(critical_findings[:3], 1):
                print(f"\n{i}. Line {finding['line']}: [{finding['severity']}] {finding['bug_type']}")
                print(f"   {finding['description']}")
                print(f"   Code: {finding['code_snippet'][:60]}...")
                
                # FIX: Handle NoneType safely using (val or '')
                rule_id = finding.get('rule_id') or ''
                if rule_id.startswith('INFINEON'):
                    print(f"   ⚡ INFINEON-SPECIFIC RULE")
        
        # Show Infineon-specific findings
        infineon_findings = []
        for f in inspector_result['findings']:
            # FIX: Handle NoneType safely
            rule_id = f.get('rule_id') or ''
            if (f['bug_type'].startswith('INFINEON') or 
                'VOLATILE' in f['bug_type'] or
                'ISR' in f['bug_type'] or
                'MISRA' in rule_id):
                infineon_findings.append(f)
        
        if infineon_findings:
            print(f"\n🏭 Infineon-Specific Findings ({len(infineon_findings)}):")
            for finding in infineon_findings:
                print(f"  • Line {finding['line']}: {finding['bug_type']}")
    
    except Exception as e:
        print(f"❌ Inspector failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*80)
    print("🎯 INTEGRATION TEST RESULTS")
    print("-"*80)
    
    # Evaluate test success
    if inspector_result['total_findings'] > 0:
        print("✅ SUCCESS: Integration working!")
        print(f"✅ Agents detected {inspector_result['total_findings']} issues")
        print(f"✅ {inspector_result['metrics']['critical_count']} critical issues found")
        
        # Check for specific Infineon issues
        has_infineon_issues = any(
            f['bug_type'].startswith('INFINEON') 
            for f in inspector_result['findings']
        )
        
        if has_infineon_issues:
            print("✅ Infineon-specific rules are being detected")
        else:
            print("⚠️ No Infineon-specific rules triggered (might need more test cases)")
        
        print("\n🎯 Ready for next agent: Diagnostician")
    else:
        print("❌ FAILURE: No issues detected in test code")
        print("   Check that the inspector patterns match the test code")
    
    print("\n" + "="*80)

async def quick_test():
    """Quick test for development"""
    print("🚀 Quick Integration Test")
    
    # Simple test
    test_code = """
    void test() {
        int buf[10];
        for(int i=0; i<=10; i++) buf[i]=0;
    }
    """
    
    librarian = LibrarianAgent()
    inspector = InspectorAgent()
    
    lib_result = await librarian.analyze_context(test_code, "embedded")
    print(f"Librarian: {len(lib_result['keywords_found'])} keywords")
    
    insp_result = await inspector.analyze_code(test_code, lib_result)
    print(f"Inspector: {insp_result['total_findings']} findings")
    
    if insp_result['findings']:
        print("First finding:", insp_result['findings'][0]['description'])

if __name__ == "__main__":
    # Run quick test or full test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test agent integration")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(quick_test())
    else:
        asyncio.run(test_full_workflow())
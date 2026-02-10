#!/usr/bin/env python3
"""
Test script for Inspector Agent
Run this to verify the inspector is working correctly
"""

import asyncio
import sys
import os

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.inspector import InspectorAgent

async def test_basic_inspector():
    """Test inspector with basic embedded C code"""
    print("🧪 TEST 1: Basic Inspector (Rule-Based Only)")
    print("="*60)
    
    # Initialize inspector without API key (uses rule-based analysis)
    inspector = InspectorAgent()
    
    # Simple test code with obvious bugs
    test_code = """
    #include <stdio.h>
    
    void dangerous_function() {
        int buffer[10];
        // Off-by-one error
        for(int i = 0; i <= 10; i++) {
            buffer[i] = i * 2;
        }
        
        // Uninitialized variable
        int result;
        result = result + 5;
    }
    
    int main() {
        dangerous_function();
        return 0;
    }
    """
    
    result = await inspector.analyze_code(test_code)
    
    print(f"✅ Language detected: {result['language_detected']}")
    print(f"✅ Total findings: {result['total_findings']}")
    print(f"✅ Using Gemini: {result['analysis_method'] == 'gemini_enhanced'}")
    
    if result['findings']:
        print("\n📋 Findings found:")
        for i, finding in enumerate(result['findings'][:3], 1):
            print(f"\n{i}. Line {finding['line']}: [{finding['severity']}] {finding['bug_type']}")
            print(f"   Description: {finding['description']}")
            print(f"   Code: {finding['code_snippet'][:50]}...")
    else:
        print("\n❌ No findings detected - something is wrong!")
    
    return result['findings']

async def test_infineon_specific():
    """Test with Infineon-specific code patterns"""
    print("\n\n🧪 TEST 2: Infineon-Specific Patterns")
    print("="*60)
    
    inspector = InspectorAgent()
    
    # Infineon-specific test code
    test_code = """
    // AURIX TC3xx Example
    #include <stdint.h>
    
    // Missing volatile for hardware register (Infineon rule)
    uint32_t* SENSOR_CTRL = (uint32_t*)0xF0001000;
    
    // Correct with volatile
    volatile uint32_t* TIMER_CNT = (uint32_t*)0xF0002000;
    
    // ISR with complex operations (bad practice)
    void __attribute__((interrupt)) ADC_ISR(void) {
        float reading = read_adc_float();  // Float in ISR
        process_reading(reading);
    }
    
    // Buffer overflow in sensor data
    void collect_sensor_data(uint8_t* buffer, uint8_t count) {
        for(uint8_t i = 0; i <= count; i++) {
            buffer[i] = read_sensor();
        }
    }
    
    // Memory leak
    void process_frame() {
        uint8_t* frame = malloc(1024);
        // No free!
    }
    """
    
    result = await inspector.analyze_code(test_code)
    
    print(f"✅ Language detected: {result['language_detected']}")
    
    # Look for Infineon-specific findings
    infineon_findings = [f for f in result['findings'] 
                         if f['bug_type'].startswith('INFINEON') or 
                            'VOLATILE' in f['bug_type'] or
                            'ISR' in f['bug_type']]
    
    if infineon_findings:
        print(f"✅ Found {len(infineon_findings)} Infineon-specific issues")
        for finding in infineon_findings:
            print(f"\n• Line {finding['line']}: {finding['description']}")
    else:
        print("⚠️ No Infineon-specific findings (might need to adjust patterns)")
    
    # Show all findings
    print(f"\n📊 Total findings: {result['total_findings']}")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = sum(1 for f in result['findings'] if f['severity'] == severity)
        if count > 0:
            print(f"  {severity}: {count}")
    
    return result['findings']

async def test_with_gemini():
    """Test with Gemini API if available"""
    print("\n\n🧪 TEST 3: Gemini-Enhanced Analysis")
    print("="*60)
    
    # Try to get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("⚠️ No GOOGLE_API_KEY found in environment")
        print("  Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("  Or add to .env file in backend folder")
        return []
    
    print("🔄 Testing with Gemini API...")
    
    try:
        inspector = InspectorAgent(gemini_api_key=api_key)
        
        # Complex test case
        test_code = """
        // Complex embedded code for Gemini to analyze
        #include <stdlib.h>
        #include <string.h>
        
        typedef struct {
            float x;
            float y;
            float z;
        } SensorData;
        
        SensorData* create_sensor_array(int count) {
            // Allocate memory
            SensorData* array = malloc(count * sizeof(SensorData));
            
            // Initialize with zeros
            for(int i = 0; i < count; i++) {
                array[i].x = 0.0f;
                array[i].y = 0.0f;
                array[i].z = 0.0f;
            }
            
            return array;
        }
        
        void process_sensor_frame(SensorData* frame, int size) {
            // Potential issues for Gemini to find:
            // 1. No null check on frame
            // 2. Could overflow if size is wrong
            // 3. Floating point operations might not be safe
            for(int i = 0; i < size; i++) {
                frame[i].x = frame[i].x * 1.05f + 0.1f;
                frame[i].y = frame[i].y / calculate_factor(i);  // Division risk
                frame[i].z = sqrt(frame[i].z * frame[i].z);  // Math library needed
            }
        }
        
        float calculate_factor(int index) {
            if(index == 0) return 0.0f;  // Division by zero risk
            return (float)index / 100.0f;
        }
        """
        
        result = await inspector.analyze_code(test_code)
        
        print(f"✅ Gemini analysis complete!")
        print(f"✅ Total findings: {result['total_findings']}")
        
        if result['findings']:
            print("\n🔍 Gemini-specific findings:")
            gemini_findings = [f for f in result['findings'] 
                              if f.get('suggested_fix')]  # Gemini provides fixes
            for i, finding in enumerate(gemini_findings[:2], 1):
                print(f"\n{i}. Line {finding['line']}: {finding['bug_type']}")
                print(f"   {finding['description']}")
                if finding.get('suggested_fix'):
                    print(f"   💡 Fix: {finding['suggested_fix']}")
        
        return result['findings']
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return []

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧠 INSPECTOR AGENT TEST SUITE")
    print("="*80)
    
    all_findings = []
    
    # Test 1: Basic inspector
    findings1 = await test_basic_inspector()
    all_findings.extend(findings1)
    
    # Test 2: Infineon-specific
    findings2 = await test_infineon_specific()
    all_findings.extend(findings2)
    
    # Test 3: Gemini (optional)
    findings3 = await test_with_gemini()
    all_findings.extend(findings3)
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    total_tests = sum(1 for f in [findings1, findings2, findings3] if f)
    print(f"Tests completed: {total_tests}/3")
    print(f"Total bugs detected across all tests: {len(all_findings)}")
    
    # Group by severity
    severities = {}
    for finding in all_findings:
        severity = finding['severity']
        severities[severity] = severities.get(severity, 0) + 1
    
    print("\nSeverity breakdown:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        count = severities.get(severity, 0)
        if count > 0:
            print(f"  {severity}: {count}")
    
    # Check if inspector is working
    if len(all_findings) > 0:
        print("\n✅ SUCCESS: Inspector agent is detecting bugs!")
        print("🎯 Ready for integration with other agents.")
    else:
        print("\n❌ WARNING: No bugs detected. Check inspector patterns.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
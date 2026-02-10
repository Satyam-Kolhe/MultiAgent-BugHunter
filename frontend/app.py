"""
Streamlit Frontend for Infineon Bug Hunter
Multi-Agent System UI for Code Analysis
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any
import os

# Page configuration
st.set_page_config(
    page_title="Infineon Bug Hunter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .bug-card {
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 0.5rem 0;
        background-color: #fff5f5;
    }
    .fix-card {
        padding: 1rem;
        border-left: 4px solid #44ff44;
        margin: 0.5rem 0;
        background-color: #f5fff5;
    }
    .severity-critical { color: #d32f2f; font-weight: bold; }
    .severity-high { color: #f57c00; font-weight: bold; }
    .severity-medium { color: #fbc02d; font-weight: bold; }
    .severity-low { color: #388e3c; }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_code(code: str, context_type: str = "embedded", include_fixes: bool = True) -> Optional[Dict[str, Any]]:
    """Send code to API for analysis"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/analyze",
            json={
                "code": code,
                "context_type": context_type,
                "include_fixes": include_fixes
            },
            timeout=300  # 5 minutes timeout for analysis
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def analyze_file(file, context_type: str = "embedded", include_fixes: bool = True) -> Optional[Dict[str, Any]]:
    """Upload file to API for analysis"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {
            "context_type": context_type,
            "include_fixes": str(include_fixes).lower()
        }
        response = requests.post(
            f"{API_BASE_URL}/api/analyze/file",
            files=files,
            data=data,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_results(result: Dict[str, Any]):
    """Display analysis results"""
    if result.get("status") != "success":
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Execution time
    exec_time = result.get("execution_time", 0)
    st.success(f"✅ Analysis completed in {exec_time:.2f} seconds")
    
    # Librarian Results
    librarian = result.get("librarian_result", {})
    with st.expander("📚 Librarian: Context & Documentation", expanded=False):
        if librarian.get("keywords_found"):
            st.write("**Keywords Found:**", ", ".join(librarian["keywords_found"][:10]))
        if librarian.get("context_summary"):
            st.write("**Context Summary:**", librarian["context_summary"])
        if librarian.get("relevant_documents"):
            st.write(f"**Relevant Documents:** {len(librarian['relevant_documents'])} found")
            for doc in librarian["relevant_documents"][:3]:
                st.write(f"- {doc.get('source', 'Unknown')} (relevance: {doc.get('relevance_score', 0):.2f})")
    
    # Inspector Results
    inspector = result.get("inspector_result", {})
    with st.expander(f"🔍 Inspector: Bug Detection ({inspector.get('total_findings', 0)} issues found)", expanded=True):
        metrics = inspector.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Critical", metrics.get("critical_count", 0))
        with col2:
            st.metric("High", metrics.get("high_count", 0))
        with col3:
            st.metric("Medium", metrics.get("medium_count", 0))
        with col4:
            st.metric("Low", metrics.get("low_count", 0))
        
        findings = inspector.get("findings", [])
        if findings:
            for finding in findings:
                severity = finding.get("severity", "UNKNOWN")
                severity_class = f"severity-{severity.lower()}"
                st.markdown(f"""
                <div class="bug-card">
                    <p><span class="{severity_class}">[{severity}]</span> <strong>{finding.get('bug_type', 'Unknown')}</strong> - Line {finding.get('line', '?')}</p>
                    <p>{finding.get('description', 'No description')}</p>
                    <pre>{finding.get('code_snippet', '')}</pre>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No bugs found! 🎉")
    
    # Diagnostician Results
    diagnostician = result.get("diagnostician_result")
    if diagnostician:
        with st.expander("🔬 Diagnostician: Root Cause Analysis", expanded=False):
            diagnoses = diagnostician.get("diagnoses", [])
            for diagnosis in diagnoses:
                st.write(f"**{diagnosis.get('bug_type', 'Unknown')}** (Line {diagnosis.get('line', '?')})")
                st.write(f"**Root Cause:** {diagnosis.get('root_cause', 'Unknown')}")
                st.write(f"**Impact:** {diagnosis.get('impact_analysis', 'Unknown')[:200]}...")
                st.divider()
    
    # Fixer Results
    fixer = result.get("fixer_result")
    if fixer:
        with st.expander("🔧 Fixer: Automated Fixes", expanded=True):
            fixes = fixer.get("fixes", [])
            if fixes:
                for i, fix in enumerate(fixes, 1):
                    st.markdown(f"""
                    <div class="fix-card">
                        <h4>Fix #{i}: {fix.get('bug_type', 'Unknown')}</h4>
                        <p><strong>Explanation:</strong> {fix.get('explanation', 'No explanation')}</p>
                        <p><strong>Before:</strong></p>
                        <pre style="background-color: #ffe0e0; padding: 0.5rem; border-radius: 0.25rem;">{fix.get('original', '')}</pre>
                        <p><strong>After:</strong></p>
                        <pre style="background-color: #e0ffe0; padding: 0.5rem; border-radius: 0.25rem;">{fix.get('fixed', '')}</pre>
                        {f"<p><strong>Safety Notes:</strong> {fix.get('safety_notes', '')}</p>" if fix.get('safety_notes') else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No fixes generated")

# Main UI
def main():
    st.markdown('<div class="main-header">🔍 Infineon Bug Hunter</div>', unsafe_allow_html=True)
    st.markdown("**Multi-Agent System for Bug Detection and Fixing in Embedded Code**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Health Check
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info(f"Make sure the backend is running at: {API_BASE_URL}")
            st.stop()
        
        st.divider()
        
        context_type = st.selectbox(
            "Context Type",
            ["embedded", "c", "cpp", "python"],
            help="Type of code being analyzed"
        )
        
        include_fixes = st.checkbox(
            "Generate Fixes",
            value=True,
            help="Whether to generate automated fixes"
        )
        
        st.divider()
        st.markdown("### 📊 System Status")
        try:
            stats_response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                agents = stats.get("agents", {})
                for agent, ready in agents.items():
                    status = "✅" if ready else "❌"
                    st.write(f"{status} {agent.capitalize()}")
        except:
            st.warning("Could not fetch system status")
    
    # Main Content
    tab1, tab2 = st.tabs(["📝 Code Input", "📁 File Upload"])
    
    with tab1:
        st.header("Paste Your Code")
        code_input = st.text_area(
            "Enter code to analyze",
            height=400,
            placeholder="// Paste your code here...\nvoid read_sensor() {\n    // Your code\n}",
            help="Paste your C/C++/Python code here for analysis"
        )
        
        analyze_button = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
        
        if analyze_button:
            if not code_input.strip():
                st.warning("Please enter some code to analyze")
            else:
                with st.spinner("Analyzing code... This may take a minute."):
                    result = analyze_code(code_input, context_type, include_fixes)
                    if result:
                        display_results(result)
    
    with tab2:
        st.header("Upload Code File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["c", "cpp", "h", "hpp", "py", "txt", "md", "pdf", "docx"],
            help="Upload a code file or document containing code"
        )
        
        if uploaded_file:
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Show file preview for text files
            if uploaded_file.name.endswith(('.c', '.cpp', '.h', '.hpp', '.py', '.txt', '.md')):
                try:
                    content = uploaded_file.getvalue().decode('utf-8')
                    st.code(content[:500] + ("..." if len(content) > 500 else ""), language="c")
                except:
                    st.warning("Could not preview file content")
            
            analyze_file_button = st.button("🔍 Analyze File", type="primary", use_container_width=True)
            
            if analyze_file_button:
                with st.spinner("Processing file and analyzing code... This may take a minute."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = analyze_file(uploaded_file, context_type, include_fixes)
                    if result:
                        display_results(result)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Infineon Bug Hunter - Multi-Agent System</p>
        <p>Powered by: Librarian 📚 | Inspector 🔍 | Diagnostician 🔬 | Fixer 🔧</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

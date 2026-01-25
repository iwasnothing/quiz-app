#!/usr/bin/env python3
"""
Debug script to test topic extraction and see what's happening.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from llm_prompts import llm, topic_parser, TOPIC_EXTRACTION_PROMPT
from config import settings
from ingest import extract_topic_metadata

def test_llm_connection():
    """Test if LLM is accessible."""
    print("=" * 80)
    print("Testing LLM Connection")
    print("=" * 80)
    print()
    print(f"LLM Base URL: {settings.LLM_BASE_URL}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"LLM API Key: {'SET' if settings.LLM_API_KEY and settings.LLM_API_KEY != 'EMPTY' else 'NOT SET or EMPTY'}")
    print()
    
    try:
        print("Testing simple LLM call...")
        response = llm.invoke("Say 'hello' if you can read this.")
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"✅ LLM Response: {response_text}")
        
        # Check if thinking mode is disabled
        if len(response_text) > 100 and "let me" in response_text.lower() or "hmm" in response_text.lower() or "think" in response_text.lower():
            print("\n⚠️  WARNING: Thinking mode appears to still be active (response contains reasoning)")
            print("   However, the JSON extraction fallback will handle this.")
        else:
            print("\n✅ Thinking mode appears to be disabled (clean response)")
        
        return True
    except Exception as e:
        print(f"❌ LLM Connection Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topic_extraction():
    """Test topic extraction with a sample chunk."""
    print()
    print("=" * 80)
    print("Testing Topic Extraction")
    print("=" * 80)
    print()
    
    sample_chunk = """
    Quantum mechanics is a fundamental theory in physics that describes the physical properties 
    of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum 
    physics including quantum chemistry, quantum field theory, quantum technology, and quantum 
    information science. Key concepts include wave-particle duality, uncertainty principle, 
    and quantum entanglement.
    """
    
    print("Sample chunk:")
    print("-" * 80)
    print(sample_chunk[:200] + "...")
    print("-" * 80)
    print()
    
    try:
        # Use the actual extraction function which has the JSON extraction fallback
        print("Using extract_topic_metadata() function (with JSON extraction fallback)...")
        print("Note: This function handles thinking model output by extracting JSON from the response.")
        result = extract_topic_metadata(sample_chunk)
        
        print(f"\n✅ Extraction Result: {result}")
        print(f"   Type: {type(result)}")
        print(f"   Topic: {result.get('topic_name', 'NOT FOUND')}")
        print(f"   Concepts: {result.get('sub_concepts', 'NOT FOUND')}")
        
        if result.get('topic_name') == 'Unknown':
            print("\n⚠️  WARNING: Topic is 'Unknown' - extraction may have failed!")
        if not result.get('sub_concepts'):
            print("\n⚠️  WARNING: No concepts extracted!")
        else:
            print(f"\n✅ Successfully extracted {len(result.get('sub_concepts', []))} concepts!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Topic Extraction Debug Tool")
    print("=" * 80)
    print()
    
    # Test LLM connection first
    if not test_llm_connection():
        print("\n❌ Cannot proceed - LLM is not accessible")
        sys.exit(1)
    
    # Test topic extraction
    result = test_topic_extraction()
    
    if result and result.get('topic_name') != 'Unknown' and result.get('sub_concepts'):
        print("\n✅ Extraction is working correctly!")
    else:
        print("\n❌ Extraction is not working - check LLM configuration and prompts")

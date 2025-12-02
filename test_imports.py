#!/usr/bin/env python
"""
Simple test to verify the agent import works
Run this first before the full test_agent.py
"""

print("Testing agent imports...")
print("=" * 60)

try:
    print("1. Testing langchain.agents imports...")
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    print("   ✓ AgentExecutor imported")
    print("   ✓ create_tool_calling_agent imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nFix: pip install langchain")
    exit(1)

try:
    print("\n2. Testing langchain_openai imports...")
    from langchain_openai import ChatOpenAI
    print("   ✓ ChatOpenAI imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nFix: pip install langchain-openai")
    exit(1)

try:
    print("\n3. Testing langchain_core imports...")
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    print("   ✓ ChatPromptTemplate imported")
    print("   ✓ MessagesPlaceholder imported")
    print("   ✓ tool decorator imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nFix: pip install langchain-core")
    exit(1)

try:
    print("\n4. Testing basic dependencies...")
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    print("   ✓ requests imported")
    print("   ✓ BeautifulSoup imported")
    print("   ✓ dotenv imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nFix: pip install requests beautifulsoup4 python-dotenv")
    exit(1)

try:
    print("\n5. Testing agent module import...")
    from scraper.agent_classic import create_agentic_rag
    print("   ✓ scraper.agent imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print(f"\nError details: {type(e).__name__}: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS! All imports working correctly.")
print("\nYou can now run:")
print("  python test_agent.py         # Full diagnostic")
print("  python -c \"from scraper.agent_classic import create_agentic_rag; agent = create_agentic_rag()\"")
print("=" * 60)

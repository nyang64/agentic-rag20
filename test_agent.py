#!/usr/bin/env python
"""
Test script for Agentic RAG 2.0
Run this to diagnose and test your agent setup
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    print("=" * 60)
    print("1. CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    required_vars = [
        "OPENROUTER_API_KEY",
        "DEEPSEEK_FREE_MODEL",
        "PGVECTOR_DB_URL",
    ]
    
    optional_vars = [
        "BRAVE_API_KEY",
    ]
    
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"✓ {var}: {masked}")
        else:
            print(f"✗ {var}: NOT SET")
            all_good = False
    
    print("\nOptional:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"✓ {var}: {masked}")
        else:
            print(f"  {var}: NOT SET (will use DuckDuckGo)")
    
    return all_good

def check_imports():
    """Check if all required packages are installed"""
    print("\n" + "=" * 60)
    print("2. CHECKING PACKAGE IMPORTS")
    print("=" * 60)
    
    packages = [
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_core", "langchain-core"),
        ("langchain_community", "langchain-community"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("bs4", "beautifulsoup4"),
    ]
    
    all_good = True
    for module, package in packages:
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            all_good = False
    
    return all_good

def test_llm_connection():
    """Test if we can connect to the LLM"""
    print("\n" + "=" * 60)
    print("3. TESTING LLM CONNECTION")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
        )
        
        print("Sending test message to LLM...")
        response = llm.invoke("Say 'Hello, I am working!'")
        print(f"✓ LLM Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ LLM Connection Failed: {str(e)}")
        return False

def test_web_search():
    """Test web search functionality"""
    print("\n" + "=" * 60)
    print("4. TESTING WEB SEARCH")
    print("=" * 60)
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        query = "test"
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        print("Sending test search to DuckDuckGo...")
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result__body')
        
        print(f"✓ Web Search Working: Found {len(results)} results")
        return True
    except Exception as e:
        print(f"✗ Web Search Failed: {str(e)}")
        return False

def test_vector_db():
    """Test vector database connection"""
    print("\n" + "=" * 60)
    print("5. TESTING VECTOR DATABASE")
    print("=" * 60)
    
    try:
        from scraper.raq_query import retrieve_top3
        
        print("Testing vector database query...")
        docs = retrieve_top3("test query")
        print(f"✓ Vector DB Working: Retrieved {len(docs)} documents")
        return True
    except Exception as e:
        print(f"✗ Vector DB Failed: {str(e)}")
        print("  (This is OK if you haven't scraped any data yet)")
        return False

def test_agent_import():
    """Test if agent can be imported"""
    print("\n" + "=" * 60)
    print("6. TESTING AGENT IMPORT")
    print("=" * 60)
    
    try:
        print("Attempting to import from agent_classic.py...")
        from scraper.agent_classic import create_agentic_rag
        print("✓ agent_classic.py import successful")
        return True, "agent_classic"
    except Exception as e:
        print(f"✗ agent_classic.py import failed: {str(e)}")
        
        try:
            print("\nAttempting to import from agent_simple.py...")
            from scraper.agent_simple import create_agentic_rag
            print("✓ agent_simple.py import successful")
            return True, "agent_simple"
        except Exception as e2:
            print(f"✗ agent_simple.py import failed: {str(e2)}")
            return False, None

def test_agent_execution(module_name):
    """Test running the agent"""
    print("\n" + "=" * 60)
    print("7. TESTING AGENT EXECUTION")
    print("=" * 60)
    
    try:
        if module_name == "agent_classic":
            from scraper.agent_classic import create_agentic_rag
        else:
            from scraper.agent_simple import create_agentic_rag
        
        print(f"Creating agent from {module_name}.py...")
        agent = create_agentic_rag()
        
        print("Running test query: 'What is the weather like in New York today?'")
        result = agent.invoke({"input": "What is the weather like in New York today? You can do a search and give an answer."})
        
        print("\n" + "-" * 60)
        print("AGENT OUTPUT:")
        print("-" * 60)
        print(result.get("output", result))
        print("-" * 60)
        
        print("\n✓ Agent execution successful!")
        return True
    except Exception as e:
        print(f"✗ Agent execution failed: {str(e)}")
        import traceback
        print("\nFull error:")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AGENTIC RAG 2.0 - DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Run checks
    env_ok = check_environment()
    imports_ok = check_imports()
    
    if not env_ok or not imports_ok:
        print("\n❌ CRITICAL: Environment or imports missing. Please fix above issues first.")
        sys.exit(1)
    
    llm_ok = test_llm_connection()
    web_ok = test_web_search()
    db_ok = test_vector_db()
    agent_ok, module = test_agent_import()
    
    if not llm_ok:
        print("\n❌ CRITICAL: Cannot connect to LLM. Check your API key.")
        sys.exit(1)
    
    if not agent_ok:
        print("\n❌ CRITICAL: Cannot import agent. Check error messages above.")
        sys.exit(1)
    
    # Run agent test
    exec_ok = test_agent_execution(module)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Environment Variables: {'✓' if env_ok else '✗'}")
    print(f"Package Imports: {'✓' if imports_ok else '✗'}")
    print(f"LLM Connection: {'✓' if llm_ok else '✗'}")
    print(f"Web Search: {'✓' if web_ok else '✗'}")
    print(f"Vector Database: {'✓' if db_ok else '⚠ (optional)'}")
    print(f"Agent Import: {'✓' if agent_ok else '✗'}")
    print(f"Agent Execution: {'✓' if exec_ok else '✗'}")
    
    if exec_ok:
        print("\n🎉 SUCCESS! Your Agentic RAG 2.0 is working!")
        print(f"\nYou can now use: from scraper.{module} import create_agentic_rag")
    else:
        print("\n❌ Some tests failed. Review the errors above.")

if __name__ == "__main__":
    main()

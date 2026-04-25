"""
evaluate.py
-----------
Simple evaluation script for the Agentic RAG system.
Runs a set of test cases and checks if the agent's response meets basic criteria.
"""

import json
import time
from agent import run_agent

# Test cases: (question, expected_keywords, expected_status)
TEST_CASES = [
    {
        "name": "Structured Lookup (Infosys Revenue)",
        "question": "What was Infosys revenue in FY24?",
        "expected_keywords": ["18.56", "billion"],
        "expected_status": "ok"
    },
    {
        "name": "Structured Comparison (Headcount)",
        "question": "Which company had the highest headcount in FY24?",
        "expected_keywords": ["Accenture"],
        "expected_status": "ok"
    },
    {
        "name": "Unstructured Retrieval (Risk Factors)",
        "question": "What are some risk factors for Accenture according to its documents?",
        "expected_keywords": ["risk", "Accenture"],
        "expected_status": "ok"
    },
    {
        "name": "Web Search (Stock Price)",
        "question": "What is the current stock price of Cognizant?",
        "expected_keywords": ["CTSH"],
        "expected_status": "ok"
    },
    {
        "name": "Gatekeeping (Trivial)",
        "question": "Hello there!",
        "expected_keywords": ["hello", "how can I help"],
        "expected_status": "ok"
    },
    {
        "name": "Gatekeeping (Out of Scope)",
        "question": "Who won the cricket world cup in 2023?",
        "expected_keywords": ["sorry", "cannot", "financial"],
        "expected_status": "ok"
    }
]

def run_evaluation():
    print("=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    
    passed = 0
    total = len(TEST_CASES)
    
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] Running: {test['name']}")
        print(f"Question: {test['question']}")
        
        start_time = time.time()
        try:
            result = run_agent(test['question'], verbose=False)
            latency = (time.time() - start_time) * 1000
            
            answer = result.final_answer.lower()
            
            # Check keywords (case-insensitive)
            missing = [kw for kw in test['expected_keywords'] if kw.lower() not in answer]
            
            status_match = result.status == test['expected_status']
            
            is_pass = not missing and status_match
            
            if is_pass:
                print(f"✅ PASS ({latency:.0f}ms)")
                passed += 1
            else:
                print(f"❌ FAIL ({latency:.0f}ms)")
                if missing:
                    print(f"   Missing keywords: {missing}")
                if not status_match:
                    print(f"   Status mismatch: expected {test['expected_status']}, got {result.status}")
                print(f"   Answer: {result.final_answer[:100]}...")

            results.append({
                "test": test['name'],
                "status": "PASS" if is_pass else "FAIL",
                "latency_ms": latency,
                "steps": result.steps_used
            })

        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({
                "test": test['name'],
                "status": "ERROR",
                "error": str(e)
            })

    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE: {passed}/{total} Passed")
    print("=" * 60)
    
    # Optional: save to file
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()

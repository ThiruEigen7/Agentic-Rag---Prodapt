# Rate Limit Handling

## Problem
Gemini API has rate limits, especially on free tier:
- **Free tier**: 15 requests per minute (RPM) across all models
- **Response**: 503 UNAVAILABLE, 429 Too Many Requests, or "overloaded" errors

## Solution: Exponential Backoff with Retry

Both `tools/query_data.py` and `agent.py` now include automatic retry logic:

### Retry Strategy
1. **Detect rate limit errors**: Check for "503", "429", "UNAVAILABLE", or "overloaded" in error message
2. **Exponential backoff**: Wait `2^attempt` seconds before retrying
   - Attempt 1: Wait 1 second (2^0)
   - Attempt 2: Wait 2 seconds (2^1)
   - Attempt 3: Wait 4 seconds (2^2)
3. **Max retries**: 3 attempts total (configurable via `max_retries` parameter)
4. **Non-rate-limit errors**: Fail immediately (don't retry)

### Implementation Details

#### tools/query_data.py
```python
def _generate_sql(question: str, max_retries: int = 3) -> str:
    """Includes retry logic with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(...)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(code in error_msg for code in ["503", "429", "UNAVAILABLE", "overloaded"])
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise e
```

#### agent.py
```python
def _llm(prompt: str, max_retries: int = 3) -> str:
    """Same retry logic for main LLM calls"""
    
def _llm_json(prompt: str, max_retries: int = 3) -> dict:
    """Same retry logic for JSON-formatted responses"""
```

## Usage Tips

### For Rapid Requests
If you're testing multiple queries in quick succession:
- Space them out by 10+ seconds
- Or use a single query that covers multiple questions
- Or wait for the retry logic to complete

### Example: Automatic Retry in Action
```bash
$ python tools/query_data.py "What was Infosys revenue in FY24?"
Question : What was Infosys revenue in FY24?
────────────────────────────────────────────────────────────
⏳ Rate limited. Retrying in 1s (attempt 1/3)...
⏳ Rate limited. Retrying in 2s (attempt 2/3)...
SQL      : SELECT company, fiscal_year, revenue_bn_USD FROM financials WHERE ...
Source   : financials.db — Infosys | FY24 (5 rows)
Rows     : 5

company fiscal_year  revenue_bn_USD
Infosys        FY24           4.617
...
```

### Monitor Retries
Check stderr for retry messages:
```bash
python tools/query_data.py "Your question" 2>&1
```

## API Quotas (Free Tier)
- **15 RPM** (requests per minute) across all models
- **1,000,000 tokens/day** for text generation
- **Shared across**: All google.generativeai calls in your projects

### Staying Within Limits
1. **Space requests**: 1 request every 4 seconds = 15 RPM maximum
2. **Batch questions**: Ask multi-part questions instead of sequential queries
3. **Cache prompts**: Use prompt caching for repeated schema queries (planned feature)
4. **Upgrade tier**: Paid API tiers have higher limits

## Configuration

To adjust retry behavior, edit the `max_retries` parameter:

### Increase retries (more patience):
```python
result = query_data("Your question", max_retries=5)  # Up to 5 attempts
```

### Decrease retries (fail faster):
```python
result = query_data("Your question", max_retries=2)  # Only 2 attempts
```

### Disable retries (not recommended):
```python
result = query_data("Your question", max_retries=1)  # No retry
```

## Monitoring

The system prints retry attempts to stderr, so you can see what's happening:
- ✅ Success on first attempt: Silent
- ⏳ Rate limited: Shows retry message
- ❌ Failed after max retries: Raises exception

## Future Improvements
- [ ] Prompt caching (reduce request count for repeated queries)
- [ ] Request batching (combine multiple questions)
- [ ] Smart rate limit detection (pre-emptive throttling)
- [ ] Fallback to local models (if primary API fails)

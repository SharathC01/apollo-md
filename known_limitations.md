## 3. Quote Verification Rate: 69% (247/360 records)

**Issue:** 113 records have source_quote values that cannot be found 
verbatim in the extracted PDF text after whitespace and typography 
normalization. These are marked quote_verified=False and confidence 
tier "unverified" in the evidence table.

**Root causes identified:**
- LLM slightly paraphrases quotes despite verbatim instruction
- Cross-page sentence splits not captured by section-aware ingest
- Multi-column reflow artifacts in two-column PDFs

**Current mitigation:**
- quote_verified flag shown explicitly in UI (✓/✗/?)
- Unverified records still displayed but visually distinguished
- Confidence tier downgraded for unverified records

**Fix if time allows:**
- Stricter extraction prompt: require quote to be max 15 words 
  (shorter quotes are easier to verify verbatim)
- Fuzzy matching in verify_quote() with threshold >0.95 similarity
  instead of exact substring match

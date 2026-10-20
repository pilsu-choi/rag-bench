---
type: Lesson
title: Run-on text makes oversized chunks
description: Sentence-based packing left a 94k-character chunk (tables without punctuation), which broke embedding after 15 minutes of work.
tags: [lesson, chunking, embedding]
timestamp: 2026-10-20T09:15:00Z
resource: ../../rag_bench/prep.py
---

# Run-on text makes oversized chunks

**What happened.** The chunker split articles into paragraphs and sentences, then packed them into
chunks of up to 500 characters. Some Wikipedia text, such as lists and tables flattened without
punctuation, is one "sentence" that runs for tens of thousands of characters. One distractor chunk
had 94,353 characters. It exceeded bge-m3's 8,192-token limit, so vLLM returned an error with no
`data` field and the embedding run died after about 900 of 1,563 batches. Nothing had been saved.

**Fix.**
- Units longer than 500 characters are now hard-split, so the longest chunk is 501 characters.
- `embed()` sends `truncate_prompt_tokens: 8192` as a safety net and raises with the server's
  message.
- The fix changed 43 chunks in 12 base documents. One synthetic query pointed at a changed chunk and
  was dropped, leaving 598 queries.

**Takeaway.**
- Check the chunk length distribution (max, not only the mean) before a long embedding job.
- Save long jobs per batch or per shard so that one failure doesn't lose all the work.

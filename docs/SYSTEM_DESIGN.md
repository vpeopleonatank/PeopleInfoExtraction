# 1) High-level approach (anti-hallucination by design)

1. **Segment → Extract → Verify → Link → Upsert**

   * **Segment:** split article into passages (e.g., 512–1,200 tokens) with sentence boundaries.
   * **Extract:** run **hybrid extractors**: (a) deterministic (regex for phones, dates, Vietnamese national IDs, license plates), (b) NER (small task models), (c) LLM for hard stuff (roles, “what they did”, relationships) but under **strict constraints**.
   * **Verify:** every field must have a **verbatim evidence span** (char offsets). If no span, set `value=null` and `status="no_evidence"`.
   * **Link:** canonicalize names → co-reference within doc → entity resolution across corpus → produce a candidate **Person** and related nodes.
   * **Upsert:** write only fields with `confidence >= threshold` and keep **provenance** (doc_id, sentence_id, span).

2. **LLM as a *validator/extractor***, not an author.

   * Temperature = 0.
   * Force **JSON schema** with tools/function-calling or constrained decoding.
   * Require the model to **quote exact substrings** for each field and to return offsets.
   * If a value isn’t present verbatim, it must be null (or “unknown”). No guessing.

3. **Small specialist models first, LLM second**

   * Use a lightweight Vietnamese NER (PhoBERT-based, VnCoreNLP, spaCy custom) for **PERSON, ORG, LOC** and your regex module for **PHONE/ID/DOB**.
   * Let the LLM do **role labeling, event frames** (“who did what to whom, when, where, why”) and relation typing—things that are harder with rules.

### Workflow comparison: cascaded NER+LLM vs dual-LLM

**Workflow A — NER → LLM → Assembly → Validation**

1. **NER pre-processing:** fast specialist model marks every person span and delivers offsets + snippets.
2. **LLM enrichment:** pass the mention list plus local context (1–2 sentences) into the LLM so it can disambiguate, add roles/actions, and map attributes back to the canonical span IDs.
3. **Structured assembly:** deterministic code (or the same LLM with a strict schema tool) consolidates mentions into person profiles, applies linking, and enforces evidence tracking.
4. **Validation:** a light verifier (classifier or prompt-based self-check) replays each field against the text to flag hallucinations or low evidence.

*Pros:* excellent recall/precision trade-off, cheaper tokens (LLM reads only relevant windows), easy to parallelize per mention, and the assembly layer keeps deterministic control.  
*Cons:* requires maintaining the NER/rule stack and a mention-to-context join step; upstream NER recall caps what the LLM can ever see.

**Workflow B — LLM extraction → LLM validation**

1. **LLM extraction:** send the raw passage to an instruction-tuned model that outputs the full schema directly.
2. **LLM validation:** a second pass (same or smaller model) re-reads the passage + extracted JSON to verify each field, drop hallucinations, or request `no_answer`.

*Pros:* simpler pipeline (no classic NER), easier to iterate on prompts, and validation can focus on logical consistency or cross-field rules.  
*Cons:* higher token cost because the first pass must read entire passages, validation doubles LLM latency, and you rely entirely on prompt discipline for recall/coverage.

**Guidance**

* Pick Workflow A when latency, controllability, and evidence tracing matter most (e.g., compliance dashboards) or when you already own strong NER/rule components.
* Pick Workflow B for rapid prototyping or domains where bespoke NER models do not exist yet, but budget for heavier prompt engineering and more aggressive verifier rules.
* Hybridize when possible: even inside Workflow B you can inject regex/NER hints as “candidate mentions” to improve recall without forcing a full assembly tier.

# 2) Extraction design (schema + prompt guardrails)

### JSON schema (Pydantic-ish)

```json
{
  "people": [
    {
      "name": {"text": "", "start": -1, "end": -1},
      "aliases": [{"text":"", "start":-1, "end":-1}],
      "age": {"value": null, "start": -1, "end": -1},
      "birth_date": {"value": null, "start": -1, "end": -1},
      "birth_place": {"text": "", "start": -1, "end": -1},
      "phones": [{"text":"", "start":-1, "end":-1}],
      "national_id": {"text":"", "start":-1, "end":-1},
      "roles": [{"label":"suspect|victim|official|witness|lawyer|judge|other","sentence_id":0}],
      "actions": [
        {
          "predicate":"arrested|charged|sentenced|confessed|searched|seized|…",
          "object_person_name":"", "amount_vnd": null, "law_article":"", "sentence_years": null,
          "sentence_id": 0
        }
      ],
      "provenance": {"doc_id":"", "passage_id":0},
      "confidence": 0.0
    }
  ]
}
```

### LLM prompt (pattern)

* System:

  * “Extract only what is explicitly present in the text. **Never infer** ages, phone numbers, or places that aren’t verbatim.
  * Return **valid JSON** that matches the schema.
  * For each value, include exact substring and character offsets relative to the provided passage.
  * If absent, output `null` or empty array.
  * If you can’t conform, return: `{"error":"no_answer"}`.”
* User:

  * Provide **one passage** + its `doc_id`, `passage_id`, and the **article publish date** (so the model can convert “35 tuổi” → birth year estimate if you allow it).

### Verbatim check (post-LLM)

* For every non-null field with a `text` value, verify `passage[text.start:text.end] == text`.
* For derived values (e.g., `birth_year = article_date.year - age`), set `derivation="age_to_year"` and keep original span for “35 tuổi”.

### Deterministic modules (before/after LLM)

* **Phones:** `(?:\\+84|0)(?:\\d[ -]?){8,10}` → normalize to E.164 (+84...).
* **Vietnamese IDs:** CCCD/CMND patterns (12 digits; legacy 9 digits).
* **Money:** VNĐ patterns (triệu/tỷ), normalize to integer VND.
* **Dates:** dd/mm/yyyy, “ngày 16/10”, “tháng 10/2025”.
* **Law articles:** “Điều 174 BLHS”, etc.

# 3) Intra-doc & cross-doc entity linking

### Intra-doc coref

* Cluster name mentions + pronouns + appositives:
  “Nguyễn Văn A (35 tuổi, quê Quảng Nam)” → coref cluster for A contains aliases (“A”, “ông A”, “bị cáo A”).
* Use rules + embeddings + small coref model; keep sentence IDs.

### Cross-doc resolution

* Build a **blocking key** to limit comparisons:

  * `name_key = normalize(name) → lowercase, strip diacritics, remove middle-name stopwords`
  * Optional blocks: hometown/province, birth_year±1, organization, phone hash.
* **Scoring model** combines:

  * Name similarity (accent-insensitive and accent-sensitive), token Jaro-Winkler, initials match.
  * Attribute overlap: phone exact match (strong), national_id (very strong), birthdate (strong), hometown (medium), co-mentioned orgs (weak).
  * Textual context embeddings (mean of surrounding sentences).
* Thresholds:

  * `p(match) >= 0.9` → auto-merge
  * `0.6–0.9` → queue for human review
  * `<0.6` → new node

### Canonicalization

* For each **Person**, choose canonical `display_name` by majority and source authority weighting; store all alias spellings (with/without diacritics).

# 4) Event & relation extraction (the “what they did”)

Represent articles as **events** with roles (who/what/where/when/how much). This makes your graph much more expressive and auditable.

**Event node:** `CaseEvent {type, time, location, amount_vnd, law_article, doc_id, sentence_id}`
**Edges:**

* `(Person)-[:ROLE {label:'suspect'|'victim'|'official'}]->(CaseEvent)`
* `(CaseEvent)-[:MENTIONED_IN]->(Article)`
* `(:CaseEvent)-[:INVOLVES_ORG]->(:Organization)`
* Optional: `(:Person)-[:ASSOC_WITH {evidence}]->(:Person)` for co-defendant/associate.

Have the LLM output **event frames** per sentence (or per paragraph) with pointers to spans and sentence IDs; your verifier confirms spans.

# 5) Graph DB schema (Neo4j/Cypher example)

### Constraints & indexes

```cypher
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.uid IS UNIQUE;
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name_lc);
CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.doc_id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:CaseEvent) REQUIRE e.eid IS UNIQUE;
```

### Upsert helpers

```cypher
// Deterministic UID for Person: hash(name_lc_no_diacritics, birth_year?, hometown?)
WITH $person AS p
MERGE (per:Person {uid: p.uid})
ON CREATE SET per.name = p.display_name, per.name_lc = p.name_lc, per.created_at = timestamp()
SET per.last_seen = timestamp(),
    per.aliases = apoc.coll.toSet(coalesce(per.aliases,[])+p.aliases),
    per.hometown = coalesce(per.hometown, p.hometown),
    per.birth_year = coalesce(per.birth_year, p.birth_year),
    per.confidence = coalesce(per.confidence, 0.0) * 0.8 + p.confidence * 0.2;

MERGE (a:Article {doc_id: $doc.doc_id})
ON CREATE SET a.url = $doc.url, a.published_at = $doc.published_at;

MERGE (e:CaseEvent {eid: $event.eid})
ON CREATE SET e.type = $event.type, e.time = $event.time, e.location = $event.location,
              e.amount_vnd = $event.amount_vnd, e.law_article = $event.law_article;

MERGE (per)-[:ROLE {label:$role.label, sentence_id:$role.sentence_id, evidence:$role.evidence}]->(e)
MERGE (e)-[:MENTIONED_IN]->(a);
```

**Provenance pattern:** store `{doc_id, sentence_id, start, end}` on edges or an `Evidence` node; keep raw quote to audit.

# 6) Confidence, dedup, and governance

* **Field-level confidence:** start from detector confidence; discount if span is short/ambiguous; boost if repeated across independent sources.
* **Entity-level confidence:** aggregate fields; require at least one strong identifier (phone, national_id, DOB) or ≥2 medium signals.
* **Fact aging:** if old facts conflict with newer ones (e.g., age), prefer newer publish date; keep history with `valid_from/valid_to`.
* **Contentious categories (law/crime):**

  * Separate **allegations** vs **convictions**; use `status: alleged|charged|convicted|acquitted`.
  * Keep **source type** and **publisher reputation score**; never present unverified allegations as facts.
  * Consider local legal and privacy compliance before making profiles discoverable.

# 7) Practical tricks to kill hallucinations

* **Span-only policy:** any field without a substring + offsets = dropped.
* **Passage gating:** feed the model only the relevant paragraph(s) found by BM25/embedding retrieval; smaller context reduces “creative” output.
* **Negative instruction:** “If unsure, return null.” + evaluate that you actually get nulls.
* **Two-phase check:** Extract → Verify (string match) → If mismatch, auto-set null.
* **Self-consistency check:** run the LLM twice; only accept fields that agree **and** share the same evidence span.
* **Post-hoc rule filters:** e.g., phone must start with `+84` or `0` and be 10–11 digits after normalization.
* **Red teaming:** add test cases where the passage lacks a phone/age to ensure the model outputs null.

# 8) Vietnamese-specific notes

* Normalize diacritics both ways: store `name_lc` and `name_ascii` (folded).
* Age expressions: “35 tuổi”, “SN 1989”, “sinh năm 1989”. Compute `birth_year = published_year - age` and mark as `derived=true`.
* Places: province/city dictionaries (Quảng Nam, Thừa Thiên Huế, TP.HCM vs “Thành phố Hồ Chí Minh”). Keep `place_id` from a gazetteer to stabilize edges.
* Law refs: map “Điều 174 BLHS 2015” to a canonical `LawArticle` node.

# 9) Minimal PoC you can build this week

1. **Corpus loader** → store raw HTML + cleaned text + sentence offsets.
2. **Regex + NER** pass → emit spans for phones, DOB, money, law articles, PERSON names.
3. **LLM extractor** (temperature 0, constrained JSON) per paragraph with the schema above.
4. **Verifier** → drop anything without exact span match; compute confidence.
5. **Intra-doc coref** → cluster mentions.
6. **Cross-doc linker** with a simple blocking key + similarity scoring.
7. **Neo4j upsert** with provenance on edges.
8. **Dashboard**: show Person → list of evidence quotes with links back to article + sentence.

# 10) Example: super-tight extraction prompt (drop-in)

Use this when calling your 8B model (or larger) with function-calling:

> You are an information extractor. **Do not guess.** Work only from the passage below.
> Return **valid JSON** exactly matching this schema: … (paste schema)
> For every non-null field, also return `"start"` and `"end"` offsets.
> If a field isn’t explicitly present in the text, set it to `null` (or empty array).
> Passage metadata: `doc_id=…`, `passage_id=…`, `published_at=YYYY-MM-DD`.
> **Output only JSON.**

And set:

* temperature=0, top_p=1.0
* max_tokens sized to schema
* enable JSON schema/tool mode if your server supports it

---

If you want, I can:

* give you a ready-to-use **Pydantic models + verifier** function,
* sketch the **Cypher upsert procedures**,
* or adapt this to your existing Milvus/Postgres stack (you can store spans + embeddings for fast evidence retrieval).

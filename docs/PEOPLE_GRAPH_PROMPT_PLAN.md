# People Graph Prompt Plan

## Goal
- Extract graph-ready people profiles from Vietnamese news passages for insertion into a Neo4j-style graph while preserving verbatim spans and provenance.

## Prompt Guidance (to replace the user guidance block)
- Only list individuals mentioned with proper names or explicit nicknames (capitalized); one entry per person. Merge nicknames into `aliases`.
- Skip mentions that are only common nouns or pronouns without a proper name (e.g., "người đàn ông", "tài xế", "anh ta", "họ", "nạn nhân"). Do not infer names from titles/relations.
- All `text/start/end` values must come verbatim from the passage; if absent, use null or empty list and set start/end = -1.
- Graph relations from passage:
  - Organizations that co-occur with the named person go into `organizations` with exact spans.
  - Actions are only created when the sentence contains the person’s name; fill predicate and spans for object/location/time/items/law_article; use the sentence_id of that sentence.
- Do not create any value or relationship without an explicit phrase in the passage; no guessing.

## Graph Mapping (ETL from existing schema)
- Nodes: Person {doc_id, name, aliases, national_id, birth_date, age, address, occupation, confidence}; Organization {name, doc_id}; Action {predicate, object_description, time, location, amount, law_article, sentence_years, doc_id, sentence_id}; optional Phone/Alias nodes if normalizing.
- Edges: Person-[:MEMBER_OF|:RELATED_TO]->Organization (from organizations spans); Person-[:HAS_PHONE]->Phone; Person-[:PLAYED_ROLE {label, sentence_id}]->Action (from roles); Person-[:PARTICIPATED_IN {sentence_id}]->Action (from actions); optionally Action-[:AT_LOCATION]->Location and Action-[:AT_TIME]->Time if normalized as nodes.
- Provenance: attach start/end, passage_id, source_text, doc_id to nodes/edges; skip creating a value/edge when start/end = -1.
- Upsert keys: Person keyed by (doc_id, name.text) plus national_id when present; Action keyed by (doc_id, passage_id, sentence_id, predicate, source_text) to avoid merging unrelated events.

## Schema Options
- Recommended: keep the current extraction schema and build graph edges in ETL from existing fields (organizations, roles, actions, phones, aliases, provenance).
- Optional extension (only if recall demands explicit edges): add `relationships` under each person:
  - relationships: [{"target_text": "", "target_type": "person|organization", "relation": "ally|rival|works_for|family|other", "start": -1, "end": -1, "sentence_id": 0}]
- Optional: add `action_id` to actions to give a stable key for downstream upserts; otherwise derive from doc_id + passage_id + sentence_id + predicate + source_text.

## ETL Plan to Graph
- Inputs: `data/cache/workflow_b_fast/pass1/*.json` (or pass2 if validated), containing `people`, `passage.text`, `doc_id`, `passage_id`, `sentence_id` hints.
- Keying/normalization:
  - person_key = `(doc_id, name.text)`; if `national_id` exists, use `(doc_id, national_id)` as stronger key.
  - action_key = `(doc_id, passage_id, sentence_id, predicate, response.raw substring/object_description.text)`.
  - org_key = `(doc_id, org.text)`.
- Node upserts (Cypher-style pseudo):
  - `MERGE (p:Person {doc_id: $doc_id, name: $name}) ON CREATE SET p.national_id=..., p.birth_date=..., p.age=..., p.address=..., p.occupation=..., p.confidence=... , p.source_text=...`
  - `MERGE (o:Organization {doc_id: $doc_id, name: $org_text})`
  - `MERGE (a:Action {doc_id: $doc_id, passage_id: $p_id, sentence_id: $s_id, predicate: $pred, source_text: $src}) ON CREATE SET a.object_description=..., a.location=..., a.time=..., a.amount=..., a.law_article=..., a.sentence_years=...`
  - Optional: `MERGE (ph:Phone {number: $phone})`
- Relationships:
  - `p-[:MEMBER_OF {start, end, passage_id, sentence_id}]->o` for each `organizations[]` with valid spans.
  - `p-[:HAS_PHONE {start, end, passage_id}]->ph` for each phone span.
  - `p-[:PLAYED_ROLE {label, sentence_id}]->a` using roles (only if matching action sentence_id; else attach directly to person as property).
  - `p-[:PARTICIPATED_IN {sentence_id}]->a` for each action; attach items_seized as `[:TARGET_ITEM]->(:Item {name})` if you normalize, else store as list on Action.
  - Optional: split `location`/`time` into nodes and link via `[:AT_LOCATION]` / `[:AT_TIME]`.
- Provenance:
  - Store `start/end/passage_id` on every relationship created from a span.
  - Keep `raw_passage` or `source_text` on Action/Person for auditing; include `model` and `prompt_version` as properties on nodes or via metadata nodes if needed.
- Validation/filters:
  - If `start/end == -1` → skip creating that node/edge or value.
  - Drop/flag any record with `status != "ok"` from pass1; prefer pass2 validated outputs when available.
- Batch/load:
  - Export per-doc NDJSON with resolved keys; use `neo4j-admin import` or APOC `apoc.periodic.iterate` to stream merges in batches (e.g., 1k rows).
  - Maintain idempotency via MERGE keys above; set `updated_at` on each load for change tracking.

## Intra-Article Person-Person Relations (CO_MENTION only)
- Edge type: `CO_MENTION` when two people appear in the same article; accumulates counts across passages/sentences.
- Construction:
  - For each article, collect persons keyed by `(doc_id, person_key)` with spans per passage/sentence.
  - For each passage/sentence, form all unordered person pairs with valid spans (`start/end != -1`) and add evidence.
- Edge properties:
  - `doc_id`, `passage_id`, `sentence_id` when available, `count`, `evidence` (array of span pairs per co-occurrence), optional `source_hash`.
  - MERGE key: `(doc_id, personA, personB, 'CO_MENTION')` with names sorted to stay idempotent; increment `count`, append evidence up to a cap.
- Filters:
  - Skip pairs if either person has invalid spans or placeholder names.
  - Cap evidence array length to control payload size; keep counts for frequency.
- Load:
  - Emit NDJSON for co-mention edges per article, then `apoc.periodic.iterate` to MERGE `CO_MENTION` edges, incrementing counts and evidence.

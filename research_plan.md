# Document-Native Multimodal Evidence-Graph RAG for Long-Document QA

## Project brief

This project tests whether document-native relationships between text, figures,
tables, captions, and pages improve multimodal evidence retrieval for long-document
question answering. The target benchmark is MMDocRAG (Dong et al., NeurIPS 2025;
verify the released dataset version, split, and license before implementation).

Rather than treating a document as a flat set of embedded chunks, the system models
its evidence units and their structural, referential, and semantic links as a
heterogeneous graph. The emphasis is on whether those native links recover
complementary evidence that dense retrieval and generic entity graphs miss.

## Research Question

Can document-native relationships between text, figures, tables, captions, and pages
improve completeness of multimodal evidence retrieval beyond flat dense retrieval?

**Hypotheses**

- **H1 — retrieval:** graph expansion improves gold-evidence recall at a fixed
  candidate budget, especially on cross-page and cross-modal questions.
- **H2 — relationship type:** structural and referential edges add value beyond a
  generic entity co-occurrence graph; semantic edges may add recall but also noise.
- **H3 — stretch:** coverage-guided traversal reduces unnecessary re-retrieval
  relative to adaptive dense retrieval, without lowering answer quality.

**Not claiming**: "we invented GraphRAG" or "we invented adaptive RAG" (both exist — Self-RAG,
Adaptive-RAG, PolyG, 2025 ACL query-driven multimodal GraphRAG, AAAI'26 Relink, SAM-RAG).

**Claiming**: document-native structural relationships (page/caption/cross-reference), not
generic entity-KG relationships, can recover complementary evidence that flat/generic-graph
retrieval misses. The study identifies *which* relationship types matter on a controlled
MMDocRAG evaluation setup.

## Experimental contract

Lock the following before inspecting results. This makes comparisons interpretable
and prevents gains caused by a larger context or a different generation setup.

| Item | Rule |
|---|---|
| Dataset | Record release/version, split, access date, and any exclusions. |
| Prototype subset | Stratify the 20–30 document subset; use it only for development, never final claims. |
| Candidate budget | Use identical seed count and final candidate cap for every system. |
| Models | Keep embedder, reranker, generator/judge, prompts, temperature, and context limit fixed. |
| Randomness | Fix all seeds and report variation when generation is sampled. |
| Primary metric | Gold-evidence Recall@k at the fixed final candidate budget. |
| Secondary metrics | Quote-selection P/R/F1, answer quality, latency, and cost. |

Use official benchmark metrics and question labels when released. If a substitute is
needed, document it explicitly rather than describing it as an official metric.

## Contributions (paper framing)

- **C1** — Document-native heterogeneous evidence graph (node types: PAGE, TEXT, IMAGE,
  TABLE, FIGURE, CAPTION; edge types: structural, referential, semantic).
- **C2** — Systematic ablation of which relationship types improve retrieval/selection,
  broken down by MMDocRAG's own question-type labels (cross-page, cross-modal, both).
- **C3** (stretch) — Query-Element Coverage as a deterministic, principled stopping
  criterion for adaptive graph traversal (not entity/number matching alone — validated
  against what the question actually requires, including non-textual elements like "Figure 4").

---

## Phase 0 

- [ ] Get MMDocRAG data access: 313 docs / 4,055 QA pairs, gold + noisy quotes, question-type
      labels. Verify these counts and fields against the released benchmark rather than
      relying on the paper alone.
- [ ] Subsample 20–30 documents for prototyping with a recorded sampling seed and
      question-type distribution. **Do not** treat this as the final evaluation set.
- [ ] Create a run manifest: dataset version, document/QA IDs, preprocessing version,
      model versions, prompts, seeds, and retrieval/context budgets.
- [ ] Inspect 10 representative documents to verify page geometry, caption links,
      identifiers, OCR, and source offsets support the planned graph edges.
- [ ] Environment: Python, a vector DB (Chroma/FAISS — whatever's already installed, no new
      dep), a graph lib (networkx is enough at this scale, no Neo4j needed).
- [ ] Pull an embedding model + one LLM for generation/judging (whatever you already have
      API access to — GPT-4.1/Claude/local — doesn't need to match their exact model list).



**Exit criterion:** a reproducible dense-only baseline emits per-question candidate
lists and retrieval metrics on the chosen prototype subset.

---

## Phase 1 — Evidence Graph + Static Expansion (MUST SHIP)

Goal: working pipeline, first Recall@k / F1 table vs their flat retrieval baseline.

### 1.1 Graph construction (per document)
- [ ] Nodes: every quote becomes a typed node — PAGE, TEXT, IMAGE, TABLE, FIGURE, CAPTION.
- [ ] Retain source provenance on every node: document ID, page ID, source span or
      bounding box, extraction method, and raw text/OCR where available.
- [ ] Structural edges: same-page, caption-proximity (layout-adjacent), figure↔caption,
      table↔caption. Deterministic, cheap — parse page/position metadata MMDocRAG already has.
- [ ] Referential edges: regex/pattern match "as shown in Figure N" / "Table N" in text
      quotes → edge to that node. Also explicit cross-page references if phrased that way.
- [ ] Semantic edges: entity/number overlap (extract via regex + NER — spaCy is enough) between
      text and table/figure OCR'd content; embedding similarity as a softer semantic edge.
- [ ] Store edge type + a raw confidence score per edge (don't combine yet — Phase 2 does that).
- [ ] Unit-test reference parsing, including unresolved and duplicated identifiers;
      unresolved mentions must not silently become edges.

 (bulk of Phase 1 — reference-parsing and entity-overlap logic take the
most iteration; structural edges are near-trivial once page metadata is parsed).

### 1.2 Retrieval pipeline
- [ ] Vector search → top-5 seed nodes.
- [ ] 1–2 hop graph expansion from seeds (no learned weights yet, just edge presence).
- [ ] Rerank combined candidate set, cap at C15/C20 (their exact candidate budget — keeps
      comparison apples-to-apples).
- [ ] Feed to LLM for quote selection + answer generation (their exact task setup).
- [ ] Log seeds, traversed edges, pre-rerank candidates, final candidates, and each
      expanded candidate's edge provenance for every question.



### 1.3 Evaluation
- [ ] Run both: (a) flat dense retrieval only [baseline], (b) your graph-expanded retrieval.
- [ ] Use MMDocRAG's released metrics where available: Recall@k, quote-selection
      Precision/Recall/F1, and answer quality. Document any unavailable metric or
      substitute rather than calling it an exact reproduction.
- [ ] **Split every metric by question type**: normal / cross-page / cross-modal /
      cross-page+cross-modal. This breakdown *is* the headline table.
- [ ] Use paired confidence intervals (for example, bootstrap intervals) for the
      dense-only versus graph-expanded primary comparison.
- [ ] Audit at least 25 changed examples across gains, regressions, and no-change cases.

 (mostly compute/run time + result aggregation).


**Decision gate:** if graph retrieval shows no credible recall movement on the
cross-page/cross-modal subset, diagnose edge quality, parsing, and candidate budget
before investing in further features. Treat that outcome as a useful negative result,
not a reason to tune selectively until it disappears.

---

## Phase 2 — Ablation + Edge Validation + Failure Analysis (MUST SHIP)

This phase *is* the actual research contribution — not the graph itself.

### 2.1 Edge weighting (not raw LLM filtering)
- [ ] First complete the separate edge-family ablation below; only then combine
      edge types via an interpretable weighted score:
      `w(e) = α·S_structural + β·S_referential + γ·S_semantic + δ·S_entailment`
- [ ] Entailment score: cheap NLI/entailment check — does linked text actually
      support/explain the linked figure/table, not just co-occur. Use a lightweight
      classifier or single LLM call per edge (not per traversal step).
- [ ] Validate entailment on a small manually labelled edge sample before relying on
      it in traversal; report that validation separately from end-task results.



### 2.2 Ablation matrix (six systems — this answers "which relationship matters")

| Model | Structural | Referential | Semantic | Entailment | Adaptive |
|-------|:---:|:---:|:---:|:---:|:---:|
| A (dense only) | ✗ | ✗ | ✗ | ✗ | ✗ |
| B | ✓ | ✗ | ✗ | ✗ | ✗ |
| C | ✓ | ✓ | ✗ | ✗ | ✗ |
| D | ✓ | ✓ | ✓ | ✗ | ✗ |
| E | ✓ | ✓ | ✓ | ✓ | ✗ |
| F (full, stretch) | ✓ | ✓ | ✓ | ✓ | ✓ |

- [ ] Run A–E (F needs Phase 3) on the same data, with the same seed count, candidate
      cap, reranker, generator, and question-type split.
- [ ] Report absolute scores and paired differences from A. Predefine C versus the
      generic graph baseline, and D/E versus C, as the key edge-type comparisons.

(mostly re-running pipeline with edge subsets toggled — infra from Phase 1 reused).

### 2.3 Generic GraphRAG baseline (critical — proves document-native > generic KG)
- [ ] Build one generic entity-graph baseline (e.g. simple entity co-occurrence graph,
      using the same nodes and traversal budget but no page/caption/adjacency/reference
      edges) for comparison. It need not be a state-of-the-art GraphRAG implementation;
      it is the controlled "graph without document-native edge types" baseline.



### 2.4 Failure taxonomy
- [ ] Categorize errors into: missing evidence / incomplete evidence / wrong modality /
      distractor evidence / citation error (their paper's own stated failure clusters).
- [ ] Report counts per category, before vs after each edge type added.
- [ ] Use a fixed, documented error sample. If feasible, have a second annotator
      review a subset; otherwise label the analysis as single-annotator.

 (manual/semi-automated error categorization on a sample of failures).



**Exit criterion:** the ablation, generic-graph control, and failure analysis can
support or reject the document-native-structure hypothesis.

---

## Phase 3 — Coverage-Guided Adaptive Retrieval (STRETCH, optional)

Only start after Phase 1+2 numbers exist and show signal.

### 3.1 Query-Element Coverage (deterministic, not "evidence sufficiency")
```
def query_element_coverage(question, retrieved_nodes):
    required = extract_required_elements(question)
    # entities, numbers (regex+NER) AND explicit references
    # e.g. "Figure 4" as its own required element, not just text tokens
    matched = 0
    for elem in required:
        if elem.is_reference:
            matched += 1 if elem.target_node in retrieved_nodes else 0
        else:
            matched += 1 if elem.is_covered_by(retrieved_nodes) else 0
    return matched / len(required) if required else 1.0
```
- [ ] Threshold τ (e.g. 0.9) → sufficient vs insufficient.
- [ ] Specify `extract_required_elements` and `is_covered_by` before tuning τ. A
      lexical match alone must not cover an explicit figure/table reference.
- [ ] Validate the coverage/sufficiency signal separately: sample at least 30 cases,
      hand-check against gold evidence, and report precision/recall or agreement as
      its own number. Do not treat an LLM judge as ground truth.

(extraction logic + validation sampling).

### 3.2 Graph-guided re-expansion loop
- [ ] Insufficient coverage → traverse unexplored neighbors of already-retrieved nodes
      (not blind query rewrite) → recheck coverage → repeat (bounded, e.g. max 3 rounds).
- [ ] Compare against generic adaptive-RAG baseline: retrieve → insufficient → rewrite
      query → retrieve again (no graph guidance).
- [ ] Match the maximum rounds, context budget, and model-call budget across both loops.



### 3.3 Efficiency/cost comparison
- [ ] Track LLM calls, tokens, latency per question for both loops.
- [ ] Report medians and tail latency as well as averages.
- [ ] Make a "better and cheaper" claim only when both answer quality and measured
      cost remain favorable under the same budget.



---

## Full Timeline Summary

| Phase | Content | Time | Required? |
|---|---|---|---|
| 0 | Setup, data access, subsample | 1–2 days | Yes |
| 1 | Evidence graph + static expansion + first table | 10–13 days | **Yes — must ship** |
| 2 | Edge ablation + generic-GraphRAG baseline + failure analysis | 7–10 days | **Yes — must ship** |
| 3 | Coverage-guided adaptive retrieval + cost comparison | 6–9 days | Stretch only |



The Phase 1 decision gate applies: do not proceed to adaptive retrieval without a
credible, audited static-retrieval signal.

## What NOT to do
- Don't build Phase 3 before Phase 1+2 numbers exist.
- Don't skip the generic-GraphRAG baseline — "graph beats flat vector search" is not
  the claim; "document-native structure beats generic graph structure" is.
- Don't claim novelty on "query-driven evidence graph" alone (AAAI'26 Relink already
  does this) or "adaptive RAG" alone (Self-RAG/Adaptive-RAG/PolyG already exist).
  Novelty = the combination + MMDocRAG's controlled multimodal-evidence-selection setting
  + the relationship-type ablation finding.

## Draft novelty statement
> We study document-native multimodal evidence structure as a mechanism for
> improving evidence completeness in long-document QA. On a controlled multimodal
> evidence-selection benchmark, we isolate the contribution of structural,
> referential, and semantic relationships to retrieval performance on cross-page
> and cross-modal questions.

# Architectural patterns for expanding a multi-agent legal RAG system

**The strongest architecture for your system is a three-layer hybrid — not the monolithic TaskSpec approach Perplexity proposed.** Use Neo4j for what it excels at (argument graphs, ontology, constraints, cross-document alignment), lightweight YAML/Python configs for pipeline sequencing and parameters, and document-type-specific prompt templates for agent behavior modification. This hybrid reuses your existing infrastructure, avoids framework lock-in, and provides a clear path from reply briefs to general family law documents. The reply brief problem specifically is best solved by a purpose-built "diff-and-respond" pipeline that slots into this hybrid as one pipeline configuration, treating the reply as a constrained transformation of the opening brief conditioned on the opposing brief.

## Why Perplexity's config-file-centric approach falls short

The proposed TaskSpec + DocumentTypeSpec + domain ontology in YAML/JSON has real appeal — declarative configs are easy to version-control, validate with Pydantic, and diff in code review. Industry practice confirms this: CrewAI uses YAML for agent definitions, Google ADK supports single-YAML multi-agent systems, and Microsoft's Agent Framework enables YAML/JSON agent definitions deployable via standard CI/CD. For purely parametric variation (different context budgets, different refinement pass counts, different model selections per document type), **YAML configs are the right tool**.

But reply briefs aren't a parametric variation of opening briefs — they're a structurally different task requiring argument extraction from two adversarial documents, cross-document alignment, mischaracterization detection, and scope constraints. No amount of YAML configuration captures "align each opposing argument to its opening brief counterpart, then generate a targeted response while ensuring no new issues are raised." That logic lives in code. Meanwhile, the relationships between document types (an opening brief *generates* a reply brief, which *must not exceed the scope of* its parent) and the legal constraints governing each type are fundamentally graph problems — exactly what your **4 Neo4j databases** already handle.

The config-only approach creates a parallel configuration universe disconnected from your data layer. Every legal constraint you encode in YAML is a relationship that should be queryable alongside your case law, argument structures, and citation graphs. A hybrid that puts relationships in the graph, parameters in configs, and behavior in prompts avoids this duplication while keeping each layer simple.

## The three-layer hybrid architecture in detail

**Layer 1: Neo4j-native ontology and constraints.** Document types, their input requirements, legal constraints, argument structures, and domain knowledge live as labeled property graph nodes and edges in your existing Neo4j databases. A `(:DocumentType {name: "ReplyBrief"})` node connects via `[:REQUIRES_INPUT]` to both `(:InputSpec {name: "OpeningBrief"})` and `(:InputSpec {name: "OpposingBrief"})`, and via `[:HAS_CONSTRAINT]` to `(:LegalConstraint {name: "NoNewIssues"})`. This is not speculative — Squirro embeds AI prompt templates directly inside graph nodes using BPMN workflow graphs as state machines, with "the graph controlling the journey while the large language model executes micro-tasks at each stop." Outbrain embedded Neo4j into their workflow engine to manage **1,000+ data flows** with "relatively small development investment." Neo4j's own documentation demonstrates request-state modeling where adding a workflow step means adding nodes and edges rather than rewriting configuration files.

**Layer 2: Lightweight pipeline configs.** Simple Python dictionaries or YAML files define step sequences, context budgets, model assignments, and refinement pass counts per document type. These handle the 80% of variation that is genuinely parametric. Keep these minimal — a pipeline config for a reply brief is just `steps: [extract_arguments, align_arguments, generate_responses, assemble, refine, verify]` plus numerical parameters. Validate with Pydantic models. This is where Perplexity's insight is correct: declarative configuration for declarative concerns.

**Layer 3: Document-type-specific prompt templates.** Each agent in your 7-agent council gets parameterized prompt templates that adapt behavior per document type. Research from the MASS framework (Multi-Agent System Search, arXiv 2502.02533, February 2025) demonstrates that **prompt optimization alone shows "significant advantages in token-effectiveness over other building blocks"** including scaling agents and adding self-refinement loops. For many document types, prompt modification is sufficient — a motion-drafting prompt versus a declaration-drafting prompt versus a trial-brief prompt may be all that distinguishes those pipelines, since the underlying retrieval-draft-refine-verify cycle remains the same.

The design principle is simple: **use each tool for what it's best at.** Neo4j handles relationships, constraints, and ontology. YAML handles parameter lists and step sequences. Prompt templates handle agent behavior. Custom Python code handles specialized logic (argument extraction, alignment) that doesn't fit any configuration format.

## Reply briefs demand a diff-and-respond pipeline

No existing production system generates complete reply briefs autonomously end-to-end, but several commercial systems implement key pieces. **CaseMark** offers a dedicated "Reply Brief for Appellant" workflow that takes two input documents, extracts arguments and legal theories from the opening brief, maps each opposing argument to a rebuttal section, and cross-references the opening brief for consistency. **Bloomberg Law's Brief Analyzer** identifies arguments by section in an uploaded opposing brief and flags "points potentially lacking in the opposing counsel's argument" — **94%** of users reported it reduced response preparation time. Harvey AI's architecture decomposes generated responses into individual factual claims and cross-references each against authoritative sources, reducing hallucination to approximately **0.2%**.

The academic blueprint comes from IBM's Project Debater, published in *Nature* (2021). Its rebuttal module pre-mines claims relevant to the opponent's likely positions, then at runtime matches these against the opponent's actual arguments and generates targeted responses. The critical insight for your system: **offline preparation of counter-arguments paired with real-time alignment** is the pattern that works.

Your reply brief pipeline should execute in five phases:

**Phase 1 — Argument extraction.** Parse both the opening brief and opposing brief into structured argument nodes. Each argument gets a claim, supporting grounds, legal authorities, and page/paragraph provenance. Store as nodes in Neo4j.

**Phase 2 — Argument alignment.** Match opposing arguments to opening brief arguments using sentence-BERT embeddings (your existing 768-dimensional embeddings work here) plus Neo4j graph traversal through shared legal issues. The CAIL 2020–2023 Argument Mining benchmarks demonstrate adversarial argument pair extraction achieving **0.56–0.90 accuracy** depending on difficulty, sufficient for assisted drafting with human review. Store alignments as `[:RESPONDS_TO]`, `[:ATTACKS]`, and `[:CONCEDES]` edges.

**Phase 3 — Mischaracterization detection.** For each claim the opposing brief makes about your arguments, retrieve the actual text from the opening brief and run NLI classification (e.g., `roberta-large-mnli`). Entailment means accurate characterization; contradiction means mischaracterization; neutral means incomplete characterization. Flag contradictions with probability above **0.7** as definite mischaracterizations and **0.4–0.7** as possible distortions for attorney review.

**Phase 4 — Constrained response generation.** For each aligned argument pair, retrieve relevant case law via your existing reciprocal rank fusion pipeline, generate a targeted rebuttal, then validate scope: compute cosine similarity between the generated argument and the nearest issue in the opening brief's "issue registry." If below threshold, flag as a potential new issue. This enforces the no-new-issues constraint computationally rather than relying solely on prompt instructions.

**Phase 5 — Citation and provenance assembly.** Every sentence carries a provenance record mapping to Opening Brief page/paragraph, Opposing Brief page/paragraph, or legal authority citation. Use citation anchors like `[OB:p12¶3]` and `[RB:p7¶2]` embedded in chunk text during ingestion, stored as metadata on Neo4j nodes. Your existing citation verification pipeline then validates all legal authorities.

This pipeline slots into the hybrid architecture as one pipeline configuration that the system routes to for reply briefs, while opening briefs, motions, and declarations use simpler pipeline configs with prompt-modified agents.

## IBIS plus Toulmin creates the optimal argument graph for Neo4j

For representing legal arguments as graphs that enable cross-document alignment, a **hybrid IBIS + Toulmin** schema in Neo4j provides the best tradeoff between formal rigor, implementation simplicity, and adversarial alignment capability.

IBIS (Issue-Based Information System) maps naturally to adversarial briefs: **Issues** are legal questions the court must decide, **Positions** are each party's proposed answers, and **Arguments** are pros and cons supporting each position. Both parties' Positions respond to the same Issues, making alignment explicit and queryable. Toulmin provides internal structure within each argument — Claim, Grounds, Warrant, Backing, Qualifier, Rebuttal — originally modeled on legal courtroom arguments. The most significant Toulmin implementation is **Split-Up** (Stranieri & Zeleznikow, 1992–2001), a decision support system for Australian **family law** property division using a hierarchy of 94 factors represented as Toulmin argument trees.

The AIF (Argument Interchange Format) has been **directly implemented in Neo4j** by Hautli-Janisz et al., who constructed a knowledge base from 30 BBC Question Time debates annotated in AIF, converted to ASPIC+ argumentation theory, with arguments, defeat relations, and dialogue structures instantiated as a Neo4j graph. This proves the pattern works.

Your Neo4j schema should use IBIS at the top level for issue tracking and adversarial alignment, with Toulmin structure inside each argument:

- `(:LegalIssue)` nodes with `standard_of_review`, `burden_of_proof`, `status`, and 768-dim embeddings
- `(:Position)` nodes with `party`, `brief_type`, `section_ref`, linked via `[:RESPONDS_TO]` to Issues
- `(:Argument)` nodes with Toulmin fields (`claim_text`, `grounds_text`, `warrant_text`, `qualifier`, `strength_score`) linked via `[:SUPPORTS]` or `[:OBJECTS_TO]` to Positions
- Cross-brief alignment via `[:OPPOSES]` between Positions and `[:REBUTS]` / `[:CONCEDES]` between Arguments
- `(:Authority)` nodes linked via `[:CITES]` with `[:DISTINGUISHES]` and `[:OVERRULES]` inter-authority relationships

For the Carneades framework — specifically designed for legal reasoning with **per-issue proof standards** (preponderance, clear-and-convincing, beyond-reasonable-doubt) and separate burden-of-production/burden-of-persuasion handling — consider adopting its proof standard properties on your Issue nodes even if you don't implement the full framework. This maps directly to appellate brief requirements where different issues carry different standards of review.

## Neo4j-native ontology beats OWL for a solo developer

Four approaches exist for jurisdiction-specific legal ontologies, and the evaluation is clear:

**OWL/RDF ontologies** (LKIF-Core, LegalRuleML, Akoma Ntoso) offer maximum expressiveness and formal inference but require specialized tooling (Protégé, SPARQL), are too verbose for LLM prompt injection, and represent significant implementation overhead. LKIF-Core has been demonstrated for German family law with Carneades, and Akoma Ntoso has been successfully converted to Neo4j property graphs (Italian Legislative Property Graph, 2024), but building a family-law-specific OWL ontology from scratch is weeks of work that a solo developer cannot justify. **No dedicated family law OWL ontology exists** in the published literature.

**Neo4j-native labeled property graphs** are the right choice. You already have 4 Neo4j databases. Standards of review, factor tests, burden frameworks, and jurisdictional variations all map naturally to nodes and relationships. A `(:StandardOfReview {name: "Abuse of Discretion", deference_level: "high"})` node connected via `[:APPLIES_TO]` to `(:SubDomain {name: "Child Custody"})` and `[:APPLIES_IN]` to `(:Jurisdiction {name: "California"})` is queryable, visualizable in Neo4j Browser, and trivially serialized into LLM prompt context via Cypher results. Vector indexes on concept nodes enable hybrid retrieval — semantic similarity search combined with graph traversal — which is exactly the pattern your existing RAG pipeline uses.

The **Neosemantics (n10s)** plugin bridges the gap: if you ever want to import existing OWL ontologies (like LKIF-Core modules) into Neo4j, `n10s.onto.import.fetch()` converts OWL classes to `:Class` nodes and `rdfs:subClassOf` to `:SCO` relationships, with inference support via `n10s.inference.nodesLabelled()`. This lets you start with a pragmatic property graph and optionally layer in formal ontology concepts later.

Supplement the Neo4j ontology with **flat JSON/YAML config files** for jurisdiction-specific prompt templates and LLM-injectable structured data (factor lists, standard-of-review descriptions). These are version-controlled alongside code and directly injectable into prompts — YAML shows **up to 54% higher accuracy** than JSON when parsed by LLMs in benchmarks.

## Provenance tracking requires citation anchors and a three-class source model

Reply briefs must track provenance across three document classes simultaneously: the opening brief (your prior arguments), the opposing brief (opponent's arguments), and legal authorities (case law, statutes). This is harder than standard RAG provenance, which typically tracks only retrieval from a knowledge corpus.

The most implementable pattern comes from Tensorlake's citation-aware RAG architecture: insert lightweight citation anchors (e.g., `‹c›2.1‹/c›`) into chunk text during ingestion, store spatial metadata (page, paragraph, bounding box) separately, and prompt the LLM to reference anchor IDs in its output. Resolution maps anchor IDs back to source documents, pages, and paragraphs. For your system, use anchors like `[OB:p12¶3]` for "Opening Brief, page 12, paragraph 3" and `[RB:p7¶2]` for "Respondent's Brief, page 7, paragraph 2."

In Neo4j, every `(:GeneratedText)` node connects via `[:SOURCED_FROM {page, paragraph}]` to the specific `(:Document)` it cites. This creates a queryable provenance graph where you can ask: "Which generated arguments in the reply brief cite the opposing brief's claims about custody?" — a query that flat citation metadata cannot answer efficiently.

Harvey AI's approach — decomposing generated responses into individual factual claims, then cross-referencing each against authoritative sources — is the gold standard for verification. Their system achieves approximately **0.2% hallucination rates**. The Citation-Enforced RAG paper (arXiv 2603.14170) formalizes this as a "source-first architecture" that embeds and retrieves only extracted source content, maintains span-level traceability throughout, and abstains when evidence is insufficient.

## Framework evaluation if you want external orchestration

If the hybrid approach eventually becomes unwieldy as you scale to 8+ document types, **LangGraph** is the framework to adopt. It has first-class Neo4j integration via the `langchain-neo4j` package, native support for graph cycles (critical for your refinement loops), multi-model orchestration (OpenAI, Anthropic, Google all supported), durable execution with crash recovery, and LangSmith observability with a free tier of 5,000 traces/month. It has **~25,000 GitHub stars** and is used by Klarna and Replit in production.

Avoid **CrewAI** despite its 44,600 stars and easier learning curve — its "black box" debugging is unacceptable for legal document generation where every citation must be verifiable. It also lacks native Neo4j integration. Avoid **AutoGen/AG2** entirely — the Softmax Data "Definitive Guide to Agentic Frameworks in 2026" (February 2026) explicitly states "AG2 is not production-ready for most enterprise use cases," and the Microsoft Agent Framework merger has fragmented the ecosystem.

If you stay with custom orchestration, **Pydantic AI** (15,000+ GitHub stars, MIT licensed) is worth evaluating as a lighter-weight complement. Its dependency injection pattern is ideal for passing Neo4j connections and retrieval strategies into agents, and its validated structured outputs prevent the malformed citations that plague legal AI systems. It composes naturally with YAML configs parsed into Pydantic models.

**Do not build an event-driven architecture.** Kafka, Redis Streams, and pub/sub patterns solve distributed systems problems that do not exist at single-developer scale. The Confluent and Solace examples are architecturally elegant but require message brokers, event schemas, and subscription management that represent 6–8 weeks of infrastructure work with no proportional benefit for your use case.

## Implementation roadmap for a solo developer

The entire expansion can be executed in approximately 5 weeks:

**Week 1** — Implement document-type-specific prompt templates for the existing 7-agent council plus simple Python dict/YAML pipeline configs. This immediately enables basic handling of different document types through prompt variation alone, which the MASS framework research confirms is the most token-efficient lever for changing multi-agent system behavior.

**Week 2** — Add Neo4j-native ontology nodes: document types, legal constraints, standards of review, factor tests, jurisdictional variations. Create the hybrid IBIS + Toulmin argument schema in a dedicated database. This establishes the graph infrastructure that the reply brief pipeline will use.

**Week 3** — Build argument extraction from briefs into structured Toulmin/IBIS nodes. Use your existing LLM agents to parse briefs into claims, grounds, warrants, and authorities, storing results in Neo4j. Implement argument alignment using sentence-BERT embeddings and graph traversal.

**Week 4** — Build the diff-and-respond pipeline for reply briefs: mischaracterization detection via NLI, constrained response generation with scope validation, and cross-document citation assembly with provenance anchors.

**Week 5** — Integrate everything: wire the reply brief pipeline into the hybrid config system, add pipeline configs for motions and declarations (which are simpler — primarily prompt variations on the existing pipeline), test end-to-end, and refine.

## Conclusion

The key architectural insight is that your expansion problem has three distinct layers that should not be conflated. Relationships and constraints are graph problems — solve them in Neo4j. Pipeline sequencing and parameters are configuration problems — solve them in YAML. Agent behavior adaptation is a prompting problem — solve it in templates. The reply brief challenge specifically is an argument-alignment-and-constrained-generation problem that requires purpose-built code, not configuration.

This hybrid avoids the two common failure modes: over-engineering (building a full workflow engine or event-driven architecture for a system that will have perhaps 6 document types) and under-engineering (trying to handle fundamentally different workflows through YAML configs alone). The fact that you already have 4 Neo4j databases, working multi-model orchestration, and a citation verification pipeline means your existing infrastructure handles the hard parts. The expansion is about making it configurable and building the argument alignment capability — not replacing what works.
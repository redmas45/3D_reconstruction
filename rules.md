# Global Engineering Standard: Professional Code Quality & Architecture Guidelines

This document outlines the expanded, world-class engineering standards required for building scalable, maintainable, and production-grade software ecosystems. These rules govern all codebases, from frontend applications and backend APIs to automated pipelines and data processing engines.

---

## 1. Name Elements for 2 AM Readability (Intent Over Brevity)
Names must clearly communicate intent, context, and responsibility. Code is read significantly more often than it is written; avoid clever shortcuts, non-standard abbreviations, or ambiguous single-letter variables.

* **Variable Names:** Use descriptive nouns that reflect the exact data held.
    * *Bad:* `t`, `c_list`, `f_data`
    * *Good:* `elapsed_time_seconds`, `active_claims_list`, `processed_frame_data`
* **Function/Method Names:** Use strong action verbs that explicitly describe the operation.
    * *Bad:* `proc()`, `handle_doc()`, `calculate()`
    * *Good:* `process_incoming_fnol_submission()`, `extract_ocr_document_metadata()`, `calculate_sla_compliance_rate()`
* **Class Names:** Use clear nouns or noun phrases representing an entity, component, or conceptual domain.
    * *Good:* `ClaimsWorkflowAutomationEngine`, `DigitalWalletPassGenerator`, `StripeLedgerReconciliationService`

## 2. Single Responsibility Principle (One Function, One Job)
A function or class should have one, and only one, reason to change. If your explanation of a function requires the word "and", it is a candidate for decomposition.

* **Atomic Operations:** Break complex logic into tightly focused, re-usable functions that perform a single operational primitive.
* **Integration vs. Action:** Separate functions that orchestrate workflow steps from functions that perform the actual processing/computation.
* **Maintainability Impact:** Isolating tasks simplifies writing comprehensive unit tests, accelerates debugging, and decreases the blast radius of structural changes.

## 3. Total Elimination of Magic Numbers and Literals
Hardcoded literals (strings, integers, floats) introduce fragile dependencies, hinder global refactoring, and obscure business rationale.

* **Named Constants:** Every literal that possesses semantic weight must be declared as a named constant or structured enum.
* **Centralized Configuration:** Infrastructure values (timeouts, retry limits, file size caps) belong in config environments or strict registry objects.
* **Example Implementation:**
    * *Bad:* `if (claim.amount > 500) { route_to_handler(); }`
    * *Good:* `if (claim.amount > MAX_AUTO_APPROVE_CLAIM_VALUE_GBP) { route_to_handler(); }`

## 4. Explicit and Defensive Error Handling
Production software must fail gracefully, predictably, and informatively. Swallowing errors or assuming happy-path execution is strictly forbidden.

* **Specific Exception Handling:** Never catch generic top-level exceptions (e.g., raw `Exception` or `Error`) unless logging and re-throwing. Catch exact operational exceptions (e.g., `TimeoutError`, `ValidationError`).
* **Clean Failure States:** Ensure database connections, open file handles, and stream resources are guaranteed to close using transactional boundaries or resource managers (`try-catch-finally`, `with` blocks).
* **Sufficient Auditing:** Log exceptions with complete context (transaction IDs, error stack traces, payload snapshots) while sanitizing sensitive user data or PII. Surface clean, actionable, non-technical messages to end users.

## 5. Elimination of Unintended Side Effects (Pure Functions)
State mutations hidden within standard functions create unpredictable runtime tracking, complex race conditions, and difficult-to-reproduce bugs.

* **Immutability by Default:** Functions should treat input arguments as read-only. Avoid modifying arrays, objects, or data models passed by reference.
* **Explicit State Changes:** If a function must mutate state or interact with the external environment (I/O, databases, file systems), clearly isolate it or structure it as an explicit state-transition method within an orchestrator class.
* **Data Integrity:** Return a newly constructed data model or payload instead of modifying existing data structures.

## 6. Comprehensive Strict Typing & Schema Enforcement
Type definitions act as a compile-time firewall against runtime exceptions and form a living, machine-enforced documentation layer.

* **Signature Completeness:** Every function must explicitly annotate all argument types and the exact return type.
* **No Dynamic Escapes:** Avoid structural fallback types (like Python's `Any` or TypeScript's `any`) unless dealing with un-parsed wire protocols at an input boundary.
* **Data Model Validation:** Enforce strict runtime data schemas (using tools like Pydantic, TypeScript interfaces, or Mongoose schemas) at all persistence and ingestion layers.

## 7. Informative Comments (Documenting the "Why", Not the "What")
Code explains the mechanism; comments must explain the underlying business intent, architectural constraints, or mathematical rationale.

* **Code Self-Documentation:** If the code is complex, refactor the naming, structure, or abstractions to make it legible before writing a descriptive comment.
* **Strategic Explanations:** Use comments exclusively to justify non-obvious engineering decisions, edge-case fixes, upstream architectural limitations, or complex business logic overrides.
* **Drift Prevention:** Outdated comments are more dangerous than no comments. Ensure comments are updated in lockstep during every refactoring phase.

## 8. Flat and Compact Call Stacks (Early Returns & Micro-Functions)
Deeply nested, expansive functions tax cognitive load and obscure logical exit paths. Aim to keep code readable from top to bottom.

* **Physical Size Limits:** Strive to keep functions within 30 lines of execution logic. If a function stretches beyond a screen-height, abstract its internal sub-blocks.
* **Guard Clauses & Early Returns:** Eliminate cascading `if-else` pyramids by immediately checking preconditions and returning or throwing errors early.
* **Cyclomatic Complexity Cap:** Maintain a maximum nesting depth of 2 levels. Use loop-extraction techniques or functional map/filter operations to keep blocks flat.

## 9. Zero-Trust Ingestion (Rigorous Boundary Validation)
All data crossing an architectural boundary (API endpoints, webhook targets, public methods, uploaded files, configuration strings) must be treated as hostile and untrusted.

* **Sanitization and Typing:** Validate structure, types, constraints, ranges, and characters immediately upon arrival at the boundary layer.
* **Structural Defensiveness:** Reject invalid payloads immediately at the perimeter before allowing data to traverse downstream to application logic or storage tiers.
* **Fail-Fast Architecture:** Catching anomalies early prevents corrupt database tracking, injection vulnerabilities, and deep-stack processing crashes.

## 10. Intentional Abstraction (The Rule of Three)
Premature abstraction is a leading driver of bloated, rigid, and over-engineered architectures. Write code for clarity today, and abstract for reuse tomorrow.

* **Duplicate Twice:** Copying and pasting code or logic exactly twice is perfectly acceptable if it preserves clear separation between two separate business domains.
* **Abstract on the Third Recurrence:** Only when a pattern or logical block emerges for the *third* time should you design and implement a generalized abstraction, helper module, or shared utility interface.
* **Domain Preservation:** Ensure you are abstracting shared *structural realities*, not coincidental similarities that happen to look alike today but will evolve in entirely different directions tomorrow.

## 11. Modular Architecture, Classes, and File Size Budgets
Large files hide design flaws, slow review, and make demo-critical changes risky. A production module must have an explicit ownership boundary and should not become a dumping ground for unrelated workflows.

* **File Size Budget:** Keep source files under 500 lines by default. A file may exceed 500 lines only when it is a deliberate registry, generated artifact, schema, migration, or dense test fixture. Files above 800 lines require active refactoring into domain modules before new feature work is added.
* **Class Boundaries for Stateful Domains:** Use classes for cohesive domain services that own state, dependencies, or policy, such as orchestration engines, response grounders, retrieval services, adapter scanners, and setup planners. Constructor dependencies should make external collaborators explicit.
* **Pure Helpers Stay Small:** Do not force OOP where a stateless pure function is clearer. Pure parsing, formatting, validation, and scoring helpers are acceptable when they are short, typed, side-effect free, and grouped in a focused module.
* **Orchestrator Thinness:** Orchestrators should coordinate workflow steps, not contain all business logic. Move retrieval ranking, product formatting, action repair, cache policy, prompt assembly, and browser action grounding into separately testable services.
* **No Mega-Modules:** If a module contains multiple unrelated sections, split by domain capability instead of by technical convenience. Example: product response grounding belongs in a product response service, not in a voice pipeline orchestrator.
* **Performance Reality:** Classes and objects do not automatically make software faster or more memory efficient. Performance comes from bounded data structures, avoiding repeated I/O, avoiding unnecessary copies, lazy loading expensive dependencies, and using clear ownership so caches and resources are managed intentionally.

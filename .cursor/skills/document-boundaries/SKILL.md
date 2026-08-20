---
name: document-boundaries
description: Documents the strict constraints, invariants, and silent failures of a specific module or directory.
---

# Boundary Documentation Workflow

When invoked to document a module, do not document basic functions that can be inferred by reading the source code. Instead, generate a markdown file detailing the module's constraints.

## Required Sections

1. **Always:** List required patterns for this module.
2. **Ask First:** List high-risk areas requiring human intervention before modification.
3. **Never:** List hard system constraints and anti-patterns.
4. **Silent Failures & Gotchas:** Document counter-intuitive behaviors, unhandled edge cases in external APIs, or data flow invariants that might trick another agent.

## Execution
Review the codebase for the specified module, identify these four categories, and generate the documentation file in the module's root directory or the main `docs/` folder, appending a link to the central orientation index.
---
name: create-adr
description: Generates an Architecture Decision Record (ADR) when a new dependency, pattern, or system boundary is introduced.
---

# Architecture Decision Record Workflow

When invoked, you must create a new Architecture Decision Record in the `docs/adr/` directory.

## Step 1: Naming the File
Generate a filename using the format: `YYYY-MM-DD-short-descriptive-title.md`.

## Step 2: Drafting the Content
Ensure the document strictly follows this structure:

### Context
Write a maximum of two sentences explaining the specific technical or business problem being solved.

### Decision
Clearly state what is being built, implemented, or chosen. Include specific library names, architectural patterns, or API changes.

### Consequences
List the new constraints for future agents and developers. Explicitly state what practices are now required and what practices are now deprecated due to this decision.

## Step 3: Updating the Index
After creating the ADR, open `docs/architecture/README.md` and add a link to the new ADR in the Orientation Table.
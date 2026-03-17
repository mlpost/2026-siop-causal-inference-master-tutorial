---
name: update-team-readme
description: Intelligently update TEAM_README.md when code, architecture, or documentation changes occur in the SIOP 2026 Master Tutorial repository.
---

# Update Team README

This skill maintains the TEAM_README.md file by detecting changes in the codebase and updating the appropriate sections to keep documentation current and accurate.

## When to Use

- Use this skill when new methods are added to CausalDiagnostics or CausalInferenceModel classes
- This skill is helpful for updating documentation after architecture changes
- Use when dependencies are added, removed, or version-bumped in requirements.txt
- This skill is helpful for reflecting new notebook cells or sections in scenario2_workshop.ipynb
- Use when file sizes, line counts, or repository structure changes
- This skill is helpful for documenting new gotchas, bug fixes, or best practices discovered
- Use when new analysis methods (GEE, survival, DML) are implemented
- This skill is helpful for updating variable glossaries or data schema changes

## Instructions

### Step 1: Analyze the Change Context
- Examine recent commits, file modifications, or user-described changes
- Identify which sections of TEAM_README.md are affected:
  - Section 1: Project summary (if purpose/scope changes)
  - Section 2: Repository map (if files added/removed/renamed)
  - Section 3: Quick start (if setup process changes)
  - Section 4: Architecture (if classes/methods change)
  - Section 5: Synthetic dataset (if data generation changes)
  - Section 6: Notebook walkthrough (if cells added/modified)
  - Section 7: Common gotchas (if new issues discovered)
  - Section 8: Data regeneration (if process changes)
  - Section 9: Glossary (if new terms introduced)

### Step 2: Section-Specific Update Guidelines

**For Section 2 (Repository Map):**
- Update file line counts by examining actual files
- Add new files with their purpose and size
- Remove deleted files
- Update the file size reference table with current line counts

**For Section 4 (Architecture Deep Dive):**
- Add new methods to the appropriate class method tables
- Update method counts in the summary
- Describe new functionality in the method group descriptions
- Update internal pipeline descriptions if workflow changes

**For Section 6 (Notebook Walkthrough):**
- Update cell counts (total, code, markdown)
- Add new sections to the section table
- Update the typical analysis pattern code if methods change
- Reflect new analysis approaches in the walkthrough

**For Section 7 (Common Gotchas):**
- Add newly discovered issues with clear explanations
- Update existing gotchas if solutions improve
- Maintain the issue/explanation table format

**For Section 9 (Glossary):**
- Add new technical terms with precise definitions
- Update existing definitions if understanding improves
- Maintain alphabetical order within categories

### Step 3: Preserve Documentation Style
- Maintain the technical, detailed tone throughout
- Use tables extensively for structured information
- Keep the teaching-focused perspective (this is workshop material)
- Preserve code examples and command snippets exactly
- Use consistent formatting: **bold** for emphasis, `code` for variables/methods
- Maintain the existing markdown structure and hierarchy

### Step 4: Update Cross-References
- Ensure method names match exactly between sections
- Verify file paths and names are consistent
- Update any line count references to match current reality
- Check that variable names in examples match the actual codebase

### Step 5: Validation Checks
- Verify all mentioned methods actually exist in the codebase
- Confirm file sizes and line counts are accurate
- Ensure new dependencies are reflected in the dependencies table
- Check that notebook cell counts match the actual notebook
- Validate that code examples are syntactically correct

### Step 6: Maintain Context and Purpose
- Remember this is documentation for teammates to quickly understand the codebase
- Keep the 30-second summary accurate and current
- Ensure the "What This Project Does" section reflects any scope changes
- Maintain the focus on causal inference methodology and SIOP 2026 workshop context

### Step 7: Ask for Clarification When Needed
- Use the ask questions tool if the scope of changes is unclear
- Ask for confirmation if major structural changes are detected
- Clarify if new methods should be documented in detail or just mentioned
- Confirm if changes affect the workshop teaching flow

### Best Practices
- Always examine the actual files rather than assuming content
- Preserve the existing detailed, reference-style documentation approach
- Update the "Last updated" date at the bottom
- Maintain the balance between comprehensiveness and readability
- Keep technical accuracy as the top priority
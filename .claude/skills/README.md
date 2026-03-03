# Claude Code Skills for Python Sprint Zero

This directory contains custom Claude Code skills tailored to this project's strict architectural and coding standards.

## Installed Skills (Tier 1 - Highest Value)

### 1. Hexagonal Architecture Feature Scaffolder
**Directory:** `hexagonal-architecture-scaffolder/`
**Trigger phrases:** "create a new feature", "scaffold a feature", "add a new domain entity", "create a new API endpoint"

**Purpose:** Guides Claude through creating complete features across all hexagonal architecture layers (domain → application → infrastructure → interface) with proper dependency injection, layer boundaries, and test coverage.

**Key features:**
- Step-by-step scaffolding across all layers
- Enforces layer boundary rules
- Generates repository interfaces in domain, implementations in infrastructure
- Creates use cases with dependency injection
- Generates DTOs and controllers
- Creates corresponding test files
- References existing coconut feature as example

**Supporting files:**
- `references/existing-coconut-example.md` - Complete example from codebase
- `references/layer-dependency-rules.md` - Import dependency rules

---

### 2. Test Generator with One-Assertion Rule
**Directory:** `test-generator/`
**Trigger phrases:** "generate tests", "create tests for", "write test cases", "add test coverage"

**Purpose:** Generates tests following the project's mandatory one-assertion-per-test rule with proper naming (`test_should_X_when_Y`), Arrange-Act-Assert structure, and appropriate mocking.

**Key features:**
- Enforces one assertion per test (critical project rule)
- Generates descriptive test names following pattern
- Creates Arrange-Act-Assert structure
- Provides testing patterns by layer (domain, application, infrastructure, interface)
- Shows how to properly mock dependencies
- Includes examples of exception testing, fixtures, and HTTP testing

**Supporting files:**
- `references/test-examples-from-codebase.md` - Actual test examples from project

---

### 3. Self-Documenting Code Refactorer
**Directory:** `self-documenting-refactor/`
**Trigger phrases:** "remove comments", "make code self-documenting", "refactor to eliminate comments", "improve code clarity"

**Purpose:** Enforces the project's strict NO COMMENTS policy by helping refactor code to be self-explanatory through expressive naming instead of comments.

**Key features:**
- Identifies comments in code
- Suggests refactoring strategies (extract to well-named function, expressive variables, boolean conditions, named constants)
- Provides before/after examples
- Teaches naming conventions for functions, variables, constants, booleans
- Includes common refactoring patterns

**Supporting files:**
- `references/refactoring-examples-from-codebase.md` - Examples from the codebase
- `scripts/find_comments.py` - Executable script to scan for comments in codebase

**Usage:**
```bash
# Find all comments in source code
python .claude/skills/self-documenting-refactor/scripts/find_comments.py src/

# Expected output: 0 comments
```

---

### 4. Property-Based Testing
**Directory:** `property-based-testing/`
**Trigger phrases:** "property-based testing", "roundtrip test", "serialization test", "test encode/decode", "hypothesis test"

**Purpose:** Provides guidance for property-based testing across multiple languages. Helps identify patterns where PBT provides stronger coverage than example-based tests.

**Key features:**
- Automatic detection of testable patterns (serialization, validation, normalization)
- Property catalog with formulas (roundtrip, idempotence, invariant, commutativity, etc.)
- Decision tree for task-based guidance
- Language-specific library recommendations
- Smart contract testing support (Echidna, Foundry)

**Supporting files:**
- `references/design.md` - Property-Driven Development approach
- `references/generating.md` - Test generation patterns
- `references/libraries.md` - PBT libraries by language
- `references/refactoring.md` - Refactoring for testability
- `references/reviewing.md` - Quality checklist
- `references/strategies.md` - Input generation strategies

---

### 5. Test-Driven Development
**Directory:** `test-driven-development/`
**Trigger phrases:** "implement feature", "fix bug", "write code", "TDD", "red-green-refactor"

**Purpose:** Enforces the TDD methodology with strict Red-Green-Refactor cycle. Ensures no production code is written without a failing test first.

**Key features:**
- Iron Law: No production code without failing test first
- Red-Green-Refactor cycle enforcement
- Common rationalizations and rebuttals
- Verification checklist for each phase
- Debugging integration guidance

**Supporting files:**
- `testing-anti-patterns.md` - Common testing pitfalls to avoid

---

### 6. Systematic Debugging
**Directory:** `systematic-debugging/`
**Trigger phrases:** "debug", "fix bug", "test failure", "unexpected behavior", "error", "not working"

**Purpose:** Provides systematic debugging methodology. Ensures root cause investigation before attempting fixes.

**Key features:**
- Four-phase process (Root Cause → Pattern Analysis → Hypothesis → Implementation)
- Multi-component system diagnostics
- Evidence gathering before fixes
- Architecture questioning after 3+ failed fixes
- Red flag detection for premature fix attempts

**Supporting files:**
- `root-cause-tracing.md` - Backward tracing through call stack
- `defense-in-depth.md` - Multi-layer validation patterns
- `condition-based-waiting.md` - Replace timeouts with condition polling

---

### 7. Verification Before Completion
**Directory:** `verification-before-completion/`
**Trigger phrases:** "done", "complete", "fixed", "passing", "commit", "create PR", "finished"

**Purpose:** Requires running verification commands and confirming output before making any success claims. Evidence before assertions, always.

**Key features:**
- Gate function: Identify → Run → Read → Verify → Claim
- Common failure patterns and what each claim requires
- Rationalization prevention table
- Red flag detection for premature claims

---

## How Claude Code Uses Skills

### Automatic Invocation
Claude Code automatically loads and applies skills when trigger phrases are detected in user requests. For example:
- User: "Create a new user feature" → Loads Hexagonal Architecture Scaffolder
- User: "Generate tests for the UserService" → Loads Test Generator
- User: "This code has comments, make it clearer" → Loads Self-Documenting Refactorer
- User: "I need to debug this test failure" → Loads Systematic Debugging
- User: "Implement the login feature" → Loads Test-Driven Development

### Manual Invocation
You can also explicitly invoke skills:
- `/hexagonal-architecture-scaffolder` - Load architecture scaffolding guidance
- `/test-generator` - Load test generation guidance
- `/self-documenting-refactor` - Load code refactoring guidance
- `/property-based-testing` - Load property-based testing guidance
- `/test-driven-development` - Load TDD methodology guidance
- `/systematic-debugging` - Load debugging methodology guidance
- `/verification-before-completion` - Load completion verification guidance

### Progressive Disclosure
Skills use progressive disclosure:
- Main `SKILL.md` contains core guidance (1,500-2,000 words)
- `references/` directory contains detailed examples and patterns
- `templates/` directory contains ready-to-use code templates
- `scripts/` directory contains executable helper scripts

Claude Code loads the main skill first, then references supporting files as needed.

## Skill Development Guidelines

These skills follow Claude Code best practices:

1. **Clear trigger phrases** - Specific, concrete phrases users would naturally say
2. **Imperative writing** - Direct instructions ("Create", "Add", "Use")
3. **Progressive disclosure** - Main guidance is concise, details in references
4. **Project-specific** - Tailored to this project's unique rules and patterns
5. **Complementary to testing** - Skills teach generation, automated tests validate

## Project-Specific Rules Enforced by Skills

All skills enforce these critical project rules:

- **NO COMMENTS** - Code must be self-documenting
- **One assertion per test** - Each test function has exactly ONE assertion
- **Layer boundaries** - Respect hexagonal architecture import restrictions
- **Dependency injection** - Always use Lagom DI, never direct instantiation
- **100% test coverage** - Every function must have tests
- **Type hints** - All function signatures must have type hints
- **`__init__.py` required** - All Python packages must have this file

## Testing Skills

To test if skills are working:

1. **Try trigger phrases in Claude Code:**
   - "Create a new product feature"
   - "Generate tests for ProductService"
   - "Remove comments from this code"
   - "Debug this test failure"
   - "Implement the search feature"

2. **Verify skill loads:**
   - Claude should reference the skill in its response
   - Guidance should match skill content

3. **Check skill is applied:**
   - Generated code follows skill patterns
   - Layer boundaries respected
   - Tests have one assertion
   - No comments in generated code

## Extending Skills

To add more skills:

1. Create directory: `.claude/skills/skill-name/`
2. Add `SKILL.md` with YAML frontmatter and instructions
3. Optionally add `references/`, `templates/`, `scripts/` subdirectories
4. Use specific trigger phrases in description
5. Keep main skill body concise (1,500-2,000 words)
6. Update this README

## Additional Recommended Skills (Tier 2)

Future skills to consider:

- **Observability Pattern Injector** - Structured logging, metrics, tracing patterns
- **CDCT Test Generator** - Consumer-Driven Contract Testing patterns
- **Dependency Injection Scaffolder** - Lagom container setup patterns
- **API Route Generator** - Complete endpoint scaffolding
- **Test Coverage Analyzer** - Identify untested code gaps

## Resources

- [Claude Code Skills Documentation](https://docs.anthropic.com/en/docs/claude-code)
- Project coding standards: `.claude/CLAUDE.md`

## Verification

Check all skills are present:
```bash
ls -la .claude/skills/*/SKILL.md
```

Expected output: 7 SKILL.md files (one for each Tier 1 skill)

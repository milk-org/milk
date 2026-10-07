# Project: milk

## Critical Rules

Previously, a poorly-prompted AI has developed a large amount of features with poor consideration for existing API capabilities.

Your goal is to strive for concision and consistency.
Always point to duplicate or similar:
- Infrastructure
- Internal features
- User-facing features
- User-facing documentation (`docs/`)
- AI-facing documentation (`.agents/`)
`docs/` content shall have precedence over `.agents` content.
Never duplicate between these folders, always link the agent to the human doc.

You are TO NEVER use any `git` feature that:
- manipulates branch, HEAD, tag pointers
- interacts with github
- overwrites any file in the working tree

## Scope discipline

Do not scope-creep: implement exactly what was asked, nothing more
(no extra abstractions, config knobs, or "while I'm at it" fixes).
Do not burn tokens exploring code beyond what's needed for the
literal request.
If a request is ambiguous or under-specified, do not guess and
expand scope to compensate. Instead ask, e.g.:
- "Should I spend tokens exploring the code to get this exactly
  right, or do you already know the constraint I'm missing?"
- "Should I do exactly this literally, and just flag edge cases I
  see, rather than handling them?"

## More

Read `AGENTS.md` for full onboarding context and
follow its reading order (section 2). This has less precendence than the "Critical Rules" hereabove.

## Key Directories

- `.agents/rules/` — always-on guardrails
- `.agents/skills/` — deep-dive instruction sets
- `.agents/workflows/` — on-demand task templates
- `src/milk_module_example/` — code templates
- `docs/` — user and developer documentation

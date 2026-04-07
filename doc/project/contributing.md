# Contribution policy overview

Welcome to Milk !

As we're preparing to go through the CACAO++ initiative, we are also paving way for onboarding more developers into the project.
Our goal is to reach self-sustainability of community-promoted development of the package by late 2027.
If you're reading this, you probably have a sense of how the package is organized and layered, and there will be much more on-point documentation on the matter.

> [!NOTE]
> This document should become the entrypoint for project policy, and establishes some ground for future development, both for CACAO as an engineering and research project in its own right, and as a software project in particular.
> It's also a first version, and I should expect to modify this frequently in upcoming times.

## TOC

- `code_guidelines.md`: style and in-code documentation rules.
- `git_guidelines.md`: branching and pull request process.
- `ai_guidelines.md`: philosophical comments about the use of AI agents for work in MILK.
- `project_trace.md`: keep trace of project milestones, publication (and policy thereof), contributing teams and groups, and development directions.

## Communication channels

Preferred public channels:
- gitter for short bump message
- github repository issues (with either `discussion` or a more appropriate label).
  - Preferred for developer interaction.
  - We may find an alternate means of communication if requests from non-developer users become significant.
of course, maintain your best public & professional persona, and avoid sharing confidential information.

We strive for the reduction of noise, so please avoid using emails or DMs unless necessary, and similarly avoid tagging people excessively.
It is my experience that people eventually mute means of communication that result in too much spam notifications.

## Peoples' roles

Generally, we'll refer to collaborators across 4 role-tiers: user, module dev, core dev, and maintainer.

### User:

You want to use CACAO for you AO computing needs ! Please refer to the README.md for installation instructions and pointers to the tutorials. Then:
- Learn about communication channels: use GitHub Issues for bug reports and feature requests, and GitHub Discussions (or the project's Slack/mailing list) for questions and general conversation.
- When reporting a bug, include: OS/distro, compiler version, CMake flags used, and a minimal reproducing case.
- Please read the tutorials and documentation as necessary.
- Have fun !

### Module developer:

MILK/CACAO is based on a dynamic structure where the core features can be expanded with dynamic modules. The core libraries offer data structures, primitives, and templates to do this easily and efficiently.

Your code can be in a separate repo entirely, a fork of MILK, directly in MILK. You will also have a lot of flexibility for licensing. If you expect (and we'd love to have it) to merge back your code:
- You shall maintain your module's documentation
- Make sure your module compiles with all the flags it may use
- Make sure you module is found, linked, and usable the way you intend to when running the MILK build pipeline
- As much as possible, comply with other software quality requirements. You'll be helped !

### Core engine developer:

You're now contributing to the milk core engine, macros, common routines, so as to fix or expand them for your application. The core engine refers to code that is used and distributed across all the expansion modules.
- You're expected to discuss your features and approach with the maintainers
- Please document your work (traceable commits and docs do count here)
- Provide testing, argue for significant changes (including of course your own intended application)
- Comply with all the software quality rules, and integration tests (e.g. compiling and testing features in all supported combinations, etc.)

### Maintainer

- You get to make decisions.
- You must promise to help everyone else and to review your code and to discuss any new things within the reasonable limits of time and sanity.
- Besides critical project management decisions, you should strive to keep yourself out of the critical path.


## Development process philosophy

I suggest we work towards _suggesting_ a minimum noise environment to make everyone's experience as pleasant and efficient as possible.

Key points are (to be better sorted in the future):

- __Be concise and precise__ in code, documentation, and project communication.
- __Less is more__: a concise codebase, factorized primitives and libraries, naming variables and functions so that its all self-explanatory. Less work scrolling files, less oculary fatigue. The idea is the following: if you're adding long comments, maybe you should restructure the code to make the intent clear? Sometimes, but rarely, it does need to be convoluted, but need we cite Donald Knuth about "__premature optimzation__" ?
- __Priority goes to automation.__ Automated, deterministic, reproducible and invertible workflows are better than pages of dos-and-donts (there are still be dos-and-donts). _If it's not in a script, can't for sure expect it_: __if it is in the workflows, then it is policy__.
- __Help others help you__ in issue reporting, prioritize conveying information to describe and reproduce (what, where, when) and try and offer a minimal reproducing example unless impractical.
- __Follow the workflow__ for git(hub) in particular. It's detailed in connex files.
- __Respect work and time__ by keeping exchanges and contributions professional, respectful, on-topic, and well scoped.

## More

- Follow `code_guidelines.md` for style and in-code documentation rules.
- Follow `git_guidelines.md` for branching and pull request process.
- Run required workflows and hooks before opening or updating pull requests.

## About AI

In this day and age, it seems we need to take a stance. Contributing with AI agents in MILK and CACAO is acceptable, under the following conditions (on top of all the other requirements that apply to content submitted to the repository).

These specific rules are warranted by the enormous power that is wielded. This enables modifications and submissions with a scope and speed that we haven't seen before; therefore, special rules apply, see `ai_guidelines.md`.
The executive summary is the following:
- Be reasonable
- Maintain accountability
- Maintain traceability
- Account for other people not using AI like you would -- including reviewers, other core engine dev, etc.

## Information: about making modules

You should strive to minimize your dependencies, both internal and external.

The whole idea is to follow the templatization designed to quickly develop, embed, include, configure and distribute new features:
- Start from `milk_module_example` in `src/` — it is the canonical skeleton.
- Each module lives in its own directory under `src/` (or in a separate repo/submodule).
- Provide a `CMakeLists.txt` that integrates with the top-level build via the `EXTRAMODULES` mechanism.
- Include at minimum: a README, a CLI registration function, and one or more testable functions.
- See `doc/templatemodule/` for annotated examples.

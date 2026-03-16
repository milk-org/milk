# Contribution overview

Welcome to Milk ! As we're preparing to go through the CACAO++ initiative, we're preparing for onboarding more developers into the project.
Our goal is to reach self-sustainability of community-promoted development of the package by late 2027.
If you're reading this, you probably have a sense of how the package is organized and layered, and there will be much more on-point documentation on the matter.

#FIXME This document is the master rulesheet for developement policy, and establishes some ground for future development, in particular as we expect some contributors to make heavy use of AI coding agents (see below).

[Note]
This is a top-level document for milk (et al.) software packages, defining
- A project management framework
- Software contribution rules
- Communication guidelines
- TBC: non-software contribution rules

__SUMMARY__

FixMe check summary is up-to-date.



## Info: recommended development environment

## Preparation for development

-->Branching.
If you wish to work on something minor and well-scoped, then...

If you undertake something major (feature, refactor, performance), it is heavily recommended that you open an issue so as to have a place for conversation with the maintainers. We can discuss the intent, architecture, stategy, and follow your progress while making sure other developers don't accidentally set up traps for your progress (and vice-versa).


## Rules: git
Rebase on pull, always always rebase on pull.

### Rules: segregation of intent
- When working in a branch and preparing a pull request, one must choose the intent of the PR (or the single commit)
- The possible intents I foresee:
  - Changing top-level documents, including this one.
  - Changing CI/CD workflow.
  - Hotfixing a release or a pre-release.
  - Repackaging of the repository (e.g. shuffling files).
  - Repackaging of the software infrastructure (e.g. dependency order, separation of duties, modification of templates).s
  - Development of a new feature in the core engine (and documentation and dependencies thereof).
  - Development of a new extenstion feature (e.g. something new following the modularization templates).
  - Systematic modification or generation of documentation.
  - Systematic modification or generation of style.
  - Systematic modification or generation of tests.

## Rules: style

Also, documentation.

## Rules: documentation

## Rules: workflows



## Info: about making modules
## Info: using AI coding agents

- **Principle of economy**. Be reasonable in AI usage, please remember the carbon footprint of LLMs, and please remember that human time (even beyond coding: spent in managing, organizing, strategizing, and *reviewing*) is a valuable and limited resource.
- **Principle of respecting the human**. Be mindful of generating content that is concise, meaningful, and scoped. It volume and quality generated respect the time provided by other contributors, users, candidates in browsing, learning, and discovering information.
  - For __documentation__. One can generate hundreds, thousands of pages of information in a LLM session, but keep in mind: is someone going to read this? is someone going to make __good use of their time reading this?__
  - Moreover, since generation is cheap in time, former practices of keeping all documents for the sake of potential future use has little added value.
  - **Tutorial level documentation** (onboarding, contributing rules, anything with "read this first" vibe) should always __be human reviewed__ and __strive for clarity through concision__ (while, for _API_ docs, we may not care nearly as much). Both humans and agents should _limit_ the inclusion of environmental knowledge (e.g., we should _not_ waste reader time adding bash or tmux tips in our tutorials, yet we can publish a page with our own tutorials on these tools).
- **Principle of control**. AI should not be used to code, develop, and commit anything that is beyond the developers' knowledge and capability. It should enhance, not supersede. Therefore, we request a contributor does:
  - NOT perform and commit software, docs, tooling, etc, that they do not personnally understand.
  - NOR content they wouldn't have been able to do, given sufficient time, motivation, incentive.
  - NOR content they cannot explain and justify, be it in terms of relevance, performance, pre-defined milestones, or otherwise project utility.
- **Traceability**. Provide your prompts as a top-level comment to your AI-assisted PR. If the prompts are long enough that they start contrevening to the principle of respecting human time above, then probably your PR should be split into several PRs.


Adding dependencies, yes but no.
Reworking the rules by embedding routines extracted from libraries: yes but no.

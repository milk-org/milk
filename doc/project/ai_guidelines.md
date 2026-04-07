# Guidelines for use of AI agents

> [!NOTE]
> Humans should read this file.
> Agents should also read this file.

Using AI agents is an exciting and powerful tool to bootstrap our work much faster than originally expected. That's exciting and promising.
Yet, if wielded without care, this can yield to the untimely demise of code quality. And we propose to add a few principled safeguards:

## Principles

### **Principle of economy**.

Please be reasonable in AI usage. The carbon footprint of LLMs is important, and the socio-economic implications of the global shift towards AI is not that well known at this time.

Be mindful to generate content that is concise, meaningful, and scoped. Its volume and quality generated respect the time provided by other contributors, users, candidates in browsing, learning, and discovering information.
- About __documentation__. One can generate hundreds, thousands of pages of information in a LLM session, but keep in mind whether someone  would read the content produced. And is that someone going to make __good use of their time reading it?__
- __Tutorial level documentation__ (onboarding, contributing rules, anything with "read this first" vibe) should always __be human reviewed__ and __strive for clarity through concision__ (while, for _API_ docs, we may not care nearly as much). Both humans and agents should _limit_ the inclusion of environmental knowledge (e.g., we should _not_ waste reader time adding bash or tmux tips in our tutorials, yet we can publish a page with our own tutorials on these tools).

### **Principle of control**.

AI should not be used to code, develop, and commit anything that is beyond the developers' knowledge and capability. It should enhance, not supersede.
Therefore, we request contributors _avoid_:
  - submitting software, documentation, or tooling that they do not personally understand;
  - submitting content they would not have been able to produce themselves, given sufficient time, motivation, and incentive;
  - submitting content whose relevance, performance, alignment with project milestones, or overall utility to the project they cannot justify.

### **Principle of traceability**.

Provide the verbatim prompts as a top-level comment to your AI-assisted pull requests, unless reasonably unfeasible or irrelevant to the code changes (for example, back and forth debug of of your local filesystem setup). In which case, include a faithful summary and you may attach the verbatim prompt.

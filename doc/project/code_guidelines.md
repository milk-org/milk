Follow these code style and documentation rules exactly.
# Coding guidelines.

This is a brief onboarding into our coding rules.

## Automation first

We have and will be implementing more and more automation scripts to review and automatize software quality.
Automations are designed to be _deterministic, extensive, and reproducible_, and should always produce the same output (this has been an issue before where toggling between two linters was not involutive).

We rely on a mature and standard tool, `pre-commit`, which is entirely configured by `.pre-commit-config.yaml`:
- Install `pre-commit` with `apt` or `pip`
- Activate it by running `pre-commit install` within the repository.
This action installs git _hooks_, ie. actions that are run automatically when some `git` actions are being performed.

Now, when you run a commit, `pre-commit` will run on all modified files and perform a number of actions:
- if the commit is non-compliant, the commit is blocked.
- for style/linting and other things, hooks will modify files into compliance automatically.
  - but they will _not_ be stage; add them back to the git index
  - and commit again. This time, pre-commit will pass.

You can bypass pre-commit and still perform a commit. Use `git commit -n|--no-verify`.

Since its local, we cannot force you to install and run `pre-commit`. However, we run a github workflow that performs the same action upon push and pull-request (the workflow literally installs and runs pre-commit, which ensures 1:1 matching between the github repo and the local sanitization).

A commit or feature branch should never mix formatting changes and functional changes, as it obscures traceability entirely.

## Recommendations on development environment


## Code organization and editing discipline

Unfortunately, there are many things beyond what can be automated, so here's a list of rules.

Keep something in mind for style: we want to do everything to allow a human to skim the codebase and understand its _intent_ and _purpose_ as quickly as possible.


### Public header, internal header, C files.

Every folder / module should consolidate its function definitions into two __function__ header files:
- One for the external API of the module.
- One for private functions / common dependencies that are used in many C files in the module. This header will not be exported at install time.

And possibly a couple more public header files, considering:
- A separate header file for one large struct definitions or multiple small definition (and similarly for classes in the case of a C++ extension)
- A separate header file for the definition of constants, enums, etc. that are relevant to the module
- A separate header file that defines exported macros header file, when the module provides significant features by means of macro-ing.

### In-code documentation

__File-Level Documentation__
  - Every `.c` / `.h` file must begin with: license header, `@file` tag, and a one-line `@brief` summary.
  - Add a short paragraph below `@brief` only if the file's role isn't obvious from its name and location.

__Function Declaration Documentation__
  - Pick all naming so that the function declaration are self-explanatory to an experienced developer
  - Add a short `/** ... */` details block only when needed.

__Broader documentation expectations__
  - New modules or significant features require a user-facing doc page or update to an existing one.
  - When touching a file, improve documentation quality across the entire file, not just the lines you changed.
  - Documentation-only PRs are welcome and encouraged.


### Structural things (important)

- No non-trivial or non-inline definitions in headers files.
- Do not use non-ascii characters (except for very specific reasons, eg. pipe or block symbols in progress bars...)
- Keep changes minimal and scoped.
- Variable names are absolutely crucial. We'll establish key conventions, but for instance use `d_` for anything that points to GPU memory
- Minimize variables scope as much as practical.
- Factorize code as much as possible. Any coding by copy-paste should put you on high alert.
  Prefer inline declaration docs in headers over separate `\param` lists, unless a
  specific case benefits from block-style parameter docs.
- Keep documentation of functions purpose in headers. In c/cpp files only point to relevant implementation details.
- Always add a comment to indicate the end of a scope that's more than a few lines long. For example:
  - `#ifdef HAVE_CUDA ....<many lines>.... #endif // #ifdef HAVE_CUDA`
  - `if (condition_met) { ....<many lines>.... } // if (<something>)`
- Prefer early exit to long running brace-statements. The preferred workflow is the one that minimizes indentation.
-
- Allocate structs on the stack except if there's a good reason not to.

> [!NOTE]
> These rules mainly focus on C/C++. Python-specific style rules will be added as the prevalence of python utilities and bindings increases. For now, Python files follow `yapf` formatting via pre-commit.

### Nits

- Prefer 2-letter variables for basic iterators, e.g. `int ii` instead of `int i`, it's much easier to ctrl+F.

## Architecture and templating

- For _module devs_, follow the template provided in `milk-module-example` and fill in the blanks ! More [somewhere].
- For _core devs_... there's probably a lot to say. We must abide by conventions and ensure the overall coherence of the codebase. Always remember: less is more, in particular when what we're counting is technical debt.


## About dependencies

### Internal

Minimize crossovers, avoid making a spaghetti dependency graph.
There should be a refactoring and careful study of internal/external header files for each subpackage.

### External dependencies

In an additional module, the licensing plan we've chosen allow you to combine your work, your license, and the milk core engine with pretty much any other dependencies' license, as long if it's for your own use. The milk core engine licensing strategy would also allow you to do things with very little restrictions as long as you keep it to dynamic linking. This may not be the case of your 3rd party licences.

Since 3rd party licences may be concerning in their compatibility with LPGL, or on the contrary be GPL and compatible with nothing else, we request that any inclusing of an additional dependency, optional or not, goes through a conversation, review, and approval with the maintainers.

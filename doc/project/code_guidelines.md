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

## Recommendations on development environment


## Code organization and editing discipline

Unfortunately, there are many things beyond what can be automated, so here's a list of rules.

Keep something in mind for style: we want to do everything to allow a human to skim the codebase and understand its _intent_ and _purpose_ as quickly as possible.


### Structural things (important)

- No non-trivial or non-inline definitions in headers files.
- Separate enumerative header files (e.g. long `#define` lists of constants) for declarative headers (the actual library's signature)
- Do not use non-ascii characters (except for very specific reasons, eg. pipe or block symbols in progress bars...)
- Keep changes minimal and scoped.
- Variable names are absolutely crucial. We'll establish key conventions, but for instance use `d_` for anything that points to GPU memory
- Minimize variables scope as much as practical.
- Factorize code as much as possible. Any coding by copy-paste should put you on high alert.
  Prefer inline declaration docs in headers over separate `\param` lists, unless a
  specific case benefits from block-style parameter docs.
- Keep documentation of functions purpose in headers. In c/cpp files only point to relevant implementation details.
- Always and a comment to indicate the end of a scope that's more than a few lines long.
  - `#endif // #ifdef HAVE_CUDA`
  - `} // if (<something>)`
  - etc
- Allocate structs on the stack except if there's a good reason not to.

#FIXME: this will have to evolve a fair few once we embed python into this repository as well !

### Nits

- No/minimal non-ascii characters.
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


# Code quality

1) Naming and Structure
  - Use project member naming convention (e.g. `m_` prefix).
  - Keep declaration ordering/grouping stable:
  - public/protected/private
  - slots/signals grouped consistently.
  - Leave a blank line between declarations for readability.

2)  Class Declarations vs Definitions
  - Do not include non-trivial function definitions inside class declarations.
  - Put function definitions below the class declaration (or in `.cpp`), preserving behavior.

3)  Scoped Block Comments
  - For any `{}` block used only to control lock/mutex lifetime, annotate the opening brace as:
    - `{ //mutex scope`



# In-code documentation

1) File-Level Documentation

2) Function Declaration Documentation
  - Every function declaration must have a brief `///` summary.
  - Add a short `/** ... */` details block only when needed.
  - Document every parameter inline at declaration site using:
    - `type name /**< [in] description */`
  - Apply to normal methods, constructors, slots, and signals.
  - Keep return-value docs where project uses them.

3)  Header Declaration Parameter Docs
  - In headers, prefer inline parameter documentation on declarations (`type name /**< ... */`) rather than separate `\param` lists, unless there is a specific reason to deviate.

4)  Changed File Documentation Pass
  - When a file is touched, update documentation quality across the full changed file, not only in modified lines.

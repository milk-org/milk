Follow these code style and documentation rules exactly.
# Coding conventions.

## Automated

- Apply the registered hooks.
  - Hooks shall be designed to be deterministic. This allows _anyone_ to change their style locally and revert upon commit to nominal styling. This also allows global modification at no merge cost.


## Code organization and editing discipline

- No non-trivial or non-inline definitions in headers files.
- Separate enumerative header files (e.g. long #define lists of constants) for declarative headers (the actual library's signature)
- Do not use non-ascii characters (except for very specific reasons, eg. pipe or block symbols in progress bars...)



- Keep changes minimal and scoped.

Better than all the below: install pre-commit (`apt install pre-commit`, or `pip install pre-commit`),run pre-commit.
It automatically filters the relevant actions on the relevant files.
#TODO find a way to run

- Run `./workflows/c_lint` on touched C/CPP files. I won't detail the code style, just do what the linter tells you ! For now its clang-format, in case that ever changes...
- Run `./workflow/python_lint`
- Run `./workflow/markdown_lint`

## Architecture and templating

- For _module devs_, follow the template provided in `milk-module-example` and fill in the blanks ! More [somewhere].
-


## Dependencies

Internal


## External dependencies

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

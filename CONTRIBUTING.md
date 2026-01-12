<!--
TODO, add: LICENSE

This file was inspired by https://github.com/pagefaultgames/pokerogue/blob/beta/CONTRIBUTING.md
-->

# Contributing to KGATE

Thank you for taking the time to contribute, every little bit helps. This project is entirely open-source and unmonetized - community contributions are what keep it alive!

Please make sure you understand everything relevant to your changes from the [Table of Contents](#-table-of-contents), and absolutely *feel free to reach out to the devs [Benjamin Loire](<benjamin.loire@univ-amu.fr>) and [Célia Brahimi](<celia.brahimi@utoulouse.fr>)*.
We are here to help and the better you understand what you're working on, the easier it will be for it to find its way into the project.

Note that, as per GitHub's [terms of service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license), any contributions made to this repository will be licensed under this repository's terms.
If you use any external code, please make sure to follow its licensing information. Please make use of [SPDX snippets](https://reuse.software/spec-3.3/#in-line-snippet-comments) for the portion of the file that is licensed differently.


## 📄 Table of Contents

- [Development Basics](#️-development-basics)
- [Environment Setup](#-environment-setup)
- [Getting Started](#-getting-started)
- [Documentation](#-documentation)
- [Testing Your Changes](#-testing-your-changes)
- [Submitting a Pull Request](#-submitting-a-pull-request)


## 🛠️ Development Basics

KGATE is built with [Python](https://www.python.org/doc/) for [knowledge graphs](<https://en.wikipedia.org/wiki/Knowledge_graph>).

If you have the motivation and experience with Python and knowledge graphs (or are willing to learn), you can contribute by forking the `dev` repository and making pull requests with contributions.


## 💻 Environment Setup

### Codespaces/Devcontainer Environment

<!--
TODO, add: prepared development environment


Arguably the easiest way to get started is by using the prepared development environment.

We have a `.devcontainer/devcontainer.json` file, meaning we are compatible with:

- [![Open in GitHub Codespaces][codespaces-badge]][codespaces-link], or
- the [Visual Studio Code Remote - Containers][devcontainer-ext] extension.

This Linux environment comes with all required dependencies needed to start working on the project.

[codespaces-badge]: <https://github.com/codespaces/badge.svg>
[codespaces-link]: <https://github.com/codespaces/new?hide_repo_select=true&repo=620476224&ref=beta>
[devcontainer-ext]: <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>

> [!IMPORTANT]
> Due to quirks of devcontainer port forwarding, you must use **`pnpm start:podman`** to start a local dev server from within a devcontainer.
> All other instructions remain the same as local development.
-->

### Podman

<!--
TODO, test: is this correct?
-->

For those who prefer Docker containers, see [this instructions page](./docs/podman.md) for information on how to setup a development environment with Podman.

### Local Development

#### Prerequisites

<!--
TODO, rework: add missing prerequisites, from dependencies
-->

- Python: >=3.10.0 - [install 3.10.0](https://www.python.org/downloads/release/python-3100/) | [install latest version](<https://www.python.org/downloads/>)
- PLACEHOLDER
- The repository [forked](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [cloned](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) locally on your device

#### Running Locally

<!--
TODO, add: how to run locally

[Could look like this:]
1. Run `pnpm install` from the repository root
    - *if you run into any errors, reach out to us via mail*
2. Run `pnpm start:dev` to locally run the project on your console`
-->


## 🚀 Getting Started

<!--
TODO, add: when we have a test folder and suite

A great way to develop an understanding of how the project works is to look at test cases (located in [the `test` folder](./tests/)).
Tests show you both how things are supposed to work.
-->

*This is a big project and you will be confused at times - never be afraid to reach out and ask questions!*

### Where to Look

Once you have your feet under you, check out the [Issues](https://github.com/BAUDOTlab/KGATE/issues) page to see how you can help us!
Most issues are bugs and are labeled with their area, such as `Decoder`, `Encoder`, `Documentation`, etc. There are also priority labels:
- `P0`: Completely breaking (very rare)
- `P1`: Major - Crash
- `P2`: Minor - Incorrect (but non-crashing) implementation
- `P3`: No impact for the user - typo, minor graphical error, etc.

You are free to comment on any issue so that you may be assigned to it and we can avoid multiple people working on the same thing.


## 📚 Documentation

<!--
TODO, add: when we have a complete docs folder

You can find the auto-generated documentation [here](https://BAUDOTlab.github.io/KGATE/main/index.html).

Additionally, the [docs folder](./docs) contains a variety of in-depth documents and guides useful for aspiring contributors. \
Notable topics include:
- [Commenting your code](./docs/comments.md)
- [Linting & Formatting](./docs/linting.md)
- [Running with Podman](./docs/podman.md)
-->

Again, if you have unanswered questions please feel free to ask!


## 🧪 Testing Your Changes

You've just made a change - how can you check if it works? You have two areas to hit:

### 1 - Manual Testing

> This will likely be your first stop. After making a change, you'll want to use the tool and make sure everything is as you expect. To do this, you will need a way to manipulate the tool to produce the situation you're looking to test.

<!--
TODO, add: when we have a folder/file for overrides

[src/overrides.py](./src/overrides.py) contains overrides for most values you'll need to change for testing, controlled through the `overrides` object.
For example, here is how you could test a scenario where the player Pokemon has the ability Drought and the enemy Pokemon has the move Water Gun:

```typescript
const overrides = {
  ABILITY_OVERRIDE: AbilityId.DROUGHT,
  ENEMY_MOVESET_OVERRIDE: MoveId.WATER_GUN,
} satisfies Partial<InstanceType<typeof DefaultOverrides>>;
```

Read through `src/overrides.py` file to find the override that fits your needs - there are a lot of them!
If the situation you're trying to test can't be created using existing overrides (or with the [Dev Save](#-development-save-file)), reach out in **#dev-corner**.
You can get help testing your specific changes, and you might have found a new override that needs to be created!
-->

### 2 - Automatic Testing

 <!--
 TODO, complete: everything here if needed
 -->

> KGATE uses *[TODO: add what we use]* for automatic testing. Checking out the existing tests in the [tests](./tests/) folder is a great way to understand how this works, and to get familiar with the project as a whole.

To make sure your changes didn't break any existing test cases, run `pytest tests/` in your terminal. You can also provide an argument to the command: to run only the [PLACEHOLDER something] tests, you could write `pytest tests/ something`. <!-- TODO, test: is it true? -->
  - __Note that passing all test cases does *not* guarantee that everything is working properly__. The project does not have complete regression testing. <!-- TODO, test: is it true? -->

Most non-trivial changes (*especially bug fixes*) should come along with new test cases.
  - To make a new test file, run  <!-- TODO, add: pytest command needed here to create --> and follow the prompts. If the encoder/decoder/etc. you're modifying already has tests, simply add new cases to the end of the file. As mentioned before, the easiest way to get familiar with the system and understand how to write your own tests is simply to read the existing tests, particularly ones similar to the tests you intend to write.
  - Ensure that new tests:
    - Are deterministic. In other words, the test should never pass or fail when it shouldn't due to randomness. This involves primarily ensuring that values are never randomly selected.
    - As much as possible, are unit tests. If you have made two distinct changes, they should be tested in two separate cases.
    - Test edge cases. A good strategy is to think of edge cases beforehand and create tests for them using `it.todo`. Once the edge case has been handled, you can remove the `todo` marker.


## ✅ Submitting a Pull Request

Most information related to submitting a pull request is contained in comments within the pull request template that is shown when you open a new pull request,
however full documentation on the pull request title format is here to best utilize the space available.

The pull request title must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format with a valid prefix and optionally a valid scope. \
If a save migrator, version increase or other breaking change is part of the PR, a `!` must be added before the `:`.

Try to keep the PR title to 72 characters or less (GitHub cuts off commit titles longer than this).

### Examples

```
feat(decoder): SpherE complete implementation
^    ^         ^
|    |         |__ Subject
|    |____________ Scope (optional)
|_________________ Prefix
```

`refactor(random): improve generation of seeds for checkpoints`

`test: improve regression testing`

### List of valid prefixes

- "chore" - Misc project upkeep (e.g. updating submodules, updating dependencies, reverting a bad commit) not covered by other prefixes
- "dev" - Improving the developer experience (such as by modifying lint rules or creating cli scripts)
- "docs" - Primarily adding/updating documentation
- "feat" - Adding a new feature (e.g. adding a new implementation of a decoder) or redesigning an existing feature
- "fix" - Fixing a bug
- "github" - Updating the CI pipeline or otherwise modifying something in the `./github/**` directory
- "misc" - A change that doesn't fit any other prefix
- "perf" - A refactor aimed at improving performance
- "refactor" - A change that doesn't impact functionality or fix any bugs (except incidentally)
- "test" - Primarily adding/updating tests or modifying the test framework
- "user" - Creating/improving the user experience (e.g. adding or updating an interface)

### List of valid scopes

- "autoencoder"
- "decoder"
- "encoder"
- "ml" - Relating to machine learning
- "math" - Relating to formulas
- "random" - Relating to randomness (e.g. seeds)

> [!IMPORTANT]
> All scopes are valid when using the "docs", "feat", "fix", "refactor" and "test" prefixes. \
> No other prefixes have valid scopes.
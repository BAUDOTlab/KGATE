<!--
This file was inspired by https://github.com/pagefaultgames/pokerogue/blob/beta/CONTRIBUTING.md
-->

# Contributing to KGATE

Thank you for taking the time to contribute, every little bit helps. This project is open source and open to contributions of all kind, from code to documentation. If you're unsure what to do, don't hesitate to [post an issue](https://github.com/BAUDOTlab/KGATE/issues).

Please make sure you understand everything relevant to your changes from the [Table of Contents](#-table-of-contents), and absolutely **feel free to reach out to the devs [Benjamin Loire](<benjamin.loire@univ-amu.fr>) and [Célia Brahimi](<celia.brahimi@utoulouse.fr>)**.
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

KGATE is built with [Python](https://www.python.org/doc/) for [knowledge graphs](<https://en.wikipedia.org/wiki/Knowledge_graph>). It uses [PyTorch](https://pytorch.org/) as the underlying machine learning framework, and is heavily inspired from [TorchKGE](https://github.com/torchkge-team/torchkge).

If you have the motivation and experience with Python and knowledge graphs (or are willing to learn), you can contribute by forking the repository and making pull requests with your changes by following the instructions in the [Getting Started](#-getting-started) section.


## 💻 Environment Setup

### Local Development

#### Prerequisites

- Python: >=3.10.0 - [install 3.10.0](https://www.python.org/downloads/release/python-3100/) | [install latest version](<https://www.python.org/downloads/>)
- [Poetry](https://python-poetry.org/)
- The repository [forked](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [cloned](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) locally on your device
- It is recommended to run the code on a GPU to perform tests, though it is not a hard constraint.

#### Running Locally

1. Create a Poetry environment from the repository root: `poetry install`
2. `poetry run python script.py` to run the file `script.py` within the poetry virtual environment.


## 🚀 Getting Started

All new development must be made from the `dev` branch. This will allow your modifications to be integrated after review by maintainers in the development branch and made available for other developers.

The `main` branch is reserved for the stable version of KGATE, and will merge the changes from `dev` from time to time when the development has become stable enough. Only emergency fixes can be pushed directly to `main`.

All changes must be thoroughly documented in the appropriate docstrings, and in the relevant documentation pages if applicable, see the [Documentation](#-documentation) section.
<!--
TODO, add: when we have a test folder and suite

A great way to develop an understanding of how the project works is to look at test cases (located in [the `test` folder](./tests/)).
Tests show you how things are supposed to work.
-->


### Where to Look

Once you have your environment set up, check out the [Issues](https://github.com/BAUDOTlab/KGATE/issues) page to see how you can help us!
Most issues are bugs and are labeled with their area, such as `Decoder`, `Encoder`, `Documentation`, etc. There are also priority labels:
- `Bug: Critical`: Completely breaking (very rare)
- `Bug: Major`: Crash
- `Bug: Minor`: Incorrect (but non-crashing) implementation
- `Bug: Smol`: No impact for the user, i.e. typo, minor graphical error, etc.

In addition, some issues are labelled `Good First Issue`. They are minor issues laying around for new contributors to make an easy contribution to KGATE and get familiar with the library.

Finally, you can filter by complexity: `High`, `Medium` or `Low` which should reflect the amount of effort and time the issue is expected to take.

You are free to comment on any issue so that you may be assigned to it and we can avoid multiple people working on the same thing.


## 📚 Documentation

To help with the documentation, work from the `doc` branch.
<!--
TODO, add: when we have a complete docs folder

You can find the auto-generated documentation [here](https://BAUDOTlab.github.io/KGATE/main/index.html).

Additionally, the [docs folder](./docs) contains a variety of in-depth documents and guides useful for aspiring contributors. \
Notable topics include:
-->

Again, if you have unanswered questions please feel free to ask!


## 🧪 Testing Your Changes

You've just made a change - how can you check if it works? You have two areas to hit:

### 1 - Manual Testing

This will likely be your first stop. After making a change, you'll want to use the tool and make sure everything is as you expect. To do this, you will need a way to manipulate the tool to produce the situation you're looking to test.

### 2 - Automatic Testing

KGATE uses pytest for automatic testing. Checking out the existing tests in the [tests](./tests/) folder is a great way to understand how this works, and to get familiar with the project as a whole.

To make sure your changes didn't break any existing test cases, run `pytest` in your terminal. You can also provide an argument to the command: to run only the decoder tests, you could write `pytest decoders`.
  - __Note that passing all test cases does *not* guarantee that everything is working properly__. The project does not have complete regression testing, or any real test suite at the moment. Contributions in that regard would be greatly appreciated. 

Most non-trivial changes (*especially bug fixes*) should come along with new test cases.
  - To make a new test file, create a new file in the [tests](./tests/) folder or extend an existing one. As mentioned before, the easiest way to get familiar with the system and understand how to write your own tests is simply to read the existing tests, particularly ones similar to the tests you intend to write.
  - Ensure that new tests:
    - Are deterministic. In other words, the test should never pass or fail when it shouldn't due to randomness. This involves primarily ensuring that values are never randomly selected, either providing fixed input and/or setting deterministic seeds..
    - As much as possible, are unit tests. If you have made two distinct changes, they should be tested in two separate cases.

## ✅ Submitting a Pull Request

Most information related to submitting a pull request is contained in comments within the pull request template that is shown when you open a new pull request, however full documentation on the pull request title format is here to best utilize the space available.

The pull request title must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format with a valid prefix and optionally a valid scope.

Try to keep the PR title to 72 characters or less (GitHub cuts off commit titles longer than this).

### Examples

```
[Feat] (decoder) SpherE complete implementation
^      ^         ^
|      |         |__ Subject
|      |____________ Scope (optional)
|___________________ Prefix
```

`[Refactor] (random) Improve generation of seeds for checkpoints`

`[Test] Improve regression testing`

### List of valid prefixes

- "Chore" - Misc project upkeep (e.g. updating submodules, updating dependencies, reverting a bad commit) not covered by other prefixes
- "Dev" - Improving the developer experience (such as by modifying lint rules or creating cli scripts)
- "Docs" - Primarily adding/updating documentation
- "Feat" - Adding a new feature (e.g. adding a new implementation of a decoder) or redesigning an existing feature
- "Fix" - Fixing a bug
- "Github" - Updating the CI pipeline or otherwise modifying something in the `./github/**` directory
- "Misc" - A change that doesn't fit any other prefix
- "Perf" - A refactor aimed at improving performance
- "Refactor" - A change that doesn't impact functionality or fix any bugs (except incidentally)
- "Test" - Primarily adding/updating tests or modifying the test framework
- "User" - Creating/improving the user experience (e.g. adding or updating an interface)

### List of valid scopes

- "architect
- "decoder"
- "encoder"
- "preprocessor"
- "evaluation"
- "training"

> [!IMPORTANT]
> All scopes are valid when using the "docs", "feat", "fix", "refactor" and "test" prefixes. \
> No other prefixes have valid scopes.
<!-- Once you have read these comments, you are free to remove them -->

<!-- Feel free to look at other PRs for examples -->

<!--
Make sure your title matches the https://www.conventionalcommits.org/en/v1.0.0/ format.
Ideally the title is less than or equal to 72 characters (GitHub cuts off commit titles longer than this length).

PLACEHOLDER: See https://github.com/BAUDOTlab/KGATE/blob/dev/CONTRIBUTING.md#-submitting-a-pull-request
for more information on the allowed scopes and prefixes.

Example:
implements(decoder): SpherE complete implementation
^           ^        ^
|           |        |__ Subject
|           |___________ Scope (optional)
|_______________________ Prefix
-->

<!--
Make sure that this PR is not overlapping with someone else's work.
Please try to keep the PR self-contained (don't change a bunch of unrelated things).
-->

<!--
The first section is mandatory if there are user-facing changes (it will be used as the base for a changelog entry).
The second and third section are mandatory.
-->

<!--
This PR template was inspired by https://github.com/pagefaultgames/pokerogue
-->

## What are the changes for using KGATE?
<!-- Summarize what the changes are from a user perspective on the application -->

## Why am I making these changes?
<!--
Explain why you decided to introduce these changes.
Does it come from an issue or another PR? Link to them if possible.
Explain why you believe this can enhance user experience.
-->
<!--
If there are existing GitHub issues related to the PR that would be fixed,
you can add "Fixes #[issue number]" (e.g.: "Fixes #1234") to link an issue to your PR
so that it will automatically be closed when the PR is merged.
-->

## What are the changes from a developer perspective?
<!--
Describe the codebase changes introduced by the PR.
You can make use of a comparison between the state of the code before and after your changes.
Ex: What files have been changed? What classes/functions/variables/etc have been added or changed?
-->

## How to test the changes?
<!--
How can a reviewer test your changes once they check out on your branch?
PLACEHOLDER: Did you make use of the `src/overrides.ts` file?
Did you introduce any automated tests?
Do the reviewers need to do something special in order to test your changes?
-->

## Checklist
- [ ] **I'm using `dev` as my base branch**
- [ ] There is no overlap with another PR
- [ ] I have provided a clear explanation of the changes
- [ ] The PR title matches the format described in [CONTRIBUTING.md](https://github.com//BAUDOTlab/KGATE/blob/dev/CONTRIBUTING.md#-submitting-a-pull-request)
- [ ] I maintained consistency with the project's text
  - [ ] Variables have consistent names
  - [ ] All comments are in American English
- [ ] I have tested the changes manually

<!--
PLACEHOLDER: for when the project has tests
- [ ] The full test suite still passes (`pnpm test:silent`)
  - [ ] I have created new automated tests (`pnpm test:create`) or updated existing tests related to the PR's changes
-->
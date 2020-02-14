# Contributing to Larq Compute Engine

👍 🎉 First off, thanks for taking the time to contribute! 👍 🎉

**Working on your first Pull Request?** You can learn how from this _free_ series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

## Ask a question or raise an issue

If you have questions about Larq Compute Engine or just want to say Hi you can [chat with us on Spectrum](https://spectrum.chat/larq).

If something is not working as expected, if you run into problems with Larq Compute Engine or if you have ideas for missing features, please open a [new issue](https://github.com/larq/compute-engine/issues).

## Project setup

See our [build guide](./docs/build.md) to get started.

## Code style

We use [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html), [`black`](https://black.readthedocs.io/en/stable/) and [`buildifier`](https://github.com/bazelbuild/buildtools/releases/tag/1.0.0) to format all of our code.

## Publish LCE converter `pip` package

1. Increment the version number in `setup.py`, and make a PR with that change.

2. Wait until your PR is reviewed and merged.

3. Create and push a new tag from the latest `master` branch (e.g. `v0.7.0`).

   ```shell
   git checkout master
   git pull
   git tag <version number>
   git push --tags
   ```

4. A [GitHub action](https://github.com/larq/compute-engine/actions) will automatically build wheels publish a release to [PyPI](https://pypi.org/) based on the tag.

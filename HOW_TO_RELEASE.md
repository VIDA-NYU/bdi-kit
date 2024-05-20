# How to release a new version

* On `devel` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Change the version in `bdikit/__init__.py` to the new version, e.g., `2.0.0` (using the format MAJOR.MINOR.PATCH).
  * In `CHANGELOG.md`, change the first version, e.g. `2.0.0.dev0 (yyyy-mm-dd)` to the to-be-released version and date.
  * Commit with message `Bump version for release`.
  * `git push`

* On `main` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Merge `devel` into `main` branch: `git merge devel`
  * `git push`
  * Release a package to PyPI:
    * `rm -rf dist/`
    * `python setup.py sdist bdist_wheel`
    * `twine upload dist/*`
  * Create a tag for the new version, `git tag <tag_name>` and push it `git push origin <tag_name>`
  
* On `devel` branch:
  * `git merge main` to make sure `devel` is always on top of `main`.
  * Change the version in `bdikit/__init__.py` appending `.dev0` to the future version, e.g. `2.1.0.dev0`.
  * Add a new empty version on top of `CHANGELOG.md`, e.g. `2.1.0.dev0 (yyyy-mm-dd)`.
  * Commit with message `Bump version for development`.
  * `git push`
 
# How to release a new version

1. Switch to branch `devel` and pull to make sure everything is in sync with remote origin.
```bash
    git checkout devel
    git pull
```

2. Change the version in `bdikit/__init__.py` to the new version, e.g., `2.0.0` (using the format MAJOR.MINOR.PATCH).

3. In `CHANGELOG.md`, change the first version, e.g. `2.0.0.dev0 (yyyy-mm-dd)` to the to-be-released version and date and listall changes included in the release.

4. Commit with title "Bump version for release {version}" and push to remote. This commit will include the file changes done in steps 2 and 3 above.
```bash
git commit -m "Bump version for release 0.x.y"
git push origin devel
```

5. Switch to master `main`, pull to make sure everything is in sync with remote origin, and then merge `devel` into the `main`;
```bash
git checkout main
git pull
git merge devel
```

6. Push the local `main` to the remote repo. This will trigger the CI tests, build the package and upload it to `https://test.pypi.org/project/bdi-kit/`.
```bash
git push origin main
```

7. Verify that CI and the publication to TestPyPI completed successfuly and test the package. E.g., using Python 3.10:
```bash
# Create a new venv and activate it
mkdir /tmp/bdikit-test/ && cd /tmp/bdikit-test/
python3.10 -m venv ./venv-test
source venv-test/bin/activate
# Install bdi-kit
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple bdi-kit
# You can now install jupyter to test the library
pip install jupyter
jupyter notebook
```

8. Switch back to the repository, and create a git tag with version name, e.g., for version 2.0.0 run:
```bash
cd ${bdikit-source-path}
# Create tag
git tag 2.0.0
# List all tags to verify the tag was created
git tag -l
```

9. Push the tag to the remote repository. This will trigger the CI tests, and
   build and publish the package to `https://pypi.org/project/bdi-kit/`.
```bash
git push --tags
```

10. Head to GitHub Actions page (https://github.com/VIDA-NYU/bdi-kit/actions), locate the workflow run ("Publish to PyPI 🐍") triggered for the new release tag and approve its execution. When the workflow finishes, the library should be available at https://pypi.org/project/bdi-kit/.

11. Head to https://github.com/VIDA-NYU/bdi-kit/releases/ and update the release
    with the CHANGELOG for the released version.

12. Switch to `devel` branch and merge the release (to make sure `devel` is always on top of `main`). If you didn't have to make any changes to `main` this will do nothing.
```bash
git checkout devel
git merge main
```
13. Change the version in `bdikit/__init__.py` appending `.dev0` to the future version, e.g. `2.1.0.dev0`. Add a new empty version on top of `CHANGELOG.md`, e.g. `2.1.0.dev0 (yyyy-mm-dd)`.

14.  Commit with message `Bump version for development`.
```bash
git add CHANGELOG.md bdikit/__init__.py
git commit -m "Bump version for development"
git push
```
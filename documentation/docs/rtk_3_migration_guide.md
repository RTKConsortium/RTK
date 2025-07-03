# RTK v3 migration guide

## Use `--size` instead of `--dimension`

As of RTK 3.0, the `--dimension` argument in applications is deprecated and
will be removed in a future release.  Please use the `--size` argument instead.
This update aligns RTK with upstream ITK's member name, where `--size` more
accurately reflects the expected parameter (e.g., image size) and avoids
confusion with spatial dimensions (e.g., 2D vs 3D).

## Forbild phantom file format only

The legacy `rtk::GeometricPhantomFileReader` has been removed starting with RTK
3.0. This handcrafted file format was not documented and everything it could do
is also achievable with the only file format for geometric phantoms accepted in
RTK 3.0, the [Forbild phantom file format](Phantom.md). The flag `--forbild`
has also been removed from command line tools as it is now the default and only
option.

## Remove config option from c++ applications

The `--config` command-line option has been removed from all RTK ggo-based
applications as it was not known to be used in practice.

## Default branch renamed from master to main

The InsightSoftwareConsortium has decided to rename its default branches from `master` to `main` (https://github.com/InsightSoftwareConsortium/ITK/issues/4732) following [GitHub's move](https://github.com/github/renaming). RTK's default branch has been renamed from `master` to `main` too. Here is what to do:

- GitHub forks: see [Matt McCormick's post](https://discourse.itk.org/t/itk-git-repository-primary-branch-name-transition-from-master-to-main/7569) to change your fork's default repository.
- Your local clones should be updated as suggested by GitHub (if `origin` is the name of https://github.com/RTKConsortium/RTK.git):

```
git branch -m master main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```

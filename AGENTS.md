# RTK AI Agent Guide

The Reconstruction ToolKit (RTK) is a cross-platform C++ toolkit for computed
tomography reconstruction, built as an ITK module. It provides command-line
applications, Python wrapping, and optional CUDA acceleration. Apache 2.0 licensed.
Build tools: **CMake**, **CTest**, and **pytest**.

RTK follows the [ITK AI Agent Guide](https://github.com/InsightSoftwareConsortium/ITK/blob/main/AGENTS.md)
for C++ conventions, commit messages, committed prose, and general pitfalls,
substituting the `rtk` namespace and file prefix for `itk`. Its build and test
instructions (Pixi) do not apply; use the RTK-specific ones below.

## Context to Load on Demand

| Task | Read |
|------|------|
| Getting started or constructing a reconstruction pipeline | [GettingStarted.md](./GettingStarted.md), [FirstReconstruction](./examples/FirstReconstruction/README.md) |
| Configuring or building RTK, including Python wrapping | [INSTALLATION.md](./INSTALLATION.md), [CMakeLists.txt](./CMakeLists.txt), [itk-module-init.cmake](./itk-module-init.cmake) |
| Changing module dependencies or compiled sources | [itk-module.cmake](./itk-module.cmake), [src/CMakeLists.txt](./src/CMakeLists.txt) |
| Writing C++ filters or CUDA kernels | Related classes in [include/](./include/) and [src/](./src/), [CodeContribution.md](./CodeContribution.md) |
| Changing geometry, coordinate transforms, or projectors | [Geometry.md](./documentation/docs/Geometry.md), [Projectors.md](./documentation/docs/Projectors.md) |
| Adding or running tests | [test/CMakeLists.txt](./test/CMakeLists.txt), related tests in [test/](./test/), [CodeContribution.md](./CodeContribution.md) |
| Changing Python APIs or wrapped template types | [wrapping/](./wrapping/), [wrapping/CMakeLists.txt](./wrapping/CMakeLists.txt), [pyproject.toml](./pyproject.toml) |
| Changing command-line applications | [applications/README.md](./applications/README.md), the application's C++, Python, and `.ggo` files |
| Formatting code | [.pre-commit-config.yaml](./.pre-commit-config.yaml), [.clang-format](./.clang-format), [.gersemi.config](./.gersemi.config) |
| Editing documentation or examples | [documentation/docs/README.md](./documentation/docs/README.md), [examples/README.md](./examples/README.md) |
| Understanding CI coverage | [.github/workflows/](./.github/workflows/) |

## Working Conventions

- On top of ITK coding conventions, CUDA declarations are in `.hcu` files and
  CUDA implementations in `.cu` files.
- Avoid unrelated changes to bundled third-party code under `utilities/`.
- Commit subjects start with an ITK prefix (`BUG:`, `COMP:`, `DOC:`, `ENH:`,
  `PERF:`, `STYLE:`) and fit in 78 characters. Agents run `pre-commit install`
  once and set `ITK_AI_AGENT=1` in the environment of `git commit`.

## C++ and Python Consistency

- **Always check Python impact when adding or modifying functionality.** Include
  the corresponding Python changes in the same change. If no Python update is
  needed, explain why in the change summary.
- For a new public C++ class, add its `.wrap` file and any required registration
  in `wrapping/CMakeLists.txt`. Inspect existing `.wrap` files for the supported
  image types and dimensions.
- Keep C++ and Python applications consistent in options, defaults, validation,
  and behavior: update `.ggo` definitions, Python argument handling, and package
  entry points together. New options use hyphen-separated lowercase names.
- Update corresponding C++ and Python examples together so they demonstrate the
  same pipeline and parameters. Provide both the C++ and Python versions of new
  examples when the required APIs are available in both languages.

## Building and Testing

Reuse an existing build tree and inspect its `CMakeCache.txt` before changing
configuration. For a new standalone build, first build ITK with `Module_RTK=OFF`;
`ITK_DIR` must point to an ITK **build tree**, not an install tree. CUDA requires
`RTK_USE_CUDA=ON`, CudaCommon, and a compatible GPU to run tests; keep CPU-only
builds working when changing shared code.

```bash
cmake --build <build-dir> -j && ctest --test-dir <build-dir> -R <regex> --output-on-failure
```

- Install Python with
  [utilities/rtk_install_python_packages.sh](./utilities/rtk_install_python_packages.sh)
  (see [INSTALLATION.md](./INSTALLATION.md)). Re-run it after C++ or `.wrap`
  changes; handwritten Python modules are symlinked to the source tree.
- Run affected Python tests against wrapping built from the modified sources, e.g.
  `python -m pytest test/rtk_argument_parser_test.py`. Test dependencies are in the
  `test` dependency group of `pyproject.toml`.
- Register C++ tests with `rtk_add_test` or `rtk_add_cuda_test`; exercise CPU and
  CUDA implementations when both are affected.
- Reference test data with `DATA{...}` and content links under `test/Input` or
  `test/Baseline`. New content links use SHA512; keep binary data out of Git.
- Run `pre-commit run --files <changed files>`. Report the build configuration and
  checks performed, including any CUDA or wrapping coverage that was unavailable.

## Critical Pitfalls

1. **Angle units:** geometry files and `AddProjection` use degrees; geometry
   stores angles internally in radians. Respect image origin, spacing, and
   direction when changing transforms.
2. **CPU and CUDA agreement:** preserve numerical agreement within justified
   tolerances. Check host/device buffer synchronization and avoid unintended
   transfers in iterative pipelines.
3. **Generated files:** edit `.ggo`, `.in`, and `.wrap` sources, not their
   generated outputs.

## Resources

- Documentation: https://docs.openrtk.org/
- Issues: https://github.com/RTKConsortium/RTK/issues
- Dashboard: https://my.cdash.org/index.php?project=RTK

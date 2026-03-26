# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the core SLAM implementation; matching public headers live in `include/`, with camera models under `include/CameraModels/` and `src/CameraModels/`. `Examples/` holds current standalone demos and YAML configs for Monocular, Stereo, RGB-D, inertial, and calibration workflows. `Examples_old/` preserves legacy demos and the ROS package in `Examples_old/ROS/ORB_SLAM3`. Vendored dependencies live in `Thirdparty/` (`DBoW2`, `g2o`, `Sophus`); avoid changing them unless you are updating a dependency. Generated outputs land in `build/`, `lib/`, and runtime files such as `CameraTrajectory.txt`.

## Build, Test, and Development Commands
Use the provided build script for a full local build:

```bash
./build.sh
```

This builds `Thirdparty/*`, unpacks `Vocabulary/ORBvoc.txt`, and compiles `lib/libORB_SLAM3.so` plus example binaries.

For iterative work, prefer CMake directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

ROS support is legacy in this checkout. The package is under `Examples_old/ROS/ORB_SLAM3`; verify `build_ros.sh` before using it.

## Coding Style & Naming Conventions
Follow the surrounding C++ style: 4-space indentation, opening braces on their own line, and `Type* name` pointer declarations. Keep class and method names in PascalCase (`LocalMapping`, `SaveTrajectoryTUM`), member fields with `m` prefixes (`mpTracker`, `mbReset`), and booleans with `b`/`mb` prefixes. Most core files use `.cc`; add new files with the extension used by the neighboring module. There is no repo-wide formatter configured, so do not reformat unrelated code.

## Testing Guidelines
There is no top-level automated test suite or coverage gate. Validate changes by rebuilding and running the closest example binary for the affected sensor path, for example:

```bash
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml
```

If you touch trajectory, timing, or calibration logic, record the dataset used and attach the resulting trajectory or timing artifact in your PR. `Thirdparty/Sophus/test` is upstream vendor coverage, not the main acceptance path here.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Update README.md` and `Update ROS scripts`. Keep commits focused and descriptive, and separate dependency/vendor changes from core SLAM changes. PRs should include the problem statement, key files changed, exact build/run commands used for validation, dataset or hardware details, and screenshots or trajectory diffs when viewer output or tracking behavior changes.

# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the core SLAM pipeline implementation; matching public headers live in `include/` (plus `include/CameraModels/`). Executable entry points and dataset configs are under `Examples/`, grouped by sensor mode (`Monocular`, `Stereo`, `RGB-D`, `*-Inertial`). Legacy demos are in `Examples_old/`. Third-party code is vendored in `Thirdparty/` (`DBoW2`, `g2o`, `Sophus`) and should only be changed when updating dependencies. Evaluation utilities and ground-truth files are in `evaluation/`. Runtime vocabulary assets are in `Vocabulary/` (`ORBvoc.txt.tar.gz`).

## Build, Test, and Development Commands
- `./build.sh`: builds third-party libs, extracts the vocabulary, and compiles `lib/libORB_SLAM3.so` plus demo binaries.
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`: manual/incremental build flow.
- `./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml <euroc_seq_dir> Examples/Stereo/EuRoC_TimeStamps/MH01.txt`: representative local run.
- `python2 evaluation/evaluate_ate_scale.py <groundtruth.txt> <estimated_traj.txt> --verbose`: compute trajectory RMSE/scale metrics.
- `./build_ros.sh`: optional ROS build; requires the ROS package path used by the script to exist in this checkout.

## Coding Style & Naming Conventions
Use C++11 and match existing style: 4-space indentation, braces on the next line for functions, and minimal formatting-only diffs. Keep class/type names in PascalCase (`KeyFrameDatabase`, `MapPoint`) and preserve file pairing (`src/Foo.cc` with `include/Foo.h`; camera models use `.cpp/.h`). Keep code in the `ORB_SLAM3` namespace. No top-level formatter/linter is enforced; if you use one locally, avoid reformatting unrelated code. Do not edit vendored `Thirdparty/` sources unless required.

## Testing Guidelines
There is no root unit-test target in `CMakeLists.txt`; validation is scenario-based. For algorithm changes, run at least one affected example and verify generated `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`. For quantitative checks, compare ATE outputs from `evaluation/evaluate_ate_scale.py` before and after your change.

## Commit & Pull Request Guidelines
Recent history favors short imperative commit subjects (for example, `Fix typos`, `Update README.md`). Prefer `area: imperative summary` for code changes (example: `tracking: reduce map update contention`). In pull requests, include:
- what changed and why
- exact run command(s) and dataset/config used
- before/after metrics or behavioral notes
- linked issue IDs when applicable

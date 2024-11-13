# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.5] - 2024-11-13

### Fixed
* outdated dependency in docker image build (#6)
* numpy explicit casting and deprecations (#7)

## [1.0.4] - 2023-07-18

### Fixed
* Replace deprecated pillow font.getsize() method (#5)

## [1.0.3] - 2023-06-20
### Fixed
* Compatibility with file format version 2.6 (#4)
* several typos in CHANGELOG

### Changed
* Removed any data inversion on loading.

## [1.0.2] - 2023-03-16
### Fixed
* `numpy` deprecation of `np.float` (#2)

### Changed
* Creep correction: Both the `sin` and `root` creep corrections were re-derived with more strict constraints to prevent un-physical/nonsensical creep behavior.
* Added function constraints to documentation. The derivation of the creep functions is not part of this repo as of now.

### Added
* Functions `sin_limit_function` and `root_limit_function` to automatically adapt the bounds for the optimizer when approximating the creep behavior in `sin` or `root` mode. This prevents `pyfastspm` from crashing / skipping the creep correction if bounds are defined too loosely.

## [1.0.1] - 2022-07-20
###  Changed
* citation file: DOI, license ids, and bump version info
* use concept DOI in README badge

## [1.0.0] - 2022-07-12
### Added
- first public release

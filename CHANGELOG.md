# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
* Creep correction: Both the `sin` and `root` creep correction where rederived with sticter constraints to prevent unphysical / nonsensical creep behaviour. 
* Added function constraints to documentation. The derivtation of the creep functions is not part of this repo as of now.

### Added
* Functions `sin_limit_function` and `root_limit_function` to automatically adapt the bounds for the optimizer when appriximating the creep bahaviour in `sin` or `root` mode. This prevents pyfast from crashing / skipping the creep correction if bounds are defined too losely. This is considered a bug fix.

## [1.0.1] - 2022-07-20
###  Changed
* citation file: DOI, license ids, and bump version info
* use concept DOI in README badge

## [1.0.0] - 2022-07-12
### Added
- first public release

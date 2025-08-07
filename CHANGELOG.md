# Changelog

All notable changes to the zerograd project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Refactored Tensor methods to use `&[usize]` instead of `&Vec<usize>` to reduce unnecessary overhead (#19)

### Added
- Added README documentation to tensor crate (#17)

### Fixed
- Updated main README with current project information (#18)

## Previous Changes

### Refactoring and Improvements
- Refactored project to modularize Tensor code into separate files (#16)
- Refactored unary operations to use generic functions (#15)
- Refactored in-place binary operations to use generic helper functions
- Refactored binary operations to use generic functions with closures
- Added Tensor error types for better error handling

### Features
- Added ReLU (Rectified Linear Unit) unary operation for neural network support

### Project Structure
- Modularized tensor functionality into separate files for better organization
- Established workspace structure with tensor crate as primary component
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-01-18

### Changed
- Use user-friendly language in reports (Response/Format/Tool Consistency)
- Replace technical variance labels with status (Excellent/Good/Needs Attention)
- Cleaner output format in terminal reports

## [0.2.1] - 2025-01-18

### Changed
- Clarified README positioning: consistency vs correctness
- Added `__version__` to package for programmatic access
- Added link to examples folder in README

## [0.2.0] - 2025-01-17

### Added
- **Baseline comparison** - Save baselines with `--save-baseline` and detect drift with `--compare`
- **Model comparison** - Compare stability across models with `--compare-model`
- **Agentic testing mode** - Full tool execution loop with `--tool-mocks`
- **User-friendly error messages** - Detailed help for API key, JSON, and configuration errors
- **Edge case test scenarios** - Examples for testing ambiguous tool routing

### Changed
- Updated Anthropic model names to current API IDs (claude-sonnet-4.5, claude-haiku-4.5, etc.)
- Simplified exit codes - always exit 0, clear result messages instead
- Improved CLI output with clear result summary at end

### Fixed
- Tool call ID handling for multi-tool OpenAI responses
- Variable naming conflict with save_baseline function

## [0.1.0] - 2025-01-05

### Changed
- **License changed from MIT to Apache 2.0** for better patent protection and enterprise adoption
- Updated repository URLs in pyproject.toml to ruvrics-ai organization

### Added
- Initial release of Ruvrics AI Stability Engine
- Core stability metrics: semantic, tool, structural, and length consistency
- Support for OpenAI models (GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Support for Anthropic models (Claude Opus 4, Sonnet 4, Sonnet 3.5, Haiku 4)
- Root cause identification with 7 distinct instability patterns
- Actionable recommendation engine
- CLI with Rich terminal output
- Three input formats: simple, messages, and tool-enabled
- Modular tool definitions via --tools flag
- Comprehensive test suite (83 passing tests)
- Example files and documentation

### Features
- Weighted stability scoring (semantic 40%, tool 25%, structural 20%, length 15%)
- Risk classification (SAFE, RISKY, DO_NOT_SHIP)
- Pattern-based risky claim detection
- Retry logic with exponential backoff
- Progress tracking with visual indicators
- JSON export of results
- Configurable thresholds

### Documentation
- Complete README with usage examples
- Technical specification in docs/
- Implementation guides
- Example input files

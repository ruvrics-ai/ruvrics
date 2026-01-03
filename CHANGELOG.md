# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-04

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

# Configuration Management Guide for AI Stability CLI

## Problem

The current adapter design has issues:
- API keys hard-coded or assumed from environment
- Model names scattered across code
- Configuration not centralized
- No validation of required settings

## Solution: Configuration Module

Create a dedicated `config.py` file that handles all configuration.

---

## File Structure Update

Add these files to your project:

```
airel/
├── config.py              # NEW: Configuration management
├── .env.example           # NEW: Example environment file
├── core/
│   ├── adapters.py        # UPDATED: Uses config module
```

---

## Implementation

### 1. airel/config.py

```python
"""
Configuration management for AI Stability CLI.

Handles:
- API keys from environment
- Model configurations
- Default parameters
- Validation of settings
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for a specific model"""
    name: str
    provider: str  # "openai" or "anthropic"
    max_tokens: int = 4096
    temperature: float = 0.0  # Default to deterministic
    supports_tools: bool = True


class Config(BaseModel):
    """Main configuration class"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Default settings
    default_runs: int = 20
    min_successful_runs: int = 15
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Thresholds (from spec Section 4)
    semantic_low_threshold: float = 85.0
    semantic_medium_threshold: float = 70.0
    
    tool_low_threshold: float = 95.0
    tool_medium_threshold: float = 80.0
    
    structural_low_threshold: float = 95.0
    structural_medium_threshold: float = 85.0
    
    length_low_threshold: float = 80.0
    length_medium_threshold: float = 60.0
    
    # Stability score weights (from spec Section 3)
    semantic_weight: float = 0.40
    tool_weight: float = 0.25
    structural_weight: float = 0.20
    length_weight: float = 0.15
    
    # Risk classification thresholds
    safe_threshold: float = 90.0
    risky_threshold: float = 70.0
    
    # Results storage
    results_dir: Path = Field(default_factory=lambda: Path.home() / ".airel")
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for specified provider"""
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key not found. "
                    "Set OPENAI_API_KEY environment variable."
                )
            return self.openai_api_key
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Supported models registry
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    # OpenAI models
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo-preview",
        provider="openai",
        supports_tools=True
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        provider="openai",
        supports_tools=True
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        supports_tools=True
    ),
    
    # Anthropic models
    "claude-opus-4": ModelConfig(
        name="claude-opus-4-20250514",
        provider="anthropic",
        supports_tools=True
    ),
    "claude-sonnet-4": ModelConfig(
        name="claude-sonnet-4-20250514",
        provider="anthropic",
        supports_tools=True
    ),
}


def get_model_config(model_identifier: str) -> ModelConfig:
    """Get configuration for a model"""
    if model_identifier in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_identifier]
    
    raise ValueError(f"Model '{model_identifier}' not supported")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance (singleton)"""
    global _config
    if _config is None:
        _config = Config()
    return _config
```

### 2. .env.example

```bash
# AI Stability CLI Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Updated adapters.py

```python
from airel.config import get_config, get_model_config

class OpenAIAdapter:
    def __init__(self, model_identifier: str):
        self.config = get_config()
        self.model_config = get_model_config(model_identifier)
        
        # Get API key from config
        api_key = self.config.get_api_key("openai")
        self.client = openai.OpenAI(api_key=api_key)
```

---

## Benefits

✅ Centralized configuration  
✅ Environment variable support  
✅ Type safety with Pydantic  
✅ Easy testing  
✅ Model registry  

This is the professional way to handle configuration!

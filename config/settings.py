#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Settings:

    DEFAULT_CONFIG = {
        "app": {
            "name": "StackMind",
            "version": "0.1.0",
            "description": "Multi-step reasoning and problem-solving AI agent"
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "enable_docs": True
        },
        "llm": {
            "provider": "mistral",
            "model": "mistral-large-latest",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "memory": {
            "type": "vector",
            "max_history": 100,
            "persistence": True,
            "storage_path": "./data/memory"
        },
        "input_processing": {
            "enable_text": True,
            "enable_image": True,
            "enable_voice": True,
            "enable_code": True,
            "max_image_size": 10 * 1024 * 1024,
            "supported_image_formats": ["jpg", "jpeg", "png", "gif", "bmp"],
            "supported_code_languages": ["python", "javascript", "java", "c", "cpp", "csharp", "go", "ruby"]
        },
        "output_processing": {
            "default_format": "markdown",
            "enable_visualization": True,
            "visualization_types": ["tree", "graph", "flowchart", "table"]
        },
        "domains": {
            "enabled": ["general", "math", "code", "science"],
            "auto_detect": True
        },
        "integrations": {
            "enable_web_search": True,
            "enable_code_execution": True,
            "enable_data_sources": True,
            "safe_mode": True,
            "allowed_domains": ["*"],
            "max_web_requests": 10
        },
        "privacy": {
            "store_conversations": True,
            "anonymize_personal_data": True,
            "data_retention_days": 30
        },
        "performance": {
            "optimization_priority": "balanced",
            "cache_results": True,
            "parallel_processing": True,
            "max_parallel_tasks": 4
        }
    }

    def __init__(self, profile: str = "default"):
        self.profile = profile
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config = self.DEFAULT_CONFIG.copy()
        config_dir = Path(__file__).parent
        config_file = config_dir / f"{self.profile}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self._deep_merge(config, file_config)
            except Exception as e:
                print(f"Error loading config file {config_file}: {e}")

        self._apply_env_overrides(config)
        return config

    def _deep_merge(self, target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _apply_env_overrides(self, config: Dict) -> None:
        prefix = "PROBLEM_SOLVER_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                key_parts = env_key[len(prefix):].lower().split('_')
                if len(key_parts) >= 2:
                    section = key_parts[0]
                    key = '_'.join(key_parts[1:])
                    if section in config and key in config[section]:
                        original_value = config[section][key]
                        if isinstance(original_value, bool):
                            config[section][key] = env_value.lower() in ('true', 'yes', '1')
                        elif isinstance(original_value, int):
                            config[section][key] = int(env_value)
                        elif isinstance(original_value, float):
                            config[section][key] = float(env_value)
                        elif isinstance(original_value, list):
                            config[section][key] = env_value.split(',')
                        else:
                            config[section][key] = env_value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self.config.get(section, {})

    def save(self) -> None:
        config_dir = Path(__file__).parent
        config_file = config_dir / f"{self.profile}.json"
        config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")

    def __str__(self) -> str:
        return json.dumps(self.config, indent=2)

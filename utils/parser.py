import os
import json
import yaml

class ConfigNamespace:
    def __init__(self, config):
        self._populate(config)

    def _populate(self, config):
        for key, value in config.items():
            nested_value = (
                ConfigNamespace(value) if isinstance(value, dict) else value
            )
            setattr(self, key, nested_value)

def load_config_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    with open(file_path, 'r') as f:
        if ext in ('.yaml', '.yml'):
            data = yaml.safe_load(f)
        elif ext == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    return ConfigNamespace(data)

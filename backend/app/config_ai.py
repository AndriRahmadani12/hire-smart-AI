from dataclasses import dataclass
@dataclass
class OpenAIConfig:
    api_key: str
    api_version: str
    azure_endpoint: str
    deployment_name: str
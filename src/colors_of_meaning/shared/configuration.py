import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from colors_of_meaning.resources import get_resource_path
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

DEFAULT_EXPERIMENT_CONFIG_PATH = "configs/base.yaml"
DEFAULT_CODEBOOK_BASE_PATH = "artifacts/codebooks"
DEFAULT_CODEBOOK_NAME = "codebook_4096"
DEFAULT_MODEL_PATH = "artifacts/models/projector.pth"
DEFAULT_CORPUS_PATH = "artifacts/encoded/test_documents.pkl"


def load_properties_file(file_path: str) -> Dict[str, str]:
    properties = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                properties[key.strip()] = value.strip()

    return properties


class ApplicationSettings(BaseSettings):
    reload: bool = False
    host: str = ""
    experiment_config: str = DEFAULT_EXPERIMENT_CONFIG_PATH

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_properties_file_settings()

    def _load_properties_file_settings(self) -> None:
        try:
            properties = self._get_properties()
            self._apply_properties(properties)
        except FileNotFoundError:
            pass

    def _get_properties(self) -> Dict[str, str]:
        properties_path = get_resource_path("application.properties")
        return load_properties_file(properties_path)

    def _apply_properties(self, properties: Dict[str, str]) -> None:
        for key, value in properties.items():
            self._apply_property(key, value)

    def _apply_property(self, key: str, value: str) -> None:
        if hasattr(self, key) and not os.environ.get(f"APP_{key.upper()}"):
            setattr(self, key, value)


class ApplicationSettingProvider:
    def __init__(self) -> None:
        self.settings = ApplicationSettings()
        self.override_settings: Dict[str, Any] = {}

    def override(self, key: str, value: Any) -> None:
        self.override_settings[key] = value

    def get(self, key: str) -> Any:
        return self._get_from_overrides(key) or self._get_from_settings(key)

    def _get_from_overrides(self, key: str) -> Any:
        return self.override_settings.get(key)

    def _get_from_settings(self, key: str) -> Any:
        if hasattr(self.settings, key):
            value = getattr(self.settings, key)

            if key == "host" and not value:
                raise ValueError("Host setting not found in properties file or environment")
            return value

        raise ValueError(f"Setting {key} not found")


def get_application_setting_provider() -> ApplicationSettingProvider:
    return ApplicationSettingProvider()


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: str = DEFAULT_MODEL_PATH
    codebook_base_path: str = DEFAULT_CODEBOOK_BASE_PATH
    codebook_name: str = DEFAULT_CODEBOOK_NAME
    corpus_path: str = DEFAULT_CORPUS_PATH


@dataclass(frozen=True)
class ExperimentRuntimeContext:
    experiment_config_path: str
    synesthetic_config: SynestheticConfig
    artifact_paths: ArtifactPaths


class ExperimentConfigurationError(RuntimeError):
    pass


def _load_experiment_config(path: str) -> SynestheticConfig:
    try:
        return SynestheticConfig.from_yaml(path)
    except OSError as error:
        raise ExperimentConfigurationError(f"Experiment config could not be loaded from '{path}'") from error


def build_experiment_runtime_context(
    settings_provider: Optional[ApplicationSettingProvider] = None,
    synesthetic_config: Optional[SynestheticConfig] = None,
    artifact_paths: Optional[ArtifactPaths] = None,
) -> ExperimentRuntimeContext:
    provider = settings_provider or get_application_setting_provider()
    experiment_config_path = provider.get("experiment_config")
    resolved_config = synesthetic_config or _load_experiment_config(experiment_config_path)
    return ExperimentRuntimeContext(
        experiment_config_path=experiment_config_path,
        synesthetic_config=resolved_config,
        artifact_paths=artifact_paths or ArtifactPaths(),
    )

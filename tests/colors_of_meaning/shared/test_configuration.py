import os
from unittest.mock import patch, mock_open

import pytest
from assertpy import assert_that

from colors_of_meaning.shared.configuration import (
    ApplicationSettings,
    ApplicationSettingProvider,
    ArtifactPaths,
    ExperimentConfigurationError,
    ExperimentRuntimeContext,
    build_experiment_runtime_context,
    load_properties_file,
)
from colors_of_meaning.shared.synesthetic_config import (
    SynestheticConfig,
    ProjectorConfig,
    CodebookConfig,
    TrainingConfig,
    DistanceConfig,
    DatasetConfig,
)


def _in_memory_config() -> SynestheticConfig:
    return SynestheticConfig(
        projector=ProjectorConfig(),
        codebook=CodebookConfig(),
        training=TrainingConfig(),
        distance=DistanceConfig(),
        dataset=DatasetConfig(),
    )


class TestLoadPropertiesFile:
    def test_should_load_properties_from_file(self):
        mock_file_content = "admin=testadmin\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).contains_entry({"admin": "testadmin"})

    def test_should_handle_empty_lines(self):
        mock_file_content = "admin=testadmin\n\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).is_length(2)

    def test_should_handle_comment_lines(self):
        mock_file_content = "admin=testadmin\n#comment line\npassword=testpassword\n"

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = load_properties_file("dummy/path")

        assert_that(result).is_length(2)


class TestApplicationSettings:
    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_load_experiment_config_setting_from_properties_file(
        self, mock_load_properties, mock_get_resource_path
    ):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "experiment_config": "configs/agnews_run.yaml",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.experiment_config).is_equal_to("configs/agnews_run.yaml")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_load_reload_setting_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "experiment_config": "configs/agnews_run.yaml",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_true()

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_load_host_setting_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {
            "experiment_config": "configs/agnews_run.yaml",
            "reload": "true",
            "host": "127.0.0.1",
        }

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.host).is_equal_to("127.0.0.1")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_use_environment_variables_over_properties(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"experiment_config": "configs/from_properties.yaml"}

        with patch.dict(os.environ, {"APP_EXPERIMENT_CONFIG": "configs/from_environment.yaml"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.experiment_config).is_equal_to("configs/from_environment.yaml")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    def test_should_use_reload_default_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_false()

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    def test_should_use_host_from_env_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.host).is_equal_to("127.0.0.1")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_handle_reload_setting_from_environment(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"reload": "false"}

        with patch.dict(os.environ, {"APP_RELOAD": "true"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.reload).is_true()

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    def test_should_default_experiment_config_when_missing_properties_file(self, mock_get_resource_path):
        mock_get_resource_path.side_effect = FileNotFoundError

        with patch.dict(os.environ, {"APP_HOST": "127.0.0.1"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.experiment_config).is_equal_to("configs/base.yaml")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_load_experiment_config_from_properties_file(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"experiment_config": "configs/structured.yaml", "host": "127.0.0.1"}

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.experiment_config).is_equal_to("configs/structured.yaml")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_read_experiment_config_from_environment(self, mock_load_properties, mock_get_resource_path):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"experiment_config": "configs/structured.yaml", "host": "127.0.0.1"}

        with patch.dict(os.environ, {"APP_EXPERIMENT_CONFIG": "configs/supervised.yaml"}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.experiment_config).is_equal_to("configs/supervised.yaml")

    @patch("colors_of_meaning.shared.configuration.get_resource_path")
    @patch("colors_of_meaning.shared.configuration.load_properties_file")
    def test_should_keep_host_setting_when_experiment_config_is_configured(
        self, mock_load_properties, mock_get_resource_path
    ):
        mock_get_resource_path.return_value = "dummy/path"
        mock_load_properties.return_value = {"host": "127.0.0.1", "experiment_config": "configs/structured.yaml"}

        with patch.dict(os.environ, {}, clear=True):
            settings = ApplicationSettings()

        assert_that(settings.host).is_equal_to("127.0.0.1")


class TestApplicationSettingProvider:
    @patch("colors_of_meaning.shared.configuration.ApplicationSettings")
    def test_should_get_host_setting_value(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings

        host_result = provider.get("host")

        assert_that(host_result).is_equal_to("0.0.0.0")

    @patch("colors_of_meaning.shared.configuration.ApplicationSettings")
    def test_should_allow_setting_override(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.experiment_config = "configs/base.yaml"
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings
        provider.override("experiment_config", "configs/overridden.yaml")

        result = provider.get("experiment_config")

        assert_that(result).is_equal_to("configs/overridden.yaml")

    @patch("colors_of_meaning.shared.configuration.ApplicationSettings")
    def test_should_allow_reload_setting_override(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.reload = False
        mock_settings.host = "0.0.0.0"

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings
        provider.override("reload", True)

        result = provider.get("reload")

        assert_that(result).is_true()

    def test_should_raise_error_for_nonexistent_setting(self):
        provider = ApplicationSettingProvider()

        provider.override("host", "0.0.0.0")

        with pytest.raises(ValueError) as excinfo:
            provider.get("nonexistent")

        assert_that(str(excinfo.value)).contains("not found")

    @patch("colors_of_meaning.shared.configuration.ApplicationSettings")
    def test_should_raise_error_for_empty_host_value(self, mock_settings_class):
        mock_settings = mock_settings_class.return_value
        mock_settings.host = ""

        provider = ApplicationSettingProvider()
        provider.settings = mock_settings

        with pytest.raises(ValueError) as excinfo:
            provider.get("host")

        assert_that(str(excinfo.value)).contains("Host setting not found")

    def test_should_resolve_experiment_config_when_overridden(self):
        provider = ApplicationSettingProvider()
        provider.override("experiment_config", "configs/structured.yaml")

        assert_that(provider.get("experiment_config")).is_equal_to("configs/structured.yaml")


class TestArtifactPaths:
    def test_should_default_model_path_when_unspecified(self):
        assert_that(ArtifactPaths().model_path).is_equal_to("artifacts/models/projector.pth")

    def test_should_default_codebook_base_path_when_unspecified(self):
        assert_that(ArtifactPaths().codebook_base_path).is_equal_to("artifacts/codebooks")

    def test_should_default_codebook_name_when_unspecified(self):
        assert_that(ArtifactPaths().codebook_name).is_equal_to("codebook_4096")

    def test_should_default_corpus_path_when_unspecified(self):
        assert_that(ArtifactPaths().corpus_path).is_equal_to("artifacts/encoded/test_documents.pkl")


class TestBuildExperimentRuntimeContext:
    def test_should_load_synesthetic_config_when_experiment_path_is_configured(self, tmp_path):
        config_path = tmp_path / "experiment.yaml"
        config_path.write_text("projector:\n  embedding_dim: 256\n")
        provider = ApplicationSettingProvider()
        provider.override("experiment_config", str(config_path))

        context = build_experiment_runtime_context(settings_provider=provider)

        assert_that(context.synesthetic_config.projector.embedding_dim).is_equal_to(256)

    def test_should_record_experiment_config_path_when_context_is_built(self, tmp_path):
        config_path = tmp_path / "experiment.yaml"
        config_path.write_text("{}")
        provider = ApplicationSettingProvider()
        provider.override("experiment_config", str(config_path))

        context = build_experiment_runtime_context(settings_provider=provider)

        assert_that(context.experiment_config_path).is_equal_to(str(config_path))

    def test_should_use_injected_config_when_override_is_provided(self):
        provider = ApplicationSettingProvider()
        provider.override("experiment_config", "configs/base.yaml")
        injected = _in_memory_config()

        with patch.object(SynestheticConfig, "from_yaml", side_effect=AssertionError("disk read attempted")):
            context = build_experiment_runtime_context(settings_provider=provider, synesthetic_config=injected)

        assert_that(context.synesthetic_config).is_same_as(injected)

    def test_should_build_with_default_provider_when_none_supplied(self):
        injected = _in_memory_config()

        context = build_experiment_runtime_context(synesthetic_config=injected)

        assert_that(context.synesthetic_config).is_same_as(injected)

    def test_should_expose_default_artifact_paths_when_context_is_built(self):
        provider = ApplicationSettingProvider()
        injected = _in_memory_config()

        context = build_experiment_runtime_context(settings_provider=provider, synesthetic_config=injected)

        assert_that(context.artifact_paths).is_equal_to(ArtifactPaths())

    def test_should_use_overridden_artifact_paths_when_provided(self):
        provider = ApplicationSettingProvider()
        injected = _in_memory_config()
        custom_paths = ArtifactPaths(model_path="custom/model.pth")

        context = build_experiment_runtime_context(
            settings_provider=provider, synesthetic_config=injected, artifact_paths=custom_paths
        )

        assert_that(context.artifact_paths).is_same_as(custom_paths)

    def test_should_raise_when_experiment_config_path_is_missing(self, tmp_path):
        missing = tmp_path / "absent.yaml"
        provider = ApplicationSettingProvider()
        provider.override("experiment_config", str(missing))

        with pytest.raises(ExperimentConfigurationError) as error:
            build_experiment_runtime_context(settings_provider=provider)

        assert_that(str(error.value)).contains(str(missing))

    def test_should_return_runtime_context_instance_when_built(self):
        injected = _in_memory_config()

        context = build_experiment_runtime_context(synesthetic_config=injected)

        assert_that(context).is_instance_of(ExperimentRuntimeContext)

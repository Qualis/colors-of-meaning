from pathlib import Path

from colors_of_meaning.shared.synesthetic_config import (
    SynestheticConfig,
    ProjectorConfig,
    CodebookConfig,
    TrainingConfig,
    DistanceConfig,
    DatasetConfig,
    StructuredMapperConfig,
)


def _create_test_yaml_content() -> str:
    return """
projector:
  embedding_dim: 512
  hidden_dim_1: 256
  hidden_dim_2: 128
  output_dim: 3
  dropout_rate: 0.2

codebook:
  bins_per_dimension: 8
  num_bins: 512

training:
  batch_size: 64
  learning_rate: 0.002
  epochs: 50
  seed: 123
  device: cpu

distance:
  metric: jensen_shannon
  smoothing_epsilon: 1e-6

dataset:
  name: custom_dataset
  train_split: train
  test_split: test
  max_samples: 1000

structured_mapper:
  alpha: 2.0
  beta: 0.5
  gamma: 1.5
  num_clusters: 32
  max_chroma: 100.0
"""


def _assert_projector_values(config: SynestheticConfig) -> None:
    assert config.projector.embedding_dim == 512
    assert config.projector.hidden_dim_1 == 256


def _assert_other_config_values(config: SynestheticConfig) -> None:
    assert config.codebook.bins_per_dimension == 8
    assert config.training.batch_size == 64
    assert config.distance.metric == "jensen_shannon"
    assert config.dataset.name == "custom_dataset"


class TestSynestheticConfig:
    def test_should_create_config_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_create_test_yaml_content())

        config = SynestheticConfig.from_yaml(str(config_path))

        _assert_projector_values(config)
        _assert_other_config_values(config)

    def test_should_save_config_to_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(embedding_dim=256),
            codebook=CodebookConfig(bins_per_dimension=8),
            training=TrainingConfig(epochs=50),
            distance=DistanceConfig(metric="wasserstein"),
            dataset=DatasetConfig(name="test_dataset"),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        assert config_path.exists()
        loaded_config = SynestheticConfig.from_yaml(str(config_path))
        assert loaded_config.projector.embedding_dim == 256
        assert loaded_config.codebook.bins_per_dimension == 8

    def test_should_use_defaults_for_missing_values(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.projector.embedding_dim == 384
        assert config.codebook.bins_per_dimension == 16
        assert config.training.batch_size == 32

    def test_should_create_parent_directories_when_saving(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )
        config_path = tmp_path / "nested" / "dir" / "config.yaml"

        config.to_yaml(str(config_path))

        assert config_path.exists()

    def test_should_load_structured_mapper_config_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_create_test_yaml_content())

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.structured_mapper.alpha == 2.0
        assert config.structured_mapper.beta == 0.5

    def test_should_load_structured_mapper_num_clusters(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(_create_test_yaml_content())

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.structured_mapper.num_clusters == 32
        assert config.structured_mapper.max_chroma == 100.0

    def test_should_default_structured_mapper_when_missing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.structured_mapper.alpha == 1.0
        assert config.structured_mapper.num_clusters == 16

    def test_should_save_structured_mapper_to_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            structured_mapper=StructuredMapperConfig(alpha=3.0),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        loaded = SynestheticConfig.from_yaml(str(config_path))
        assert loaded.structured_mapper.alpha == 3.0

    def test_should_default_structured_mapper_in_post_init(self) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )

        assert isinstance(config.structured_mapper, StructuredMapperConfig)


def _assert_projector_defaults(config: ProjectorConfig) -> None:
    assert config.embedding_dim == 384
    assert config.hidden_dim_1 == 128
    assert config.hidden_dim_2 == 64


def _assert_projector_other_defaults(config: ProjectorConfig) -> None:
    assert config.output_dim == 3
    assert config.dropout_rate == 0.1


def _assert_training_defaults(config: TrainingConfig) -> None:
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.epochs == 100


def _assert_training_other_defaults(config: TrainingConfig) -> None:
    assert config.seed == 42
    assert config.device == "cuda"


def _assert_dataset_defaults(config: DatasetConfig) -> None:
    assert config.name == "ag_news"
    assert config.train_split == "train"


def _assert_dataset_other_defaults(config: DatasetConfig) -> None:
    assert config.test_split == "test"
    assert config.max_samples is None


class TestProjectorConfig:
    def test_should_have_default_values(self) -> None:
        config = ProjectorConfig()
        _assert_projector_defaults(config)
        _assert_projector_other_defaults(config)


class TestCodebookConfig:
    def test_should_have_default_values(self) -> None:
        config = CodebookConfig()

        assert config.bins_per_dimension == 16
        assert config.num_bins == 4096


class TestTrainingConfig:
    def test_should_have_default_values(self) -> None:
        config = TrainingConfig()
        _assert_training_defaults(config)
        _assert_training_other_defaults(config)


class TestDistanceConfig:
    def test_should_have_default_values(self) -> None:
        config = DistanceConfig()

        assert config.metric == "wasserstein"
        assert config.smoothing_epsilon == 1e-8


class TestDatasetConfig:
    def test_should_have_default_values(self) -> None:
        config = DatasetConfig()
        _assert_dataset_defaults(config)
        _assert_dataset_other_defaults(config)


class TestStructuredMapperConfig:
    def test_should_have_default_values(self) -> None:
        config = StructuredMapperConfig()

        assert config.alpha == 1.0
        assert config.beta == 1.0

    def test_should_have_default_gamma_and_clusters(self) -> None:
        config = StructuredMapperConfig()

        assert config.gamma == 1.0
        assert config.num_clusters == 16

    def test_should_have_default_max_chroma(self) -> None:
        config = StructuredMapperConfig()

        assert config.max_chroma == 128.0

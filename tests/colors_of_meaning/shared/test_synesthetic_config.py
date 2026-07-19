from pathlib import Path

from colors_of_meaning.shared.synesthetic_config import (
    SynestheticConfig,
    ProjectorConfig,
    CodebookConfig,
    TrainingConfig,
    DistanceConfig,
    DatasetConfig,
    StructuredMapperConfig,
    SupervisedMapperConfig,
    CompassConfig,
    GenerationConfig,
    GroundingConfig,
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

    def test_should_read_sinkhorn_reg_when_present_in_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("distance:\n  metric: wasserstein\n  sinkhorn_reg: 0.05\n")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.distance.sinkhorn_reg == 0.05

    def test_should_round_trip_sinkhorn_reg_through_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(sinkhorn_reg=0.01),
            dataset=DatasetConfig(),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        assert SynestheticConfig.from_yaml(str(config_path)).distance.sinkhorn_reg == 0.01

    def test_should_round_trip_sentiment_source_when_serialising_structured_config(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            structured_mapper=StructuredMapperConfig(sentiment_source="labels"),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        assert SynestheticConfig.from_yaml(str(config_path)).structured_mapper.sentiment_source == "labels"

    def test_should_round_trip_concreteness_resource_when_serialising_structured_config(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            structured_mapper=StructuredMapperConfig(concreteness_resource="custom_norms.tsv"),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        assert (
            SynestheticConfig.from_yaml(str(config_path)).structured_mapper.concreteness_resource == "custom_norms.tsv"
        )

    def test_should_default_structured_config_to_neutral_sentiment_when_unspecified(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.structured_mapper.sentiment_source == "none"


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
    assert config.paragraphs_per_work == 60


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

    def test_should_default_sinkhorn_reg_to_none_when_not_provided(self) -> None:
        config = DistanceConfig()

        assert config.sinkhorn_reg is None


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

    def test_should_default_sentiment_source_to_none(self) -> None:
        config = StructuredMapperConfig()

        assert config.sentiment_source == "none"

    def test_should_default_concreteness_resource(self) -> None:
        config = StructuredMapperConfig()

        assert config.concreteness_resource == "concreteness_norms.tsv"


class TestSupervisedMapperConfig:
    def test_should_have_default_classification_weight(self) -> None:
        config = SupervisedMapperConfig()

        assert config.classification_weight == 0.1

    def test_should_have_default_num_classes(self) -> None:
        config = SupervisedMapperConfig()

        assert config.num_classes == 4

    def test_should_accept_custom_values(self) -> None:
        config = SupervisedMapperConfig(classification_weight=0.5, num_classes=10)

        assert config.classification_weight == 0.5
        assert config.num_classes == 10


class TestSupervisedMapperInSynestheticConfig:
    def test_should_default_supervised_mapper_in_post_init(self) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )

        assert isinstance(config.supervised_mapper, SupervisedMapperConfig)

    def test_should_load_supervised_mapper_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = _create_test_yaml_content() + """
supervised_mapper:
  classification_weight: 0.5
  num_classes: 10
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.supervised_mapper.classification_weight == 0.5
        assert config.supervised_mapper.num_classes == 10

    def test_should_default_supervised_mapper_when_missing_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.supervised_mapper.classification_weight == 0.1
        assert config.supervised_mapper.num_classes == 4

    def test_should_save_supervised_mapper_to_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            supervised_mapper=SupervisedMapperConfig(classification_weight=0.3, num_classes=8),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        loaded = SynestheticConfig.from_yaml(str(config_path))
        assert loaded.supervised_mapper.classification_weight == 0.3
        assert loaded.supervised_mapper.num_classes == 8


class TestCompassConfig:
    def test_should_default_drift_threshold(self) -> None:
        config = CompassConfig()

        assert config.drift_threshold == 25.0

    def test_should_default_min_beat_chars(self) -> None:
        config = CompassConfig()

        assert config.min_beat_chars == 200

    def test_should_default_mapper_to_structured(self) -> None:
        config = CompassConfig()

        assert config.mapper == "structured"

    def test_should_default_metric_to_sliced(self) -> None:
        config = CompassConfig()

        assert config.metric == "sliced"

    def test_should_default_report_and_figure_paths(self) -> None:
        config = CompassConfig()

        assert config.report_path == "reports/story_compass.md"
        assert config.figure_path == "reports/figures/story_compass.png"


class TestCompassInSynestheticConfig:
    def test_should_default_compass_in_post_init(self) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )

        assert isinstance(config.compass, CompassConfig)

    def test_should_default_compass_when_missing_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.compass.drift_threshold == 25.0

    def test_should_load_compass_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("compass:\n  drift_threshold: 12.5\n  metric: exact\n")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.compass.drift_threshold == 12.5
        assert config.compass.metric == "exact"

    def test_should_round_trip_compass_through_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            compass=CompassConfig(drift_threshold=9.0, mapper="plain"),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        loaded = SynestheticConfig.from_yaml(str(config_path))
        assert loaded.compass.drift_threshold == 9.0
        assert loaded.compass.mapper == "plain"


class TestGenerationConfig:
    def test_should_default_model_to_opus(self) -> None:
        config = GenerationConfig()

        assert config.model == "claude-opus-4-8"

    def test_should_default_effort_to_high(self) -> None:
        config = GenerationConfig()

        assert config.effort == "high"

    def test_should_default_retry_budget(self) -> None:
        config = GenerationConfig()

        assert config.retry_budget == 2

    def test_should_default_drift_threshold(self) -> None:
        config = GenerationConfig()

        assert config.drift_threshold == 25.0

    def test_should_default_prose_max_tokens(self) -> None:
        config = GenerationConfig()

        assert config.prose_max_tokens == 16000


class TestGenerationInSynestheticConfig:
    def test_should_default_generation_in_post_init(self) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )

        assert isinstance(config.generation, GenerationConfig)

    def test_should_default_generation_when_missing_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.generation.model == "claude-opus-4-8"

    def test_should_load_generation_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("generation:\n  num_beats: 12\n  effort: max\n")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.generation.num_beats == 12
        assert config.generation.effort == "max"

    def test_should_round_trip_generation_through_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            generation=GenerationConfig(retry_budget=5, words_per_beat=750),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        loaded = SynestheticConfig.from_yaml(str(config_path))
        assert loaded.generation.retry_budget == 5
        assert loaded.generation.words_per_beat == 750


class TestGroundingConfig:
    def test_should_default_threshold(self) -> None:
        config = GroundingConfig()

        assert config.threshold == 25.0

    def test_should_default_metric_to_sliced(self) -> None:
        config = GroundingConfig()

        assert config.metric == "sliced"

    def test_should_default_sample_and_report_paths(self) -> None:
        config = GroundingConfig()

        assert config.sample_path == "samples/rag_grounding.yaml"
        assert config.report_path == "reports/grounding_audit.md"


class TestGroundingInSynestheticConfig:
    def test_should_default_grounding_in_post_init(self) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
        )

        assert isinstance(config.grounding, GroundingConfig)

    def test_should_default_grounding_when_missing_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("{}")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.grounding.threshold == 25.0

    def test_should_load_grounding_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("grounding:\n  threshold: 8.5\n  metric: exact\n")

        config = SynestheticConfig.from_yaml(str(config_path))

        assert config.grounding.threshold == 8.5
        assert config.grounding.metric == "exact"

    def test_should_round_trip_grounding_through_yaml(self, tmp_path: Path) -> None:
        config = SynestheticConfig(
            projector=ProjectorConfig(),
            codebook=CodebookConfig(),
            training=TrainingConfig(),
            distance=DistanceConfig(),
            dataset=DatasetConfig(),
            grounding=GroundingConfig(threshold=6.0, metric="jensen_shannon"),
        )
        config_path = tmp_path / "output.yaml"

        config.to_yaml(str(config_path))

        loaded = SynestheticConfig.from_yaml(str(config_path))
        assert loaded.grounding.threshold == 6.0
        assert loaded.grounding.metric == "jensen_shannon"


CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


class TestConfigFileBackCompat:
    def test_should_parse_base_config_unchanged_when_loaded_from_yaml(self) -> None:
        config = SynestheticConfig.from_yaml(str(CONFIGS_DIR / "base.yaml"))

        assert config.projector.embedding_dim == 384
        assert config.dataset.name == "ag_news"
        assert config.dataset.max_samples == 12000

    def test_should_parse_structured_config_unchanged_when_loaded_from_yaml(self) -> None:
        config = SynestheticConfig.from_yaml(str(CONFIGS_DIR / "structured.yaml"))

        assert config.structured_mapper.num_clusters == 16
        assert config.structured_mapper.max_chroma == 128.0

    def test_should_parse_supervised_config_unchanged_when_loaded_from_yaml(self) -> None:
        config = SynestheticConfig.from_yaml(str(CONFIGS_DIR / "supervised.yaml"))

        assert config.supervised_mapper.classification_weight == 0.1
        assert config.supervised_mapper.num_classes == 4


README_PATH = Path(__file__).resolve().parents[3] / "README.MD"


class TestDocsConsistency:
    def test_should_document_the_configured_embedding_dim_when_reading_readme(self) -> None:
        readme_text = README_PATH.read_text()

        assert str(ProjectorConfig().embedding_dim) in readme_text

    def test_should_document_the_exact_compression_ratio_when_reading_readme(self) -> None:
        readme_text = README_PATH.read_text()

        assert "1024:1" in readme_text

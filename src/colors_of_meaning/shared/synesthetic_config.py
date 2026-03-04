from dataclasses import dataclass
from typing import Optional
import yaml  # type: ignore
from pathlib import Path


@dataclass
class ProjectorConfig:
    embedding_dim: int = 384
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    output_dim: int = 3
    dropout_rate: float = 0.1


@dataclass
class CodebookConfig:
    bins_per_dimension: int = 16
    num_bins: int = 4096


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    seed: int = 42
    device: str = "cuda"


@dataclass
class DistanceConfig:
    metric: str = "wasserstein"
    smoothing_epsilon: float = 1e-8


@dataclass
class DatasetConfig:
    name: str = "ag_news"
    train_split: str = "train"
    test_split: str = "test"
    max_samples: Optional[int] = None


@dataclass
class StructuredMapperConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    num_clusters: int = 16
    max_chroma: float = 128.0


@dataclass
class SupervisedMapperConfig:
    classification_weight: float = 0.1
    num_classes: int = 4


@dataclass
class SynestheticConfig:
    projector: ProjectorConfig
    codebook: CodebookConfig
    training: TrainingConfig
    distance: DistanceConfig
    dataset: DatasetConfig
    structured_mapper: Optional[StructuredMapperConfig] = None
    supervised_mapper: Optional["SupervisedMapperConfig"] = None

    def __post_init__(self) -> None:
        if self.structured_mapper is None:
            self.structured_mapper = StructuredMapperConfig()
        if self.supervised_mapper is None:
            self.supervised_mapper = SupervisedMapperConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "SynestheticConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        supervised_raw = config_dict.get("supervised_mapper", {})
        supervised_mapper = SupervisedMapperConfig(**supervised_raw) if supervised_raw else None

        return cls(
            projector=ProjectorConfig(**config_dict.get("projector", {})),
            codebook=CodebookConfig(**config_dict.get("codebook", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            distance=DistanceConfig(**config_dict.get("distance", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            structured_mapper=StructuredMapperConfig(**config_dict.get("structured_mapper", {})),
            supervised_mapper=supervised_mapper,
        )

    def to_yaml(self, path: str) -> None:
        config_dict = {
            "projector": self.projector.__dict__,
            "codebook": self.codebook.__dict__,
            "training": self.training.__dict__,
            "distance": self.distance.__dict__,
            "dataset": self.dataset.__dict__,
            "structured_mapper": self.structured_mapper.__dict__,
            "supervised_mapper": self.supervised_mapper.__dict__,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

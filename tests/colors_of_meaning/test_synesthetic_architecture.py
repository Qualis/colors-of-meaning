import importlib

from pytest_archon.rule import archrule


def test_should_maintain_domain_layer_independence() -> None:
    (
        archrule(
            "Domain Layer Independence",
            comment="Domain layer should not depend on other layers (Clean Architecture)",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "colors_of_meaning.infrastructure.*",
            "colors_of_meaning.interface.*",
            "colors_of_meaning.application.*",
        )
        .check("colors_of_meaning")
    )


def test_should_maintain_application_interface_independence() -> None:
    (
        archrule(
            "Application Interface",
            comment="Application layer should not depend on interface layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.interface.*")
        .check("colors_of_meaning")
    )


def test_should_maintain_application_infrastructure_independence() -> None:
    (
        archrule(
            "Application Infrastructure",
            comment="Application layer should not depend on infrastructure layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.infrastructure.*")
        .check("colors_of_meaning")
    )


def test_should_use_application_use_case_in_cli() -> None:
    (
        archrule(
            "CLI Use Case",
            comment="Interface CLI commands should depend on application use cases",
        )
        .match(
            "colors_of_meaning.interface.cli.train",
            "colors_of_meaning.interface.cli.encode",
            "colors_of_meaning.interface.cli.compare",
            "colors_of_meaning.interface.cli.compress",
            "colors_of_meaning.interface.cli.query",
            "colors_of_meaning.interface.cli.ablate",
            "colors_of_meaning.interface.cli.eval",
            "colors_of_meaning.interface.cli.eval_suite",
            "colors_of_meaning.interface.cli.encode_image",
            "colors_of_meaning.interface.cli.decode_image",
            "colors_of_meaning.interface.cli.encode_lossless",
            "colors_of_meaning.interface.cli.decode_lossless",
            "colors_of_meaning.interface.cli.interpretability",
            "colors_of_meaning.interface.cli.rate_distortion",
            "colors_of_meaning.interface.cli.compass",
        )
        .should_import("colors_of_meaning.application.use_case.*")
        .check("colors_of_meaning")
    )


def test_should_keep_the_narrative_arc_model_free_of_frameworks() -> None:
    (
        archrule(
            "Narrative Arc Model Purity",
            comment="The narrative arc value object must stay free of ML, distance, and I/O frameworks",
        )
        .match("colors_of_meaning.domain.model.narrative_arc")
        .should_not_import(
            "torch",
            "torch.*",
            "matplotlib",
            "matplotlib.*",
            "ot",
            "ot.*",
        )
        .check("colors_of_meaning")
    )


def test_should_keep_the_outline_parser_free_of_upper_layers() -> None:
    (
        archrule(
            "Outline Parser Independence",
            comment="The shared outline parser must not import application, infrastructure, or interface",
        )
        .match("colors_of_meaning.shared.outline_parser")
        .should_not_import(
            "colors_of_meaning.application.*",
            "colors_of_meaning.infrastructure.*",
            "colors_of_meaning.interface.*",
        )
        .check("colors_of_meaning")
    )


def test_should_keep_pillow_out_of_domain_data_modules() -> None:
    (
        archrule(
            "Domain Data Module Imaging Isolation",
            comment="The lossless codec port and payload framing must stay free of the Pillow imaging dependency",
        )
        .match(
            "colors_of_meaning.domain.service.data_payload",
            "colors_of_meaning.domain.service.data_image_codec",
        )
        .should(
            lambda module, direct_imports, all_imports: not any(
                imported == "PIL" or imported.startswith("PIL.") for imported in direct_imports
            ),
            "no_pillow_import",
        )
        .check("colors_of_meaning")
    )


def test_should_keep_the_lossless_codec_model_free() -> None:
    (
        archrule(
            "Lossless Codec Model-Free",
            comment="The lossless codec must not depend on the projector, embeddings, or the semantic Lab codebook",
        )
        .match(
            "colors_of_meaning.infrastructure.visualization.pillow_data_image_codec",
            "colors_of_meaning.domain.service.data_image_codec",
            "colors_of_meaning.domain.service.data_payload",
        )
        .should_not_import(
            "colors_of_meaning.infrastructure.ml.*",
            "colors_of_meaning.infrastructure.embedding.*",
            "colors_of_meaning.infrastructure.persistence.*",
            "colors_of_meaning.domain.model.color_codebook",
        )
        .check("colors_of_meaning")
    )


def test_should_implement_dataset_repository_in_document_corpus_adapter() -> None:
    from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
    from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
        DocumentCorpusDatasetAdapter,
    )

    assert issubclass(DocumentCorpusDatasetAdapter, DatasetRepository)


def test_should_keep_dataset_infrastructure_independent_of_upper_layers() -> None:
    (
        archrule(
            "Dataset Infrastructure Independence",
            comment="Infrastructure dataset adapters and the seeded sampler must not import application or interface",
        )
        .match("colors_of_meaning.infrastructure.dataset.*")
        .should_not_import(
            "colors_of_meaning.application.*",
            "colors_of_meaning.interface.*",
        )
        .check("colors_of_meaning")
    )


def test_should_keep_structure_metric_numerical_dependencies_out_of_domain() -> None:
    (
        archrule(
            "Structure Metric Dependency Isolation",
            comment="The scipy-backed structure-preservation metric must stay out of the domain layer",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "scipy",
            "scipy.*",
            "torch",
            "torch.*",
        )
        .check("colors_of_meaning")
    )


def test_should_not_import_sklearn_in_domain_layer() -> None:
    (
        archrule(
            "Domain k-means Isolation",
            comment="The learned-codebook k-means dependency must stay out of the domain layer",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "sklearn",
            "sklearn.*",
        )
        .check("colors_of_meaning")
    )


def test_should_confine_structure_preservation_evaluator_scipy_to_infrastructure() -> None:
    (
        archrule(
            "Structure Preservation Evaluator Infrastructure",
            comment="The Spearman evaluator implementation lives in infrastructure and may depend on scipy",
        )
        .match("colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator")
        .should_import("scipy.stats")
        .check("colors_of_meaning")
    )


def test_should_confine_cosine_histogram_calculator_scipy_to_infrastructure() -> None:
    (
        archrule(
            "Cosine Histogram Calculator Infrastructure",
            comment="The scipy-backed cosine histogram distance calculator lives in infrastructure and may depend on scipy",
        )
        .match("colors_of_meaning.infrastructure.ml.cosine_histogram_distance_calculator")
        .should_import("scipy.spatial.distance")
        .check("colors_of_meaning")
    )


def test_should_keep_optimal_transport_dependency_out_of_domain() -> None:
    (
        archrule(
            "Domain Optimal Transport Isolation",
            comment="The POT optimal-transport dependency must stay out of the domain layer",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "ot",
            "ot.*",
        )
        .check("colors_of_meaning")
    )


def test_should_confine_sliced_wasserstein_optimal_transport_to_infrastructure() -> None:
    (
        archrule(
            "Sliced Wasserstein Calculator Infrastructure",
            comment="The sliced-Wasserstein proxy lives in infrastructure and may depend on POT",
        )
        .match("colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator")
        .should_import("ot")
        .check("colors_of_meaning")
    )


def test_should_confine_interpretability_evaluator_sklearn_to_infrastructure() -> None:
    (
        archrule(
            "Interpretability Evaluator Infrastructure",
            comment="The sklearn/scipy-backed interpretability evaluator lives in infrastructure and may depend on them",
        )
        .match("colors_of_meaning.infrastructure.evaluation.sklearn_interpretability_evaluator")
        .should_import("sklearn.metrics")
        .check("colors_of_meaning")
    )


def test_should_keep_interpretability_numerics_out_of_domain() -> None:
    (
        archrule(
            "Interpretability Domain Numerics Isolation",
            comment="The interpretability report model and evaluator port must stay free of scipy/sklearn",
        )
        .match(
            "colors_of_meaning.domain.model.interpretability_report",
            "colors_of_meaning.domain.service.interpretability_evaluator",
        )
        .should_not_import(
            "scipy",
            "scipy.*",
            "sklearn",
            "sklearn.*",
        )
        .check("colors_of_meaning")
    )


def test_should_confine_spearman_rank_correlation_scipy_to_infrastructure() -> None:
    (
        archrule(
            "Spearman Rank Correlation Infrastructure",
            comment="The scipy-backed rank-correlation calculator lives in infrastructure and may depend on scipy",
        )
        .match("colors_of_meaning.infrastructure.evaluation.spearman_rank_correlation_calculator")
        .should_import("scipy.stats")
        .check("colors_of_meaning")
    )


def test_should_maintain_shared_module_independence() -> None:
    (
        archrule(
            "Shared Module Dependencies",
            comment="Shared module should not depend on application, infrastructure or interface",
        )
        .match("colors_of_meaning.shared.*")
        .should_not_import(
            "colors_of_meaning.application.*",
            "colors_of_meaning.infrastructure.*",
            "colors_of_meaning.interface.*",
        )
        .check("colors_of_meaning")
    )


def test_should_not_import_configuration_module_in_domain_layer() -> None:
    (
        archrule(
            "Domain Configuration Isolation",
            comment="Domain layer must stay free of configuration-loading concerns",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "colors_of_meaning.shared.configuration",
            "colors_of_meaning.shared.synesthetic_config",
        )
        .check("colors_of_meaning")
    )


def test_should_allow_all_layers_to_import_shared() -> None:
    shared_modules = [
        importlib.import_module("colors_of_meaning.shared.configuration"),
        importlib.import_module("colors_of_meaning.shared.synesthetic_config"),
    ]

    assert all(module is not None for module in shared_modules)


def test_should_not_have_circular_dependencies() -> None:
    (
        archrule(
            "No Circular Dependencies",
            comment="No modules should have circular dependencies",
        )
        .match("colors_of_meaning.*")
        .should(
            lambda module, direct_imports, all_imports: module not in direct_imports
            and module not in all_imports.get(module, set()),
            "no_circular_dependencies",
        )
        .check("colors_of_meaning")
    )

import logging
import pickle  # nosec B403
import sys
import uuid
from typing import List, Optional, cast

import uvicorn
from fastapi import APIRouter, FastAPI
from lagom import Container

from colors_of_meaning.application.use_case.compare_documents_use_case import CompareDocumentsUseCase
from colors_of_meaning.application.use_case.health_use_case import HealthUseCase
from colors_of_meaning.application.use_case.query_by_palette_use_case import QueryByPaletteUseCase
from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.security.basic_authentication import (
    BasicAuthenticator,
    SecurityDependency,
    get_basic_authenticator,
)
from colors_of_meaning.infrastructure.system.health_factory import create_health_checker
from colors_of_meaning.interface.api.controller.health_controller import create_health_controller
from colors_of_meaning.interface.api.controller.query_controller import (
    create_query_controller,
    create_unavailable_query_controller,
)
from colors_of_meaning.shared.configuration import (
    DEFAULT_CODEBOOK_NAME,
    DEFAULT_CORPUS_PATH,
    ArtifactPaths,
    ExperimentConfigurationError,
    ExperimentRuntimeContext,
    build_experiment_runtime_context,
    get_application_setting_provider,
)
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

APPLICATION_TITLE = "Colors of Meaning API"
APPLICATION_VERSION = "1.0.0"
CODEBOOK_NAME = DEFAULT_CODEBOOK_NAME
CORPUS_ARTIFACT_PATH = DEFAULT_CORPUS_PATH
DEFAULT_ARTIFACT_PATHS = ArtifactPaths()
DISTANCE_METRIC = "wasserstein"
FALLBACK_BINS_PER_DIMENSION = 16
SINKHORN_REGULARISATION: Optional[float] = None
SMOOTHING_EPSILON = 1e-8
CORPUS_UNAVAILABLE_DETAIL = "Color retrieval is unavailable: encoded corpus artifact is missing"


def _correlation_id() -> str:
    return str(uuid.uuid4())


def _load_query_codebook(artifact_paths: ArtifactPaths = DEFAULT_ARTIFACT_PATHS) -> ColorCodebook:
    repository = FileColorCodebookRepository(artifact_paths.codebook_base_path)
    codebook = repository.load(artifact_paths.codebook_name)
    if codebook is not None:
        return codebook

    logger.warning(
        "Codebook artifact absent; falling back to uniform grid",
        extra={"correlation_id": _correlation_id(), "expected_codebook": artifact_paths.codebook_name},
    )
    return ColorCodebook.create_uniform_grid(FALLBACK_BINS_PER_DIMENSION)


def _load_corpus(corpus_path: Optional[str] = None) -> Optional[List[ColoredDocument]]:
    resolved_path = corpus_path or CORPUS_ARTIFACT_PATH
    try:
        with open(resolved_path, "rb") as artifact:
            return cast(List[ColoredDocument], pickle.load(artifact))  # nosec B301  # nosemgrep
    except FileNotFoundError:
        logger.warning(
            "Encoded corpus artifact absent; query endpoint degraded",
            extra={"correlation_id": _correlation_id(), "expected_corpus": resolved_path},
        )
        return None


def _select_distance_calculator(metric: str, codebook: ColorCodebook) -> DistanceCalculator:
    if metric != "wasserstein":
        return JensenShannonDistanceCalculator(smoothing_epsilon=SMOOTHING_EPSILON)
    return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=SINKHORN_REGULARISATION)


def _log_experiment_resolution(runtime_context: ExperimentRuntimeContext) -> None:
    artifact_paths = runtime_context.artifact_paths
    logger.info(
        "Experiment configuration resolved",
        extra={
            "correlation_id": _correlation_id(),
            "experiment_config_path": runtime_context.experiment_config_path,
            "model_path": artifact_paths.model_path,
            "codebook_name": artifact_paths.codebook_name,
            "corpus_path": artifact_paths.corpus_path,
        },
    )


def _resolve_runtime_context() -> ExperimentRuntimeContext:
    try:
        runtime_context = build_experiment_runtime_context()
    except ExperimentConfigurationError as error:
        logger.error(
            "Experiment configuration could not be resolved",
            extra={"correlation_id": _correlation_id(), "reason": str(error)},
        )
        raise

    _log_experiment_resolution(runtime_context)
    return runtime_context


def get_container(runtime_context: Optional[ExperimentRuntimeContext] = None) -> Container:
    container = Container()

    context = runtime_context if runtime_context is not None else _resolve_runtime_context()
    container[ExperimentRuntimeContext] = lambda: context
    container[SynestheticConfig] = lambda: context.synesthetic_config
    container[ArtifactPaths] = lambda: context.artifact_paths

    codebook = _load_query_codebook(context.artifact_paths)

    container[ColorCodebook] = lambda: codebook
    container[DistanceCalculator] = lambda: _select_distance_calculator(DISTANCE_METRIC, codebook)  # type: ignore
    container[CompareDocumentsUseCase] = CompareDocumentsUseCase
    container[QueryByPaletteUseCase] = QueryByPaletteUseCase

    authenticator = get_basic_authenticator()
    security_dependency = SecurityDependency(authenticator)
    container[BasicAuthenticator] = lambda: authenticator
    container[SecurityDependency] = lambda: security_dependency

    health_checker = create_health_checker()
    container[HealthChecker] = lambda: health_checker  # type: ignore
    container[HealthUseCase] = HealthUseCase

    return container


_global_container: Optional[Container] = None


def get_global_container() -> Container:
    global _global_container
    if _global_container is None:
        _global_container = get_container()
    return _global_container


def _build_query_router(container: Container, corpus: Optional[List[ColoredDocument]]) -> APIRouter:
    if corpus is None:
        return create_unavailable_query_controller(CORPUS_UNAVAILABLE_DETAIL)
    return create_query_controller(container[QueryByPaletteUseCase], corpus)


def _log_query_readiness(codebook: ColorCodebook, corpus: Optional[List[ColoredDocument]]) -> None:
    logger.info(
        "Query API wiring complete",
        extra={
            "correlation_id": _correlation_id(),
            "codebook_name": CODEBOOK_NAME,
            "codebook_num_bins": codebook.num_bins,
            "corpus_documents": len(corpus) if corpus is not None else "absent",
            "distance_metric": DISTANCE_METRIC,
        },
    )


def create_app() -> FastAPI:
    application = FastAPI(title=APPLICATION_TITLE, version=APPLICATION_VERSION)

    container = get_container()
    codebook = container[ColorCodebook]
    corpus = _load_corpus(container[ArtifactPaths].corpus_path)

    application.include_router(_build_query_router(container, corpus))
    application.include_router(create_health_controller(container[HealthUseCase]))

    _log_query_readiness(codebook, corpus)

    return application


app = create_app()


def main(args: list) -> None:
    settings_provider = get_application_setting_provider()
    reload_setting = settings_provider.get("reload")
    host_setting = settings_provider.get("host")

    uvicorn.run(
        "colors_of_meaning.interface.api.main:app",
        reload=reload_setting,
        host=host_setting,
    )


def run() -> None:
    main(sys.argv[1:])

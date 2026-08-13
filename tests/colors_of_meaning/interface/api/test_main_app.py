import sys
import os
import json
import importlib

import numpy as np
import pytest
from assertpy import assert_that
from fastapi.openapi.utils import get_openapi
from fastapi.testclient import TestClient
from lagom import Container
from unittest.mock import patch, Mock, mock_open

import colors_of_meaning.interface.api.main as main_module
from colors_of_meaning.interface.api.main import (
    app,
    get_container,
    get_global_container,
    create_app,
    main,
    run,
)
from colors_of_meaning.application.use_case.compare_documents_use_case import CompareDocumentsUseCase
from colors_of_meaning.application.use_case.query_by_palette_use_case import QueryByPaletteUseCase
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import JensenShannonDistanceCalculator
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import WassersteinDistanceCalculator
from colors_of_meaning.interface.api.data_transfer_object.palette_query_dto import (
    PaletteQueryResponseDTO,
    QueryUnavailableDTO,
)
from colors_of_meaning.shared.configuration import (
    ArtifactPaths,
    ExperimentConfigurationError,
    ExperimentRuntimeContext,
)
from colors_of_meaning.shared.synesthetic_config import (
    SynestheticConfig,
    ProjectorConfig,
    CodebookConfig,
    TrainingConfig,
    DistanceConfig,
    DatasetConfig,
)

OPENAPI_JSON_FILE_PATH = "build/openapi.json"
OPENAPI_JSON_FILE_PATH_OPEN_FLAG = "w"

VALID_PALETTE_REQUEST = {"colors": [{"l": 50, "a": 0, "b": 0, "weight": 1.0}], "k": 5}
EMPTY_PALETTE_REQUEST = {"colors": [], "k": 5}


def _tiny_codebook() -> ColorCodebook:
    return ColorCodebook.create_uniform_grid(bins_per_dimension=2)


def _tiny_corpus() -> list:
    return [
        ColoredDocument(histogram=np.ones(8, dtype=np.float64) / 8, document_id="doc_a"),
        ColoredDocument(
            histogram=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            document_id="doc_b",
        ),
    ]


def _build_query_client(corpus: list) -> TestClient:
    with patch.object(main_module, "_load_query_codebook", return_value=_tiny_codebook()):
        with patch.object(main_module, "_load_corpus", return_value=corpus):
            return TestClient(create_app())


def _in_memory_runtime_context() -> ExperimentRuntimeContext:
    config = SynestheticConfig(
        projector=ProjectorConfig(),
        codebook=CodebookConfig(),
        training=TrainingConfig(),
        distance=DistanceConfig(),
        dataset=DatasetConfig(),
    )
    return ExperimentRuntimeContext(
        experiment_config_path="configs/base.yaml",
        synesthetic_config=config,
        artifact_paths=ArtifactPaths(),
    )


def create_openapi_json(app):
    os.makedirs(os.path.dirname(OPENAPI_JSON_FILE_PATH), exist_ok=True)

    with open(OPENAPI_JSON_FILE_PATH, OPENAPI_JSON_FILE_PATH_OPEN_FLAG) as json_output_file:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
            ),
            json_output_file,
        )


def test_should_create_openapi_json_file():
    if os.path.exists(OPENAPI_JSON_FILE_PATH):
        os.remove(OPENAPI_JSON_FILE_PATH)

    from colors_of_meaning.interface.api.main import app as rest

    create_openapi_json(rest)

    assert_that(OPENAPI_JSON_FILE_PATH).exists()


class TestMainApp:
    def test_should_have_application_title(self):
        assert_that(app.title).is_equal_to("Colors of Meaning API")

    def test_should_have_application_version(self):
        assert_that(app.version).is_equal_to("1.0.0")

    def test_should_have_container(self):
        container = get_container()

        assert_that(container).is_not_none()

    def test_should_expose_query_palette_route_when_app_is_built(self):
        paths = app.openapi().get("paths", {})

        assert_that(paths.get("/query/palette", {})).contains("post")

    def test_should_not_expose_create_coconut_route_when_app_is_built(self):
        paths = app.openapi().get("paths", {})

        assert_that(paths).does_not_contain_key("/coconut/")

    def test_should_not_expose_get_coconut_route_when_app_is_built(self):
        paths = app.openapi().get("paths", {})

        assert_that(paths).does_not_contain_key("/coconut/{id}")

    @patch("uvicorn.run")
    @patch("colors_of_meaning.interface.api.main.get_application_setting_provider")
    def test_main_function(self, mock_get_provider, mock_run):
        from colors_of_meaning.shared.configuration import ApplicationSettingProvider

        mock_provider = Mock(spec=ApplicationSettingProvider)
        mock_provider.get.side_effect = lambda key: True if key == "reload" else "0.0.0.0" if key == "host" else None
        mock_get_provider.return_value = mock_provider

        test_args = []
        main(test_args)

        mock_run.assert_called_once_with(
            "colors_of_meaning.interface.api.main:app",
            reload=True,
            host="0.0.0.0",
        )

    @patch("colors_of_meaning.interface.api.main.main")
    def test_run_function(self, mock_main):
        with patch.object(sys, "argv", ["script_name", "arg1", "arg2"]):
            run()

            mock_main.assert_called_once_with(["arg1", "arg2"])


class TestQueryContainerWiring:
    @patch("colors_of_meaning.interface.api.main._load_query_codebook")
    def test_should_resolve_color_codebook_when_container_built(self, mock_load):
        mock_load.return_value = _tiny_codebook()

        container = get_container()

        assert_that(container[ColorCodebook]).is_instance_of(ColorCodebook)

    @patch("colors_of_meaning.interface.api.main._load_query_codebook")
    def test_should_resolve_distance_calculator_when_container_built(self, mock_load):
        mock_load.return_value = _tiny_codebook()

        container = get_container()

        assert_that(container[DistanceCalculator]).is_instance_of(DistanceCalculator)

    @patch("colors_of_meaning.interface.api.main._load_query_codebook")
    def test_should_resolve_compare_documents_use_case_when_container_built(self, mock_load):
        mock_load.return_value = _tiny_codebook()

        container = get_container()

        assert_that(container[CompareDocumentsUseCase]).is_instance_of(CompareDocumentsUseCase)

    @patch("colors_of_meaning.interface.api.main._load_query_codebook")
    def test_should_resolve_query_by_palette_use_case_when_container_built(self, mock_load):
        mock_load.return_value = _tiny_codebook()

        container = get_container()

        assert_that(container[QueryByPaletteUseCase]).is_instance_of(QueryByPaletteUseCase)


class TestQueryArtifactLoaders:
    @patch("colors_of_meaning.interface.api.main.FileColorCodebookRepository.load")
    def test_should_load_codebook_when_artifact_present(self, mock_load):
        expected = _tiny_codebook()
        mock_load.return_value = expected

        assert_that(main_module._load_query_codebook()).is_same_as(expected)

    @patch("colors_of_meaning.interface.api.main.FileColorCodebookRepository.load")
    def test_should_fall_back_to_uniform_grid_when_codebook_artifact_absent(self, mock_load):
        mock_load.return_value = None

        codebook = main_module._load_query_codebook()

        assert_that(codebook.num_bins).is_equal_to(main_module.FALLBACK_BINS_PER_DIMENSION**3)

    def test_should_load_corpus_when_artifact_present(self):
        corpus = _tiny_corpus()

        with patch("builtins.open", mock_open(read_data=b"")):
            with patch.object(main_module.pickle, "load", return_value=corpus):
                loaded = main_module._load_corpus()

        assert_that(loaded).is_length(2)

    def test_should_return_none_when_corpus_artifact_absent(self, tmp_path):
        missing = tmp_path / "absent.pkl"

        with patch.object(main_module, "CORPUS_ARTIFACT_PATH", str(missing)):
            loaded = main_module._load_corpus()

        assert_that(loaded).is_none()

    def test_should_load_codebook_named_by_artifact_paths_when_paths_supplied(self):
        paths = ArtifactPaths(codebook_base_path="artifacts/codebooks", codebook_name="experiment_book")

        with patch.object(main_module, "FileColorCodebookRepository") as repository_class:
            repository_class.return_value.load.return_value = _tiny_codebook()
            main_module._load_query_codebook(paths)

        repository_class.return_value.load.assert_called_once_with("experiment_book")

    def test_should_read_supplied_corpus_path_when_path_provided(self, tmp_path):
        missing = tmp_path / "supplied.pkl"

        loaded = main_module._load_corpus(str(missing))

        assert_that(loaded).is_none()


class TestDistanceCalculatorSelection:
    def test_should_select_wasserstein_when_metric_is_wasserstein(self):
        calculator = main_module._select_distance_calculator("wasserstein", _tiny_codebook())

        assert_that(calculator).is_instance_of(WassersteinDistanceCalculator)

    def test_should_select_jensen_shannon_when_metric_is_not_wasserstein(self):
        calculator = main_module._select_distance_calculator("jensen_shannon", _tiny_codebook())

        assert_that(calculator).is_instance_of(JensenShannonDistanceCalculator)


class TestQueryApiContract:
    def test_should_return_palette_query_response_when_palette_posted(self):
        client = _build_query_client(_tiny_corpus())

        response = client.post("/query/palette", json=VALID_PALETTE_REQUEST)

        assert response.status_code == 200
        assert PaletteQueryResponseDTO.model_validate(response.json())

    def test_should_return_422_when_colors_is_empty(self):
        client = _build_query_client(_tiny_corpus())

        response = client.post("/query/palette", json=EMPTY_PALETTE_REQUEST)

        assert response.status_code == 422

    def test_should_return_503_when_corpus_artifact_unavailable(self):
        client = _build_query_client(None)

        response = client.post("/query/palette", json=VALID_PALETTE_REQUEST)

        assert response.status_code == 503
        assert QueryUnavailableDTO.model_validate(response.json())


class TestExperimentRuntimeWiring:
    def test_should_resolve_runtime_context_from_container_when_built(self):
        context = _in_memory_runtime_context()

        container = get_container(context)

        assert_that(container[ExperimentRuntimeContext]).is_same_as(context)

    def test_should_resolve_synesthetic_config_from_container_when_built(self):
        context = _in_memory_runtime_context()

        container = get_container(context)

        assert_that(container[SynestheticConfig]).is_same_as(context.synesthetic_config)

    def test_should_resolve_artifact_paths_from_container_when_built(self):
        context = _in_memory_runtime_context()

        container = get_container(context)

        assert_that(container[ArtifactPaths]).is_same_as(context.artifact_paths)


class TestRuntimeContextResolution:
    def test_should_return_context_when_experiment_config_resolves(self):
        context = main_module._resolve_runtime_context()

        assert_that(context).is_instance_of(ExperimentRuntimeContext)

    def test_should_reraise_when_experiment_config_cannot_be_resolved(self):
        with patch.object(
            main_module,
            "build_experiment_runtime_context",
            side_effect=ExperimentConfigurationError("boom"),
        ):
            with pytest.raises(ExperimentConfigurationError):
                main_module._resolve_runtime_context()


class TestContainerConstruction:
    def test_should_build_container_once_when_module_is_imported(self):
        importlib.reload(main_module)

        assert_that(main_module._global_container).is_none()

    def test_should_return_usable_container_from_get_global_container(self):
        main_module._global_container = None

        assert_that(get_global_container()).is_instance_of(Container)

    def test_should_cache_global_container_across_calls(self):
        main_module._global_container = None

        first = get_global_container()
        second = get_global_container()

        assert_that(first).is_same_as(second)

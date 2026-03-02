from unittest.mock import Mock
import numpy as np

from fastapi import FastAPI
from fastapi.testclient import TestClient

from colors_of_meaning.interface.api.controller.query_controller import (
    create_query_controller,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _create_test_app(mock_use_case: Mock) -> FastAPI:
    app = FastAPI()
    corpus = [
        ColoredDocument(
            histogram=np.ones(8, dtype=np.float64) / 8,
            document_id="doc_1",
        )
    ]
    router = create_query_controller(mock_use_case, corpus)
    app.include_router(router)
    return app


class TestQueryController:
    def test_should_return_200_for_valid_palette_query(self) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = [("doc_1", 0.5)]

        app = _create_test_app(mock_use_case)
        client = TestClient(app)

        response = client.post(
            "/query/palette",
            json={
                "colors": [{"l": 50, "a": 0, "b": 0, "weight": 1.0}],
                "k": 5,
            },
        )

        assert response.status_code == 200

    def test_should_return_matches(self) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = [("doc_1", 0.5)]

        app = _create_test_app(mock_use_case)
        client = TestClient(app)

        response = client.post(
            "/query/palette",
            json={
                "colors": [{"l": 50, "a": 0, "b": 0, "weight": 1.0}],
                "k": 5,
            },
        )

        assert len(response.json()["matches"]) == 1

    def test_should_return_query_colors_count(self) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = []

        app = _create_test_app(mock_use_case)
        client = TestClient(app)

        response = client.post(
            "/query/palette",
            json={
                "colors": [
                    {"l": 50, "a": 0, "b": 0, "weight": 1.0},
                    {"l": 75, "a": 10, "b": -10, "weight": 0.5},
                ],
                "k": 5,
            },
        )

        assert response.json()["query_colors"] == 2

    def test_should_return_422_for_empty_colors(self) -> None:
        mock_use_case = Mock()

        app = _create_test_app(mock_use_case)
        client = TestClient(app)

        response = client.post(
            "/query/palette",
            json={"colors": [], "k": 5},
        )

        assert response.status_code == 422

    def test_should_call_use_case_with_palette(self) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = []

        app = _create_test_app(mock_use_case)
        client = TestClient(app)

        client.post(
            "/query/palette",
            json={
                "colors": [{"l": 50, "a": 0, "b": 0, "weight": 1.0}],
                "k": 3,
            },
        )

        mock_use_case.execute.assert_called_once()

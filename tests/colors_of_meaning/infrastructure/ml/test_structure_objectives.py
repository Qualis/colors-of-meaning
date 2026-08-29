import numpy as np
import pytest
import torch

from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structure_objectives import (
    RANKING_MARGIN,
    _teacher_ordered_triplets,
    cosine_centred,
    cosine_similarity_matrix,
    delta_e_correlation,
    margin_ranking,
    offdiagonal_entries,
    pairwise_lab_distance,
)

ARCHIVED_COSINE_CENTRED_LOSS = 1.0927273035049438
OBJECTIVES = [cosine_centred, delta_e_correlation, margin_ranking]


def _teacher_embeddings() -> torch.Tensor:
    generator = np.random.default_rng(11)
    return torch.tensor(generator.normal(size=(8, 6)), dtype=torch.float32)


def _lab_from_teacher(teacher: torch.Tensor, scale: float) -> torch.Tensor:
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], dtype=torch.float32)
    return (teacher[:, :2] @ projection.t()) * scale


def _agreeing_lab(teacher: torch.Tensor) -> torch.Tensor:
    return teacher[:, :3] * 10.0


def _antipodal_lab() -> torch.Tensor:
    return torch.tensor(
        [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, -12.0, 0.0], [0.0, 0.0, 8.0], [0.0, 0.0, -8.0]]
    )


def _radially_stretched_antipodal_lab() -> torch.Tensor:
    stretched = _antipodal_lab()
    stretched[0:2] = stretched[0:2] * 4.0
    return stretched


def _matched_teacher_and_lab() -> tuple:
    generator = np.random.default_rng(19)
    directions = torch.tensor(generator.normal(size=(8, 3)), dtype=torch.float32)
    unit_directions = directions / directions.norm(dim=1, keepdim=True)
    return unit_directions, unit_directions * 1000.0


def _disagreeing_lab(teacher: torch.Tensor) -> torch.Tensor:
    return torch.flip(_agreeing_lab(teacher), dims=[0])


class TestOffdiagonalEntries:
    def test_should_exclude_self_pairs_when_building_offdiagonal_similarity(self) -> None:
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        entries = offdiagonal_entries(matrix)

        expected = torch.tensor([2.0, 3.0, 4.0, 6.0, 7.0, 8.0])
        assert torch.equal(torch.sort(entries).values, expected)


class TestCosineSimilarityMatrix:
    def test_should_place_ones_on_the_diagonal_when_vectors_are_compared_to_themselves(self) -> None:
        vectors = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

        similarity = cosine_similarity_matrix(vectors)

        assert torch.allclose(torch.diagonal(similarity), torch.ones(2), atol=1e-6)


class TestPairwiseLabDistance:
    def test_should_measure_euclidean_lab_separation_when_two_colors_differ(self) -> None:
        lab = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])

        distance = pairwise_lab_distance(lab)

        assert distance[0, 1].item() == pytest.approx(5.0, abs=1e-4)


class TestCosineCentred:
    def test_should_match_the_archived_reference_value_when_run_on_the_fixed_batch(self) -> None:
        mapper = PyTorchColorMapper(input_dim=8, hidden_dim_1=6, hidden_dim_2=4, dropout_rate=0.0, seed=17)
        generator = np.random.default_rng(3)
        batch = torch.tensor(generator.normal(size=(6, 8)), dtype=torch.float32)
        mapper.network.eval()

        with torch.no_grad():
            loss = mapper._structure_loss(batch)

        assert loss.item() == pytest.approx(ARCHIVED_COSINE_CENTRED_LOSS, abs=1e-7)

    def test_should_ignore_radial_magnitude_when_an_antipodal_pair_moves_along_its_ray(self) -> None:
        teacher = _teacher_embeddings()[:6]

        stretched_loss = cosine_centred(_radially_stretched_antipodal_lab(), teacher)

        assert stretched_loss.item() == pytest.approx(cosine_centred(_antipodal_lab(), teacher).item(), abs=1e-6)


class TestDeltaECorrelation:
    def test_should_stay_invariant_when_the_lab_output_is_uniformly_scaled(self) -> None:
        teacher = _teacher_embeddings()
        lab = _lab_from_teacher(teacher, scale=1.0)

        scaled_loss = delta_e_correlation(lab * 37.0, teacher)

        assert scaled_loss.item() == pytest.approx(delta_e_correlation(lab, teacher).item(), abs=1e-5)

    def test_should_respond_to_radial_magnitude_when_an_antipodal_pair_moves_along_its_ray(self) -> None:
        teacher = _teacher_embeddings()[:6]

        stretched_loss = delta_e_correlation(_radially_stretched_antipodal_lab(), teacher)

        assert stretched_loss.item() != pytest.approx(delta_e_correlation(_antipodal_lab(), teacher).item(), abs=1e-3)

    def test_should_return_a_neutral_loss_when_every_color_is_identical(self) -> None:
        teacher = _teacher_embeddings()

        loss = delta_e_correlation(torch.zeros(len(teacher), 3), teacher)

        assert loss.item() == pytest.approx(1.0, abs=1e-4)


class TestMarginRanking:
    def test_should_reach_zero_loss_when_the_student_reproduces_the_teacher_geometry(self) -> None:
        teacher, lab = _matched_teacher_and_lab()

        loss = margin_ranking(lab, teacher)

        assert loss.item() == 0.0

    def test_should_penalise_a_collapsed_student_by_the_margin(self) -> None:
        teacher = _teacher_embeddings()

        loss = margin_ranking(torch.zeros(len(teacher), 3), teacher)

        assert loss.item() == pytest.approx(RANKING_MARGIN, abs=1e-6)

    def test_should_only_enumerate_triplets_of_three_distinct_samples(self) -> None:
        teacher = _teacher_embeddings()

        anchors, nearer, farther = _teacher_ordered_triplets(cosine_similarity_matrix(teacher))

        assert bool(((anchors != nearer) & (anchors != farther) & (nearer != farther)).all())

    def test_should_order_every_triplet_so_the_nearer_item_is_the_teacher_favourite(self) -> None:
        teacher = _teacher_embeddings()
        similarity = cosine_similarity_matrix(teacher)

        anchors, nearer, farther = _teacher_ordered_triplets(similarity)

        assert bool((similarity[anchors, nearer] > similarity[anchors, farther]).all())

    def test_should_return_zero_when_no_teacher_triplet_clears_the_ordering_gap(self) -> None:
        teacher = torch.ones(4, 5)

        loss = margin_ranking(torch.randn(4, 3), teacher)

        assert loss.item() == 0.0


class TestObjectiveContract:
    @pytest.mark.parametrize("objective", OBJECTIVES)
    def test_should_return_a_finite_scalar_when_the_batch_is_typical(self, objective) -> None:
        teacher = _teacher_embeddings()

        loss = objective(_lab_from_teacher(teacher, scale=1.0), teacher)

        assert loss.shape == torch.Size([]) and torch.isfinite(loss)

    @pytest.mark.parametrize("objective", OBJECTIVES)
    def test_should_return_zero_when_the_batch_holds_a_single_sample(self, objective) -> None:
        teacher = _teacher_embeddings()[:1]

        loss = objective(torch.zeros(1, 3), teacher)

        assert loss.item() == 0.0

    @pytest.mark.parametrize("objective", OBJECTIVES)
    def test_should_prefer_the_agreeing_student_when_scored_against_a_disagreeing_one(self, objective) -> None:
        teacher = _teacher_embeddings()

        agreeing = objective(_agreeing_lab(teacher), teacher)
        disagreeing = objective(_disagreeing_lab(teacher), teacher)

        assert agreeing.item() < disagreeing.item()

    @pytest.mark.parametrize("objective", OBJECTIVES)
    def test_should_carry_a_gradient_back_to_the_lab_output(self, objective) -> None:
        teacher = _teacher_embeddings()
        lab = _lab_from_teacher(teacher, scale=1.0).detach().requires_grad_(True)

        objective(lab, teacher).backward()

        assert lab.grad is not None and torch.isfinite(lab.grad).all()

from typing import Protocol, Tuple

import torch
import torch.nn as nn

LAB_DISTANCE_SCALE = 100.0
RANKING_MARGIN = 0.1
TEACHER_ORDERING_GAP = 0.05
STANDARDISATION_EPSILON = 1e-8
SQUARED_DISTANCE_EPSILON = 1e-12

TripletIndices = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def offdiagonal_entries(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    keep = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix[keep]


class StructureObjective(Protocol):
    def __call__(self, lab_output: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def cosine_centred(lab_output: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    teacher_similarity = cosine_similarity_matrix(teacher_embeddings).detach()
    centred_lab = lab_output - lab_output.mean(dim=0, keepdim=True)
    student_similarity = cosine_similarity_matrix(centred_lab)
    student_offdiagonal = offdiagonal_entries(student_similarity)

    if student_offdiagonal.numel() == 0:
        return student_similarity.sum() * 0.0

    teacher_offdiagonal = offdiagonal_entries(teacher_similarity)
    return nn.functional.mse_loss(student_offdiagonal, teacher_offdiagonal)


def delta_e_correlation(lab_output: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    student_distance = offdiagonal_entries(pairwise_lab_distance(lab_output))

    if student_distance.numel() == 0:
        return lab_output.sum() * 0.0

    teacher_dissimilarity = offdiagonal_entries(1.0 - cosine_similarity_matrix(teacher_embeddings).detach())
    return 1.0 - _pearson_correlation(student_distance, teacher_dissimilarity)


def margin_ranking(lab_output: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    teacher_similarity = cosine_similarity_matrix(teacher_embeddings).detach()
    anchors, nearer, farther = _teacher_ordered_triplets(teacher_similarity)

    if anchors.numel() == 0:
        return lab_output.sum() * 0.0

    student_distance = pairwise_lab_distance(lab_output) / LAB_DISTANCE_SCALE
    nearer_distance = student_distance[anchors, nearer]
    farther_distance = student_distance[anchors, farther]

    return nn.functional.margin_ranking_loss(
        nearer_distance,
        farther_distance,
        torch.full_like(nearer_distance, -1.0),
        margin=RANKING_MARGIN,
    )


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    normalized = nn.functional.normalize(vectors, p=2, dim=1)
    return normalized @ normalized.t()


def pairwise_lab_distance(lab_output: torch.Tensor) -> torch.Tensor:
    differences = lab_output.unsqueeze(1) - lab_output.unsqueeze(0)
    return torch.sqrt((differences**2).sum(dim=2) + SQUARED_DISTANCE_EPSILON)


def _pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (_standardised(first) * _standardised(second)).mean()


def _standardised(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + STANDARDISATION_EPSILON)


def _teacher_ordered_triplets(teacher_similarity: torch.Tensor) -> TripletIndices:
    ordering_gaps = teacher_similarity.unsqueeze(2) - teacher_similarity.unsqueeze(1)
    selected = _distinct_triplet_mask(teacher_similarity) & (ordering_gaps > TEACHER_ORDERING_GAP)
    anchors, nearer, farther = torch.nonzero(selected, as_tuple=True)
    return anchors, nearer, farther


def _distinct_triplet_mask(teacher_similarity: torch.Tensor) -> torch.Tensor:
    size = teacher_similarity.shape[0]
    index = torch.arange(size, device=teacher_similarity.device)
    anchors = index.view(-1, 1, 1)
    nearer = index.view(1, -1, 1)
    farther = index.view(1, 1, -1)
    return (anchors != nearer) & (anchors != farther) & (nearer != farther)

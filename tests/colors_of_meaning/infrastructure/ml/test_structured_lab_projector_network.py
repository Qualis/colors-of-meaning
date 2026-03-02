import torch

from colors_of_meaning.infrastructure.ml.structured_lab_projector_network import (
    StructuredLabProjectorNetwork,
)


class TestStructuredLabProjectorNetwork:
    def test_should_initialize_network(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, dropout_rate=0.2)

        assert isinstance(network, torch.nn.Module)

    def test_should_forward_pass_with_correct_shape(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(2, 10)

        output = network.forward(input_tensor)

        assert output.shape == (2, 3)

    def test_should_output_valid_lightness_range(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(5, 10)

        output = network.forward(input_tensor)

        assert torch.all(output[:, 0] >= 0)
        assert torch.all(output[:, 0] <= 100)

    def test_should_output_valid_ab_range(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, max_chroma=128.0)
        input_tensor = torch.randn(5, 10)

        output = network.forward(input_tensor)

        assert torch.all(output[:, 1] >= -128.0)
        assert torch.all(output[:, 1] <= 128.0)
        assert torch.all(output[:, 2] >= -128.0)
        assert torch.all(output[:, 2] <= 128.0)

    def test_should_forward_structured_return_three_components(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(3, 10)

        lightness, hue_angle, chroma = network.forward_structured(input_tensor)

        assert lightness.shape == (3, 1)
        assert hue_angle.shape == (3, 1)
        assert chroma.shape == (3, 1)

    def test_should_output_lightness_in_valid_range_from_structured(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(5, 10)

        lightness, _, _ = network.forward_structured(input_tensor)

        assert torch.all(lightness >= 0)
        assert torch.all(lightness <= 100)

    def test_should_output_hue_angle_in_valid_range(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(5, 10)

        _, hue_angle, _ = network.forward_structured(input_tensor)

        assert torch.all(hue_angle >= -torch.pi)
        assert torch.all(hue_angle <= torch.pi)

    def test_should_output_chroma_in_valid_range(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, max_chroma=100.0)
        input_tensor = torch.randn(5, 10)

        _, _, chroma = network.forward_structured(input_tensor)

        assert torch.all(chroma >= 0)
        assert torch.all(chroma <= 100.0)

    def test_should_use_custom_max_chroma(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, max_chroma=50.0)

        assert network.max_chroma == 50.0

    def test_should_have_separate_heads(self) -> None:
        network = StructuredLabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)

        assert network.backbone is not None
        assert network.lightness_head is not None
        assert network.hue_head is not None
        assert network.chroma_head is not None

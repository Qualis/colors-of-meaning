from pytest_archon.rule import archrule


def test_should_maintain_domain_layer_independence():
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


def test_should_maintain_application_interface_independence():
    (
        archrule(
            "Application Interface",
            comment="Application layer should not depend on interface layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.interface.*")
        .check("colors_of_meaning")
    )


def test_should_maintain_application_infrastructure_independence():
    (
        archrule(
            "Application Infrastructure",
            comment="Application layer should not depend on infrastructure layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.infrastructure.*")
        .check("colors_of_meaning")
    )


def test_should_use_application_use_case_in_controller():
    (
        archrule(
            "Controller Use Case",
            comment="Interface controller should depend on application use cases",
        )
        .match("colors_of_meaning.interface.api.controller.*")
        .should_import("colors_of_meaning.application.use_case.*")
        .check("colors_of_meaning")
    )


def test_should_not_use_domain_model_in_data_transfer_object():
    (
        archrule(
            "Data Transfer Object Model",
            comment="Data Transfer Object should not depend directly on domain model",
        )
        .match("colors_of_meaning.interface.api.data_transfer_object.*")
        .should_not_import("colors_of_meaning.domain.model.*")
        .check("colors_of_meaning")
    )


def _does_not_import_argon2(module, direct_imports, all_imports):
    return not any(
        imported == "argon2" or imported.startswith("argon2.") for imported in all_imports.get(module, set())
    )


def test_should_not_ship_password_hashing_in_any_layer():
    (
        archrule(
            "No Password Hashing",
            comment="The package ships no authentication, so no module may import Argon2",
        )
        .match("colors_of_meaning.*")
        .should(_does_not_import_argon2, "does_not_import_argon2")
        .check("colors_of_meaning")
    )


def test_should_not_have_circular_dependencies():
    (
        archrule(
            "No Circular Dependencies",
            comment="No modules should have circular dependencies",
        )
        .match("colors_of_meaning.*")
        .should(
            lambda module, direct_imports, all_imports: (
                module not in direct_imports and module not in all_imports.get(module, set())
            ),
            "no_circular_dependencies",
        )
        .check("colors_of_meaning")
    )


def _does_not_import_framework(module, direct_imports, all_imports):
    frameworks = ("fastapi", "torch", "tyro", "yaml", "ot")
    return not any(
        imported == framework or imported.startswith(f"{framework}.")
        for imported in all_imports.get(module, set())
        for framework in frameworks
    )


def test_should_keep_grounding_report_free_of_frameworks():
    (
        archrule(
            "Grounding Report Purity",
            comment="The grounding report value object must not import ML, web or serialisation frameworks",
        )
        .match("colors_of_meaning.domain.model.grounding_report")
        .should(_does_not_import_framework, "does_not_import_framework")
        .check("colors_of_meaning")
    )


def test_should_keep_evaluate_grounding_use_case_off_infrastructure_and_interface():
    (
        archrule(
            "Evaluate Grounding Use Case Boundaries",
            comment="The grounding use case depends only on domain abstractions, never on infrastructure or interface",
        )
        .match("colors_of_meaning.application.use_case.evaluate_grounding_use_case")
        .should_not_import("colors_of_meaning.infrastructure.*", "colors_of_meaning.interface.*")
        .check("colors_of_meaning")
    )


def test_should_maintain_shared_module_independence():
    (
        archrule(
            "Shared Module Dependencies",
            comment="Shared module should not depend on application, infrastructure or interface",
        )
        .match("colors_of_meaning.shared.*")
        .should_not_import(
            "colors_of_meaning.application.*", "colors_of_meaning.infrastructure.*", "colors_of_meaning.interface.*"
        )
        .check("colors_of_meaning")
    )

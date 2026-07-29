============
Contributing
============

Welcome to ``colors-of-meaning`` contributor's guide. This project embraces clean architecture, Hexagonal Architecture, and Domain-Driven Design principles, so contributions should align with its modular, testable, and secure structure.

If you're new to Git or contributing to open source projects, check out the `FreeCodeCamp contribution guide`_ and `contribution-guide.org`_.

All contributors are expected to act in accordance with the `Python Software Foundation's Code of Conduct`_.

Issue Reports
=============

Found a bug or have an idea? Please search the `issue tracker`_ first, including closed issues. If none match, feel free to open a new one.

Be sure to include:

- OS and Python version
- Steps to reproduce
- A minimal working example, if possible

Documentation Improvements
==========================

The primary project documentation is ``README.MD``; the architecture rationale and design notes live in ``docs/design.md``. Improve either via a pull request.

Code Contributions
==================

Our architecture separates the system into domain, application, infrastructure, and interface layers. Follow this structure when contributing.

Setup
-----

Dependencies are managed with `uv`_. ``pyproject.toml`` holds the abstract requirements and
``uv.lock`` holds the exact, cross-platform resolved closure; both are committed.

1. Fork the `repository`_ and clone it locally.

.. code:: bash

    git clone git@github.com:Qualis/colors-of-meaning.git
    cd colors-of-meaning

2. Install ``uv`` (once):

.. code:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

3. Create the development environment from the lock. This writes ``.venv`` with exactly the
   versions in ``uv.lock`` — no resolution, no drift:

.. code:: bash

    uv sync --locked --extra testing

4. Install the quality gate runner. ``tox-uv`` makes ``tox`` build its environments with
   ``uv`` from ``uv.lock``, so a gate run takes seconds rather than rebuilding the ML stack:

.. code:: bash

    uv tool install --python 3.11 --with tox-uv tox

The ``--python 3.11`` matters: the gate environment is named ``default``, so ``tox`` runs it on
whatever interpreter ``tox`` itself was installed with. Pinning 3.11 here is what makes a local
run match the PR gate.

Changing dependencies
---------------------

Edit the abstract requirement in ``pyproject.toml``, then regenerate and commit the lock:

.. code:: bash

    uv lock

``tox`` runs ``uv sync --locked``, which fails if ``uv.lock`` is out of date with respect to
``pyproject.toml`` — so a forgotten ``uv lock`` is caught by the gate, not discovered later.

``torch`` resolves from an explicit PyTorch CPU index (``[tool.uv.sources]``) so the lock never
carries a CUDA stack the CPU-only gate cannot use, and is held at a fixed version for
reproducibility of the committed evaluation results; see ``docs/security/audit-suppressions.md``.

Because that index redirects to PyTorch's CDN, ``uv.lock`` records the resolved wheel URLs on
``download-r2.pytorch.org`` rather than the ``download.pytorch.org`` URL configured above. If
those URLs ever stop resolving, the fix is ``uv lock`` to re-resolve them — not a change to
``[[tool.uv.index]]``.

.. note::

   The lock is **CPU-only on every platform** — that is deliberate, since the gate and CI have no
   GPU. If you have a CUDA machine and want to train on it, install a CUDA build over the synced
   environment (``uv pip install torch --index-url https://download.pytorch.org/whl/cu128``).
   That deviates from ``uv.lock`` on purpose, so do not commit a lock regenerated that way.

Start Coding
------------

1. Create a new branch:

.. code:: bash

    git checkout -b my-feature

2. Implement changes. Structure your code within the appropriate layer (e.g., ``domain/model``, ``application/use_case``).

3. Add docstrings and meaningful tests.

4. Add yourself to ``AUTHORS.rst``.

Testing
-------

Run all checks via:

.. code:: bash

    tox

We require 100% test coverage and use:

- ruff (lint + format)
- bandit
- semgrep
- pip-audit
- radon
- xenon (grade A)
- mypy

Push and PR
-----------

1. Push your branch:

.. code:: bash

    git push -u origin my-feature

2. Open a Pull Request via GitHub.

Every pull request targeting ``main`` runs the **PR Gate** workflow, which executes the full ``tox`` quality gate (all static checks plus the test suite at 100% coverage) as well as ``shellcheck`` and ``ansible-lint``. A pull request must be green before it is merged.

.. note::

   Maintainers: make the ``pr-gate`` status check **required** under *Settings → Branches → Branch protection rules* for ``main`` so the gate blocks merges rather than only reporting. This is a one-time repository setting and cannot be committed as a file.

Release Process (Maintainers Only)
==================================

1. Tag the release:

.. code:: bash

    git tag vX.Y.Z
    git push upstream vX.Y.Z

2. Clean old builds:

.. code:: bash

    tox -e clean

3. Build and publish:

.. code:: bash

    tox -e build
    tox -e publish -- --repository pypi

Resources
==================================

.. _FreeCodeCamp contribution guide: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _issue tracker: https://github.com/Qualis/colors-of-meaning/issues
.. _repository: https://github.com/Qualis/colors-of-meaning
.. _uv: https://docs.astral.sh/uv/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/


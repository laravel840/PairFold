"""Setuptools installer for the PairFold package.

Install in editable mode from the repository root:

    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = ROOT / "pairfold" / "README.md"

setup(
    name="pairfold",
    version="0.1.0",
    description=(
        "Local peptide/protein structure prediction from PDB-trained "
        "fragment torsions and contact anchors."
    ),
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="PairFold",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["pairfold", "pairfold.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "biopython>=1.81",
        "tqdm>=4.66",
        "fastapi>=0.110",
        "uvicorn>=0.27",
        "pydantic>=2.0",
    ],
    extras_require={
        "bench": [
            "requests>=2.31",
            "pandas>=2.0",
            "matplotlib>=3.7",
        ],
        "esm": [
            "fair-esm>=2.0.0",
        ],
        "dev": [
            "requests>=2.31",
            "pandas>=2.0",
            "matplotlib>=3.7",
            "fair-esm>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pairfold-server=pairfold.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

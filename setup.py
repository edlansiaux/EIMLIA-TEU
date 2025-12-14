#!/usr/bin/env python
"""
Setup script for EIMLIA-TEU
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Lire les requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
else:
    requirements = []

setup(
    name="eimlia-teu",
    version="1.0.0",
    author="CHU Lille - Équipe EIMLIA",
    author_email="eimlia@chu-lille.fr",
    description="Étude d'Impact des Modèles de Langage et d'IA sur le triage aux urgences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chu-lille/eimlia-teu",
    project_urls={
        "Documentation": "https://eimlia.readthedocs.io",
        "Bug Tracker": "https://github.com/chu-lille/eimlia-teu/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eimlia-train=scripts.train_models:main",
            "eimlia-simulate=scripts.run_simulation:main",
            "eimlia-api=src.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)

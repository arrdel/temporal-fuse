"""Setup configuration for TemporalFusion."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="temporalfusion",
    version="0.1.0",
    description="TemporalFusion: Hierarchical Temporal Aggregation with Contrastive Learning for Long-Form Video Understanding",
    author="TemporalFusion Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)

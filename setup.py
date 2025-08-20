from setuptools import setup, find_packages

setup(
    name="cbm-library",
    version="0.1.0",
    description="Unified Concept Bottleneck Model Library",
    author="ISRP 25 - Weng Lab",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.20.0",
        "openai>=0.27.0",
        "clip-by-openai",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "pyyaml>=6.0",
        "apricot-select>=0.6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "sentence-transformers>=2.0.0",
        "datasets>=2.0.0",
    ],
    python_requires=">=3.8",
)
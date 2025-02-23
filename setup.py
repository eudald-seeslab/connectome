from setuptools import setup, find_packages

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "torch-geometric>=2.4.0",
    "torch-scatter",
    "torch-sparse",
    "wandb",
    "numpy>=1.26.0",
    "pandas",
    "scipy>=1.11.4",
    "pillow>=10.2.0",
    "matplotlib>=3.8.2",
    "networkx>=3.2.1",
    "tqdm>=4.65.0",
    "seaborn>=0.13.2",
    "imageio>=2.34.1",
]

# Optional dependencies for visualization
VIZ_REQUIRES = [
    "plotly",
    "dash",
]

# Development dependencies
DEV_REQUIRES = [
    "black>=24.1.1",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

setup(
    name="connectome",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "viz": VIZ_REQUIRES,
        "dev": DEV_REQUIRES,
        "all": VIZ_REQUIRES + DEV_REQUIRES,
    },
    author="Your Name",
    author_email="eudald.correig@urv.cat",
    description="A package for neural connectome analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eudald-seeslab/connectome",
)

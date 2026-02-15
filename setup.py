from setuptools import setup, find_packages

# Core dependencies -- model / data / training deps are now provided by trainyourfly
INSTALL_REQUIRES = [
    "train-your-fly[wandb]",
    "scipy>=1.11.4",
    "imageio>=2.34.1",
]

# Research-specific dependencies
RESEARCH_REQUIRES = [
    "plotly",
    "dash",
    "joblib>=1.2.0",
    "numba",
    "umap-learn",
    "scikit-learn",
]

# Development dependencies
DEV_REQUIRES = [
    "black>=24.1.1",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

setup(
    name="connectome",
    version="0.2.0",
    packages=find_packages(),
    py_modules=["paths"],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "research": RESEARCH_REQUIRES,
        "dev": DEV_REQUIRES,
        "all": RESEARCH_REQUIRES + DEV_REQUIRES,
    },
    author="Eudald Correig",
    author_email="eudald.correig@urv.cat",
    description="Research analysis tools for the Drosophila connectome project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eudald-seeslab/connectome",
)

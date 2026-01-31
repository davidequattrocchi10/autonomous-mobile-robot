from setuptools import setup, find_packages

setup(
    name="autonomous-mobile-robot",
    version="0.1.0",
    author="Davide Quattrocchi",
    description="AGV/AMR navigation system with path planning and reinforcement learning",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
)
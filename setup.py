from setuptools import setup, find_packages

setup(
    name="data-analysis",
    version="0.1.0",
    description="Data analysis utilities for data inspection and visualization.",
    packages=find_packages(include=["data_analysis", "data_analysis.*"]),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.20",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "plotly>=5.0",
        "ipython>=7.0",
        "google-colab>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

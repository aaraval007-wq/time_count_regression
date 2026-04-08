from setuptools import setup, find_packages

setup(
    name="count-regressor",           # name on PyPI
    version="0.1.0",
    author="Aarav Agarwal",
    author_email="aaraval007@gmail.com",
    description="A general-purpose class for count-based time series regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aaraval007-wq/time_count_regression",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scikit-learn",
        "scipy",
    ],
    python_requires=">=3.8",
)
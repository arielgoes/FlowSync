from setuptools import setup, find_packages

setup(
    name="flowsync",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "scapy",
        "xgboost",
        "statsmodels",
        "tqdm",
        "joblib",
    ],
)

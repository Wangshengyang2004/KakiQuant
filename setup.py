from setuptools import setup, find_packages

setup(
    name="kaki",
    version="0.1.0",
    author="Shengyang Wang",
    author_email="shengyang.wang2@dukekunshan.edu.cn",
    description="A brief description of your project",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Wangshengyang2004/KakiQuant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "python-dotenv",
        "lightgbm",
        "setuptools",
        "scikit-learn",
        "pymongo",
        "catboost",
        "xgboost",
        "pandas",
        "matplotlib",
        "seaborn",
        "openfe",
        "tqdm",
        "python-okx",
        # Note: cudf is handled separately due to CUDA version dependencies
    ],
    extras_require={
        "CUDA11": ["cudf-cu11"],
        "CUDA12": ["cudf-cu12"],
    },
    project_urls={  # Optional
        "Bug Tracker": "https://github.com/Wangshengyang2004/KakiQuant/issues",
    },
)

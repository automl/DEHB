import os

import setuptools

import datetime


name = "DEHB"
package_name = "dehb"
author = "Neeratyoy, Noor, Frank"
author_email = "mallik@cs.uni-freiburg.de"
description = "Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization"
url = "https://github.com/automl/DEHB"
project_urls = {
    "Documentation": "https://automl.org.github.io/DEHB/main",
    "Source Code": "https://github.com/automl.org/dehb",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, Neeratyoy, Noor, Frank"
version = "0.0.3"

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    """
    Read in a files contents

    Parameters
    ----------
    filepath : str
        The name of the file.

    Returns
    -------
    str
        The contents of the file.
    """

    with open(filepath, "r", encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        # Docs
        "automl_sphinx_theme",
        # Others
        "isort",
        "black",
        "pydocstyle",
        "flake8",
        "pre-commit",
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=read_file(os.path.join(HERE, "requirements.txt")).split("\n"),
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

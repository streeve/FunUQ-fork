import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FunUQ",
    version="0.1.0",
    author="Sam Reeve",
    author_email="samtreeve@gmail.com",
    description="Functional uncertainty quantification for molecular dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.rcac.purdue.edu/StrachanGroup/FunUQ",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)

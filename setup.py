from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Packages",
    version="0.1",
    author="khanderaya",
    packages=find_packages(),
    install_requires = requirements,
)
from setuptools import setup, find_packages

setup(
    name='optimizer_zoo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your package dependencies here
        "transformers>=4.36.2",
        "trl>=0.7.4",
        "torchvision",
        "torch",
        "diffuser"
    ],
)
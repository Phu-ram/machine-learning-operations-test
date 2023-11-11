from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["catboost=>1.2.2",
    "seaborn=>0.13.0",
    "gcsfs=>2023.10.0"]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)

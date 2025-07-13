from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='CUSTOMER-SEGMENTATION-MLOPS',
    version='0.1.0',
    author='Duc Tran',
    packages=find_packages(),
    install_requires=requirements,
)
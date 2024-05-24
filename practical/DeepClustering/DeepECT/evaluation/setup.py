from setuptools import setup, find_packages
import practical


def _load_readme():
    with open("README.md", "r") as file:
        readme = file.read()
    return readme


setup(
    name='practical',
    version=practical.__version__,
    packages=find_packages(exclude=["*tests"]),
    url='https://github.com/collinleiber/LMU_Master_Practical_SoSe24',
    author='Collin Leiber',
    author_email='leiber@dbs.ifi.lmu.de',
    description='Dummy project for the LMU master practical summer semester 2024',
    long_description=_load_readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=['numpy',
                      'scipy',
                      'scikit-learn',
                      'matplotlib',
                      'pandas']
)

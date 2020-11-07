from distutils.core import setup
import re
from pathlib import Path

__name__ = 'conjugate_prior'
__path__ = Path(__file__).parent.absolute()
with (__path__/__name__/"__init__py").open('r') as f:
    version = re.findall(r"__version__\s*=\s*['\"]([\d.]+)['\"]", f.read())[0]

try:
    with (__path__/"README.md").open('r') as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

setup(
    name=__name__,
    packages=[__name__],
    install_requires=[
        'setuptools',
        'scipy',
        'numpy',
        'matplotlib',
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    version=version,
    description='Bayesian Statistics conjugate prior distributions',
    author='Uri Goren',
    author_email='uri@goren.ml',
    url='https://github.com/urigoren/conjugate_prior',
    keywords=['conjugate', 'bayesian', 'stats', 'statistics', 'bayes', 'distribution', 'probability', 'hypothesis',
              'modelling'],
    classifiers=[],
)

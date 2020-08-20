from distutils.core import setup
import os, re

__name__ = 'conjugate_prior'
with open(__file__.replace("setup.py", __name__ + os.sep + "__init__.py"), 'r') as f:
    version = re.findall(r"__version__\s*=\s*['\"]([\d.]+)['\"]", f.read())[0]

with open(__file__.replace("setup.py", "README.md"), 'r') as f:
    readme = f.read()

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

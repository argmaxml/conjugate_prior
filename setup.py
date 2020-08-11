from distutils.core import setup

with open(__file__.replace("setup.py", "README.md"), 'r') as f:
    readme = f.read()
setup(
    name='conjugate_prior',
    packages=['conjugate_prior'],
    install_requires=[
        'setuptools',
        'scipy',
        'numpy',
        'matplotlib',
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    version='0.38',
    description='Bayesian Statistics conjugate prior distributions',
    author='Uri Goren',
    author_email='uri@goren.ml',
    url='https://github.com/urigoren/conjugate_prior',
    keywords=['conjugate', 'bayesian', 'stats', 'statistics', 'bayes', 'distribution', 'probability', 'hypothesis',
              'modelling'],
    classifiers=[],
)

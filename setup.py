from distutils.core import setup

setup(
    name="conjugate_prior",
    packages=["conjugate_prior"],
    install_requires=[
        'setuptools',
        'scipy',
        'numpy',
        'matplotlib',
    ],
    long_description="https://github.com/urigoren/conjugate_prior/blob/master/README.md",
    long_description_content_type="text/markdown",
    version="0.55",
    description='Bayesian Statistics conjugate prior distributions',
    author='Uri Goren',
    author_email='uri@goren.ml',
    url='https://github.com/urigoren/conjugate_prior',
    keywords=['conjugate', 'bayesian', 'stats', 'statistics', 'bayes', 'distribution', 'probability', 'hypothesis',
              'modelling'],
    classifiers=[],
)

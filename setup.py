from distutils.core import setup
with open("README.md", 'r') as f:
    long_description = f.read()
setup(
    name="conjugate_prior",
    packages=["conjugate_prior"],
    install_requires=[
        'setuptools',
        'scipy',
        'numpy',
        'matplotlib',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.71",
    description='Bayesian Statistics conjugate prior distributions',
    author='Uri Goren',
    author_email='uri@argmax.ml',
    url='https://github.com/urigoren/conjugate_prior',
    keywords=['conjugate', 'bayesian', 'stats', 'statistics', 'bayes', 'distribution', 'probability', 'hypothesis',
              'modelling', 'thompson sampling'],
    classifiers=[],
)

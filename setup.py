from distutils.core import setup
with open("README.md", 'r') as f:
    long_description = f.read()
with open("conjugate_prior/__init__.py", 'r') as f:
    for l in f:
        if l.startswith("__version__"):
            _,version=l.split("=", 1)
            version = version.strip('\'" \r\n\t')
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
    version=version,
    description='Bayesian Statistics conjugate prior distributions',
    author='Uri Goren',
    author_email='conjugate@argmaxml.com',
    url='https://github.com/argmaxml/conjugate_prior',
    keywords=['conjugate', 'bayesian', 'stats', 'statistics', 'bayes', 'distribution', 'probability', 'hypothesis',
              'modelling', 'thompson sampling'],
    classifiers=[],
)

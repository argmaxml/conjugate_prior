#https://packaging.python.org/tutorials/distributing-packages/
rm -rf dist
python setup.py sdist
twine upload dist/*

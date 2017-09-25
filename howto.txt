#https://packaging.python.org/tutorials/distributing-packages/
python setup.py sdist
twine upload dist/*

#https://packaging.python.org/tutorials/distributing-packages/
rm -rf dist
python3 setup.py sdist
python3 -m twine upload dist/*

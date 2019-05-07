ipython nbconvert --to python dataset.ipynb
python dataset.py %*
:repeat
python dataset.py --load %*
goto repeat
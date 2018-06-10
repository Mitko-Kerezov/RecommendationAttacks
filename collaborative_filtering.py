import pandas as pd
from surprise import Reader, Dataset, NMF, SVD, evaluate

reader = Reader(rating_scale=(0.5, 5.0), sep=',')
data = Dataset.load_from_file("ratings.csv", reader)

# svd
algo = SVD()
evaluate(algo, data, measures=['RMSE'])

# nmf
algo = NMF()
evaluate(algo, data, measures=['RMSE'])

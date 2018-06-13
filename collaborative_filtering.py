import pandas as pd
from surprise import Reader, Dataset, NMF, SVD, evaluate

reader = Reader(rating_scale=(0.5, 5.0), sep=',')
data = Dataset.load_from_file("ratings.csv", reader)
trainset = data.build_full_trainset()

# svd
algo = SVD()
algo.train(trainset)

uid = "3"
pred = algo.predict(uid=uid, iid="", r_ui=1287, verbose=True)
print(pred)

# evaluate(algo, data, measures=['RMSE'])

# # nmf
# algo = NMF()
# evaluate(algo, data, measures=['RMSE'])

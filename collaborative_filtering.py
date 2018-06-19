import pandas as pd
from surprise import Reader, Dataset, NMF, SVD, evaluate

reader = Reader(rating_scale=(1.0, 5.0), sep=',')
data = Dataset.load_from_file("ratings.csv", reader)
trainset = data.build_full_trainset()

# svd
algo = SVD()
algo.fit(trainset)

# First user group are drama lovers
users = [(1,200), (201,400), (401,600), (601,800), (801,1000)]

for user_group in users:
    group_avg = 0
    for user in range(user_group[0], user_group[1] + 1):
        # Movie with ID=1 is Titanic
        pred = algo.predict(uid=str(user), iid="1", r_ui=None, verbose=False)
        group_avg += pred.est
    print(group_avg/200)

# evaluate(algo, data, measures=['RMSE'])

# # nmf
# algo = NMF()
# evaluate(algo, data, measures=['RMSE'])

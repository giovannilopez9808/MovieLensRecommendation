from Modules.Dataset import MovieDataset
from Modules.params import get_params
from Modules.Model import Model

params = get_params()
dataset = MovieDataset(
    params,
)
model = Model(
    dataset.userId_len,
    dataset.movieId_len,
)
model.train(
    dataset.train
)

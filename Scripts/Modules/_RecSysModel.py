from torch.nn import (
    Embedding,
    Module,
    Linear,
)
from torch import (
    tensor,
    cat,
)


class _RecSysModel(Module):
    """

    """

    def __init__(
        self,
        n_users: int,
        n_movies: int,
    ) -> None:
        super().__init__()
        # trainable lookup matrix for shallow embedding vectors
        self.user_embed = Embedding(
            n_users,
            32
        )
        self.movie_embed = Embedding(
            n_movies,
            32
        )
        # user, movie embedding concat
        self.out = Linear(
            64,
            1
        )

    def forward(
        self,
        users: tensor,
        movies: tensor,
        ratings: tensor = None,
    ) -> tensor:
        user_embeds = self.user_embed(
            users
        )
        movie_embeds = self.movie_embed(
            movies
        )
        output = cat(
            [
                user_embeds,
                movie_embeds
            ],
            dim=1
        )
        output = self.out(
            output
        )
        return output

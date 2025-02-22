from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from os.path import join
from sklearn import (
    preprocessing,
)
from pandas import (
    DataFrame,
    read_csv,
)
from torch import (
    tensor,
    long,
)


class _MovieDataset:
    """

    """

    def __init__(
        self,
        users: dict,
        movies: dict,
        ratings: dict,
    ) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(
            self,
            item
    ) -> dict:
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]
        return {
            "users": tensor(
                users,
                dtype=long
            ),
            "movies": tensor(
                movies,
                dtype=long
            ),
            "ratings": tensor(
                ratings,
                dtype=long
            ),
        }


class MovieDataset:
    """

    """

    def __init__(
        self,
        params: dict,
    ) -> None:
        self.movieId_len: int = None
        self.userId_len: int = None
        self.ratings_df = self._read_csv(
            "ratings.csv",
            params,
        )
        self.train, self.validation, self.test = self._get_DataLoader(
            self.ratings_df,
        )

    def _read_csv(
        self,
        filename: str,
        params: dict,
    ) -> DataFrame:
        filename = join(
            params["data_path"],
            filename,
        )
        data = read_csv(
            filename,
        )
        return data

    def _preprocessing(
        self,
        data: DataFrame
    ) -> DataFrame:
        user_preprocessing = preprocessing.LabelEncoder()
        movie_preprocessing = preprocessing.LabelEncoder()
        data.userId = user_preprocessing.fit_transform(
            data.userId
        )
        data.movieId = movie_preprocessing.fit_transform(
            data.movieId
        )
        self.userId_len = len(
            user_preprocessing.classes_
        )
        self.movieId_len = len(
            movie_preprocessing.classes_
        )
        return data

    def _get_DataLoader(
        self,
        data: DataFrame,
    ) -> tuple[DataLoader]:
        data = self._preprocessing(
            data,
        )
        train, test = train_test_split(
            data,
            stratify=data.rating.values,
            random_state=1998,
            test_size=0.3,
        )
        train, validation = train_test_split(
            train,
            stratify=train.rating.values,
            random_state=1998,
            test_size=0.3,
        )
        train = _MovieDataset(
            movies=train.movieId.values,
            ratings=train.rating.values,
            users=train.userId.values,
        )
        validation = _MovieDataset(
            movies=validation.movieId.values,
            ratings=validation.rating.values,
            users=validation.userId.values,
        )
        test = _MovieDataset(
            movies=test.movieId.values,
            ratings=test.rating.values,
            users=test.userId.values,
        )
        train = self._create_DataLoader(
            train,
        )
        validation = self._create_DataLoader(
            validation,
        )
        test = self._create_DataLoader(
            test,
        )
        return train, validation, test

    def _create_DataLoader(
        self,
        data: _MovieDataset,
    ) -> DataLoader:
        dataloader = DataLoader(
            data,
            batch_size=4,
            shuffle=True,
        )
        return dataloader

from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR
)
from ._RecSysModel import _RecSysModel
from torch.nn.modules.loss import (
    MSELoss,
    _Loss,
)
from torch.nn import Module
from torch.optim import (
    Optimizer,
    Adam
)
from torch import (
    float32,
    device,
    cuda,
)


class Model:
    def __init__(
        self,
        n_users: int,
        n_movies: int
    ) -> None:
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.loss_func: _Loss = None
        self.device: device = None
        self.model: Module = _RecSysModel(
            n_users,
            n_movies,
        )
        self._get_optimizer()
        self._get_device()
        self._get_loss_function()
        self.model = self._to_gpu(
            self.model
        )

    def _get_device(
        self,
    ) -> str:
        self.device = device(
            'cuda'
            if cuda.is_available() else
            'cpu'
        )

    def _get_optimizer(
        self,
    ) -> tuple[Optimizer, LRScheduler]:
        self.optimizer = Adam(
            self.model.parameters()
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.7,
        )

    def _get_loss_function(
        self,
    ) -> _Loss:
        self.loss_func = MSELoss()

    def _to_gpu(
        self,
        _object: None
    ) -> None:
        return _object.to(self.device)

    def _to_cpu(
        self,
        _object: None
    ) -> None:
        return _object.to("cpu")

    def train(
        self,
        train_loader,
    ) -> list[float32]:
        """

        """
        epochs = 1
        total_loss = 0
        plot_steps = 5000
        step_cnt = 0
        all_losses_list = []
        self.model.train()
        for epoch_i in range(epochs):
            for _, train_data in enumerate(train_loader):
                output = self.model(
                    train_data["users"].to(self.device),
                    train_data["movies"].to(self.device),
                )
                rating = train_data["ratings"]
                rating = rating.to(float32)
                output = output.flatten()
                output = output.to(float32)
                output = self._to_cpu(output)
                loss = self.loss_func(
                    output,
                    rating,
                )
                total_loss = total_loss + loss.sum().item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step_cnt = step_cnt + len(train_data["users"])
                if step_cnt % plot_steps == 0:
                    train_data_len = len(train_data["users"])
                    avg_loss = total_loss / (train_data_len * plot_steps)
                    print("="*30)
                    print(f"epochs:\t{epoch_i}")
                    print(f"loss:\t{avg_loss}")
                    all_losses_list.append(
                        avg_loss
                    )
                    total_loss = 0
        return all_losses_list

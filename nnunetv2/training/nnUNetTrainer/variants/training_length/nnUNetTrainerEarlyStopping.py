import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from typing import Optional, Dict, cast


class EarlyStopping:
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    Based on the PyTorch Ignite implementation.

    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """

    _state_dict_all_req_keys = (
        "counter",
        "best_score",
    )

    def __init__(
            self,
            patience: int,
            logger,
            min_delta: float = 0.0,
            cumulative_delta: bool = False,
    ):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.logger = logger

    def stop_training(self, new_score: float) -> bool:
        # First epoch:
        if self.best_score is None:
            self.best_score = new_score
            return False
        # If new score is not good enough:
        elif new_score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and new_score > self.best_score:
                self.best_score = new_score
            self.counter += 1
            self.logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            # Checking patience:
            if self.counter >= self.patience:
                self.logger.info("EarlyStopping: Stop training")
                return True
            else:
                return False
        # If it is good enough:
        else:
            self.best_score = new_score
            self.counter = 0
            return False

    def state_dict(self) -> Dict[str, float]:
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return {"counter": self.counter, "best_score": cast(float, self.best_score)}

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        """Method replace internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.patience = 100
        self.min_delta = 0
        self.cumulative_delta = False
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            logger=self.logger,
            min_delta=self.min_delta,
            cumulative_delta=self.cumulative_delta
        )

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self.early_stopping.stop_training(new_score=self.logger.my_fantastic_logging['mean_fg_dice'][-1]):
                break

        self.on_train_end()

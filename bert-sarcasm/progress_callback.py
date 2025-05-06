from transformers import TrainerCallback
from tqdm.auto import tqdm

class TQDMProgressBar(TrainerCallback):
    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(total=state.max_steps, desc=f"Epoch {int(state.epoch + 1)}", leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.epoch_bar:
            self.epoch_bar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_bar:
            self.epoch_bar.close()

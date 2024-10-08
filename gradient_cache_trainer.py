from transformers import Trainer
from peft import PeftModel
from grad_cache import GradCache
from grad_cache.loss import SimpleContrastiveLoss
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach
from torch import nn
import torch.functional as F
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_peft_available, is_accelerate_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.nn import functional
from torch.utils.data import Sampler, SequentialSampler

def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)


class ContrastiveMSELoss:
    def __init__(self, n_hard_negatives: int = 0):
        self.target_per_qry = n_hard_negatives + 1

    def __call__(self, x: torch.Tensor, y: torch.Tensor, target: torch.Tensor = None, reduction: str = 'mean'):
        logits = torch.matmul(x, y.T)
        loss = functional.mse_loss(logits, target, reduce=False)
        negative_loss_scale = (~target.bool()).sum() / target.numel()
        loss_scale_matrix = torch.where(target == 1, target, negative_loss_scale)
        return (loss * loss_scale_matrix).mean()


def get_repr(v):
    state = v.last_hidden_state[:,-1]
    return functional.normalize(state, 2, -1)

def get_repr_mean(v):
    state = v.last_hidden_state.mean(dim=-2)
    return functional.normalize(state, 2, -1)


class GradientCacheTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        self.loss_fn = ContrastiveMSELoss()
        super().__init__(model, *args, **kwargs)
        self.repr_func = get_repr
        self.gradient_cache = GradCache(
            models=[model, model],
            chunk_sizes=2,
            loss_fn=self.loss_fn,
            # loss_fn=self.loss_fn,
            get_rep_fn=self.repr_func,
            fp16=True,
            scaler="placeholder" # Scaler is not initilaized on the trainer yet, so we assign it in the training step
        )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.gradient_cache.scaler = self.optimizer.scaler

        loss = self.gradient_cache(inputs["texts_a"], inputs["texts_b"], target=inputs["labels"].to(torch.float), reduction="mean")
        print("Loss", loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                embs_a = model(**inputs["texts_a"], use_cache=False)
                embs_b = model(**inputs["texts_b"], use_cache=False)
                repr_a, repr_b = self.repr_func(embs_a), self.repr_func(embs_b)
                loss = self.loss_fn(repr_a, repr_b, target=inputs["labels"])

        return (loss, torch.cat([repr_a, repr_b]), inputs["labels"])

    def _get_train_sampler(self) -> Optional[Sampler]:
        # We don't want to randomize the order...
        return SequentialSampler(self.train_dataset)


class LLM2VecGradientCacheTrainer(GradientCacheTrainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.repr_func = get_repr_mean
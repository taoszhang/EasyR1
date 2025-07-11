import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], 
                            # non_tensor_keys: List[str] = None,
                            cut_left: bool = True) -> Dict[str, torch.Tensor]: # 默认cut_left，那就应该是模型左padding?
        """Cut tensors to their effective length based on attention mask."""
        """A new function has been added, and the part exceeding the maximum length is truncated."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        # effective_len = min(self.config.max_start_length, effective_len)
        result = tensor_dict.copy()
        
        # for key in keys:
        #     if cut_left:
        #         result[key] = tensor_dict[key][:, -effective_len:]
        #     else:
        #         result[key] = tensor_dict[key][:, :effective_len]
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]

        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        valid_tensors = [t for t in tensors if t.size(1) > 0]
        if len(valid_tensors) == 0:
            # 所有输入都为空，返回一个 shape 为 (B, 0) 的空 tensor
            # 假设原始 tensors 至少一个非空，可以用 tensors[0].size(0)
            batch_size = tensors[0].size(0)
            return torch.empty((batch_size, 0), dtype=tensors[0].dtype, device=tensors[0].device)
        
        concatenated = torch.cat(valid_tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str
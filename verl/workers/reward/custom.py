# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer
import numpy as np
from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, infoseek_compute_score


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str):
        self.tokenizer = tokenizer
        if compute_score == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif compute_score == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif compute_score == "infoseek":
            self.compute_score: Callable[[str, str], RewardScore] = infoseek_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            # valid_response_length = response_mask.sum()
            
            # 为了适应responses info mask 而做出的更改。# 找出所有非零的位置
            nonzero_indices = (response_mask != 0).nonzero(as_tuple=False)
            if nonzero_indices.numel() == 0:
                # 如果没有非零元素，valid_response_length是0（tensor类型）
                valid_response_length = torch.tensor(0, device=response_mask.device, dtype=torch.long)
            else:
                # 否则取最后一个非零位置 + 1（保持是tensor）
                valid_response_length = torch.tensor(nonzero_indices[-1].item() + 1, device=response_mask.device, dtype=torch.long)

            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            if 'problem_type' in data_item.non_tensor_batch:
                problem_type = data_item.non_tensor_batch.get("problem_type", None)
                score = self.compute_score(response_str, ground_truth, problem_type)
            else:
                score = self.compute_score(response_str, ground_truth)

            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        # metrics for actions
        # if 'turns_stats' in data.meta_info:
        #     reward_metrics['env/number_of_actions/mean'] = float(np.array(data.meta_info['turns_stats'], dtype=np.int16).mean())
        #     reward_metrics['env/number_of_actions/max'] = float(np.array(data.meta_info['turns_stats'], dtype=np.int16).max())
        #     reward_metrics['env/number_of_actions/min'] = float(np.array(data.meta_info['turns_stats'], dtype=np.int16).min())
        # if 'active_mask' in data.meta_info:
        #     reward_metrics['env/finish_ratio'] = 1 - float(np.array(data.meta_info['active_mask'], dtype=np.int16).mean())
        # if 'valid_action_stats' in data.meta_info:
        #     reward_metrics['env/number_of_valid_action'] = float(np.array(data.meta_info['valid_action_stats'], dtype=np.int16).mean())
        #     reward_metrics['env/ratio_of_valid_action'] = float((np.array(data.meta_info['valid_action_stats'], dtype=np.int16) / np.array(data.meta_info['turns_stats'], dtype=np.int16)).mean())
        # if 'valid_search_stats' in data.meta_info:
        #     reward_metrics['env/number_of_valid_search'] = float(np.array(data.meta_info['valid_search_stats'], dtype=np.int16).mean())

        return reward_tensor, reward_metrics

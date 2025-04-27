import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl.protocol import DataProto
from verl.workers.fsdp_workers import FSDPWorker
from verl.utils.tracking import Tracking
import shutil
import requests
import numpy as np
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    max_end_length: int
    # logging: dict
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    rollout_n: int = 1

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.rollout_n = 1

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # actions, _ = self.env.postprocess_predictions(responses_str)
            # responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            # print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding      
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        update_rolling_state = {}
        update_rolling_state['input_ids'] = new_input_ids[:, -max_len:]
        update_rolling_state['attention_mask'] = new_attention_mask[:, -max_len:]
        update_rolling_state['position_ids'] = new_position_ids[:, -max_len:]
        # raw_prompt_ids，是input_ids去掉pad token，去掉image token
        
        update_rolling_state['raw_prompt_ids'] = rollings.non_tensor_batch['raw_prompt_ids']
        update_rolling_state['multi_modal_data'] = rollings.non_tensor_batch['multi_modal_data']
        update_rolling_state['multi_modal_inputs'] = rollings.non_tensor_batch['multi_modal_inputs']

        # NOTE: 这里对raw_prompt_ids进行裁剪，计算image token的长度
        for i in range(rollings.batch.shape[0]):
            cur_res_np = cur_responses[i].cpu().numpy()
            cur_obs_np = next_obs_ids[i].cpu().numpy()
            cur_res_unpad = cur_res_np[cur_res_np != self.tokenizer.pad_token_id]
            cur_obs_unpad = cur_obs_np[cur_obs_np != self.tokenizer.pad_token_id]
            
            image_len = (update_rolling_state['input_ids'][i] == self.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum().item()

            update_rolling_state['raw_prompt_ids'][i] = np.concatenate([
                rollings.non_tensor_batch['raw_prompt_ids'][i],
                cur_res_unpad,
                cur_obs_unpad
            ], axis=0)[-max_len+image_len-1:]

        return DataProto.from_single_dict(update_rolling_state, meta_info=rollings.meta_info)        

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id # 标记的是pad id的位置
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        # 这里的pad功能，maybe可以用pad_dataproto_to_divisor函数实现，模仿easy-r1中的实现。
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        # 添加对于non_tensor_batch的处理, np.ndarray数据类型
        for k,v in active_batch.non_tensor_batch.items():
            pad_sequence = np.tile(v[0:1], (padding_size, *[1] * (len(v.shape) - 1)))
            padded_batch[k] = np.concatenate([v, pad_sequence], axis=0)
        # 比如说pad之前是12，现在变成16，后四个样本与第一个样本相同
        padded_active_batch = DataProto.from_single_dict(padded_batch, meta_info=active_batch.meta_info) 
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def active_rollings(self, rollings: DataProto, active_mask: torch.Tensor) -> DataProto:
        rollings_active = {}
        for k, v in rollings.batch.items():
            rollings_active[k] = v[active_mask]
        for k, v in rollings.non_tensor_batch.items():
            rollings_active[k] = v[active_mask]
        return DataProto.from_single_dict(rollings_active, meta_info=rollings.meta_info)

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_prompt_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        if not self.is_validation:
            original_left_side['input_ids'] = original_left_side['input_ids'].repeat_interleave(repeats=self.config.rollout_n, dim=0)
            original_right_side['responses'] = original_right_side['responses'].repeat_interleave(repeats=self.config.rollout_n, dim=0)
            original_right_side['responses_with_info_mask'] = original_right_side['responses_with_info_mask'].repeat_interleave(repeats=self.config.rollout_n, dim=0)

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0] * self.config.rollout_n, dtype=torch.int)
        # valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0] * self.config.rollout_n, dtype=torch.int)
        # valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0] * self.config.rollout_n, dtype=torch.int)
        active_num_list = [active_mask.sum().item() * self.config.rollout_n]
        rollings = gen_batch
        # breakpoint()
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # 根据注意力掩码进行裁剪, 去掉多余的padding部分的内容
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
            )
            rollings_active = self.active_rollings(rollings, active_mask)

            # gen_output = self._generate_with_gpu_padding(rollings_active)
            gen_batch, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_batch.meta_info.update({'no_sleep': False})
            gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)
            # breakpoint()
            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])  # 取response</search>或</answer>前的内容，相当于截断，截断</search>优先

            if step == 0 and not self.is_validation:
                rollings = rollings.repeat(repeat_times=self.config.rollout_n, interleave=True)
                active_mask = active_mask.repeat_interleave(repeats=self.config.rollout_n, dim=0)
                # NOTE：这个方法应该是控制训练和测试的时候，每次应该rollout几次的关键！！！
                rollings.meta_info['n'] = 1

            # 这里会将rollings和rollings_active对齐
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations 这里分为三种类型，search, answer, invalid，如果是invalid，则接上一段prompt然后再次执行生成，只有answer对应的内容，修改为done
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )            

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            # turns_stats[curr_active_mask] += 1
            # valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            # valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states  
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side( 
                original_right_side,
                responses_ids,
                next_obs_ids
            ) 

        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = self.active_rollings(rollings, active_mask)

            gen_batch, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_batch.meta_info.update({'no_sleep': False})
            gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)
            # breakpoint()
            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            # valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            # valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

        # original_left side就是最开始的输出，original_right_side是在每次只取search或者是answer之前的内容，拼接得到的输出
        # meta_info['turns_stats'] = turns_stats.tolist()
        # meta_info['active_mask'] = active_mask.tolist()
        # meta_info['valid_action_stats'] = valid_action_stats.tolist()
        # meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            # meta_info: Dict) -> Tuple[Dict, Dict]:
                            meta_info: Dict) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # 对于right_side['responses']，如果超过max_end_length，则进行裁剪，否则pad到max_end_length
        assert final_output['responses_with_info_mask'].shape[1] == final_output['responses'].shape[1]
        if final_output['responses'].shape[1] > self.config.max_end_length:
            final_output['responses'] = final_output['responses'][:, :self.config.max_end_length]
        elif final_output['responses'].shape[1] < self.config.max_end_length:
            padded_responses = torch.full(
                (final_output['responses'].shape[0], self.config.max_end_length-final_output['responses'].shape[1]),
                self.tokenizer.pad_token_id,
                dtype=final_output['responses'].dtype,
                device=final_output['responses'].device
            )
            final_output['responses'] = torch.cat([
                final_output['responses'],
                padded_responses
            ], dim=1)
            final_output['responses_with_info_mask'] = torch.cat([
                final_output['responses_with_info_mask'],
                padded_responses
            ], dim=1)
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            # right_side['responses']
            final_output['responses'],
        ], dim=1)

        # Create attention mask and position ids
        # final_output['attention_mask'] = torch.cat([
        #     self.tensor_fn.create_attention_mask(left_side['input_ids']),
        #     self.tensor_fn.create_attention_mask(final_output['responses'])
        # ], dim=1)
        # final_output['response_mask'] = self.tensor_fn.create_attention_mask(
        #     final_output['responses']
        # )
        # breakpoint()
        # responses_with_info_mask是对于ret info进行mask的responses
        # 这里给的是对检索信息进行mask的responses_mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        final_output['response_mask'] = self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_single_dict(final_output, meta_info=meta_info)
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

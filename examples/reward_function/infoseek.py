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

import re
from mathruler.grader import grade_answer
from word2number import w2n
import string
from typing import Any, Dict, Generator, List, Tuple, Union


def format_reward(predict_str: str) -> float:
    # 定义正则单元
    think_pattern = r"<think>.*?</think>"
    text_search_pattern = r"<text_search>.*?</text_search>"
    image_search_pattern = r"<image_search>.*?</image_search>"
    infomation_pattern = r"<information>.*?</information>"
    answer_pattern = r"<answer>.*?</answer>"
    
    # 编译单元表达式（用于匹配顺序）
    unit_pattern = f"({think_pattern})|({text_search_pattern})|({image_search_pattern})|({infomation_pattern})|({answer_pattern})"
    unit_re = re.compile(unit_pattern, re.DOTALL)

    # 匹配整个输入
    matches = list(unit_re.finditer(predict_str))

    # 如果没有匹配项，或拼接起来和原文不一致（中间出现非法内容），直接返回 0
    reconstructed = ''.join(m.group(0) for m in matches)
    if re.sub(r'\s+', '', reconstructed) != re.sub(r'\s+', '', predict_str):
        return 0.0

    # 提取结构标签序列
    sequence = []
    for m in matches:
        if m.group(1):
            sequence.append('T')
        elif m.group(2):
            sequence.append('TS')
        elif m.group(3):
            sequence.append('IS')
        elif m.group(4):
            sequence.append('I')
        elif m.group(5):
            sequence.append('A')

    # 规则检测
    if len(sequence) < 2 or sequence[0] != 'T' or sequence[-1] != 'A':
        return 0.0


    # 这一个format应该是害了很多次训练，比如compare里面，如果是两次图像检索，就会导致format为0，所以就是给了错误的reward
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            return 0.0  # 连续重复
        
    # for i in range(1, len(sequence)):
    #     if sequence[i] == sequence[i-1] and sequence[i] in {'T', 'A'}:
    #         return 0.0  # think 或 answer 连续重复

    # 检查 TS 或 IS 后必须跟 I
    for i in range(len(sequence)):
        if sequence[i] in {'TS', 'IS'}:
            if sequence[i+1] != 'I':
                return 0.0  # 顺序不符合要求

    if sequence.count('A') != 1:
        return 0.0  # 不止一个 answer
    return 1.0

def text_search_reward(predict_str: str) -> float:
    predict_str = predict_str.replace("<text_search> and </text_search>", "") # 去掉invid部分的prompt
    pattern = re.compile(r"<text_search>.*?</text_search>", re.DOTALL)
    matches = re.findall(pattern, predict_str)
    return float(len(matches))

def image_search_reward(predict_str: str) -> float:
    predict_str = predict_str.replace("<image_search> and </image_search>", "")
    predict_str = predict_str.replace("<image_search> image n </image_search>", "")
    predict_str = predict_str.replace("For example, to view the first image, I should generate <image_search> image 1 </image_search>.", "")
    pattern = re.compile(r"<image_search>.*?</image_search>", re.DOTALL)
    matches = re.findall(pattern, predict_str)
    return float(len(matches))

def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))

def infoseek_string_accuracy_reward(predict_str: str, ground_truth: list) -> float:
    try:
        # ground_truth = ground_truth.strip()
        # content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        # given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        content_match = re.findall(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match[-1].strip() if content_match else predict_str.strip()
        for answer in ground_truth:
            # if grade_answer(given_answer, answer):
            if grade_answer(normalize_answer(given_answer), normalize_answer(answer)):
                return 1.0

    except Exception:
        pass

    return 0.0

def replace_number_words(text):
    # 定义一个正则表达式模式，用于匹配可能的数字词组
    pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                         r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                         r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                         r'eighty|ninety|hundred|thousand|million|billion|trillion|'
                         r'point|and|[-\s])+\b', re.IGNORECASE)

    def convert(match):
        try:
            # 尝试将匹配的数字词组转换为数字
            return str(w2n.word_to_num(match.group()))
        except ValueError:
            # 如果转换失败，返回原始匹配内容
            return match.group()

    # 使用正则表达式替换文本中的数字词组
    return pattern.sub(convert, text)

def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string and return their indices.

    Args:
        s: The input string to search.
        c: The character to search for.

    Yields:
        int: The index of the next occurrence of the character.
    """
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)

def clean_str_range(text: str) -> str:
    """Clean range expression in a string (e.g., '9-10' --> '9 - 10').

    Args:
        text: The input string containing the range expression.

    Returns:
        str: The cleaned string with proper spacing around the hyphen.
    """
    # try:
    idx_list = list(find_all(text, '-'))
    idx_replace = [
        idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()
    ]
    new_str = ''.join(
        ' - ' if idx in idx_replace else s for idx, s in enumerate(text)
    )
    return new_str

def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parses numerical answer string into numbers (a single number or a range).

    1) Clean the string and extract numbers;
    2) if there are 2 numbers, return a range as [minimum value, maximum value]
        else if there is 1 number, return a single number
        else return [0, 0]

    Args:
        string_number: A string representing a numerical answer.

    Returns:
        A single digit or a list with 2 numbers.
    """
    # Clean string
    string_number = replace_number_words(string_number)
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [
        n.replace(',', '').strip('.') for n in numerical_numbers_tmp
    ]
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
            numerical_numbers.append(float(n))
        else:
            numerical_numbers.append(float(n))

    # Use the first 2 numbers
    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val = numerical_numbers[0]
        second_val = numerical_numbers[1]
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]

def infoseek_numerical_accuracy_reward(predict_str: str, ground_truth: list) -> float:
    try:
        # ground_truth = ground_truth.strip()
        # content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        # given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        content_match = re.findall(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match[-1].strip() if content_match else predict_str.strip()
        # 将数字词组转换为数字
        given_answer = replace_number_words(given_answer)
        ground_truth = [
            replace_number_words(answer) for answer in ground_truth
        ]
        min_value = min([float(answer) for answer in ground_truth])
        max_value = max([float(answer) for answer in ground_truth])
        # given_answer = float(given_answer)
        given_answer = process_numerical_answer(given_answer)
        if min_value <= given_answer <= max_value:
            return 1.0

    except Exception:
        pass

    return 0.0

def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0
    
# def compute_score(
#     reward_inputs: List[Dict[str, Any]],
#     max_response_length: int,
#     overlong_buffer_length: int,
#     overlong_penalty_factor: float,
# ) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for dapo reward function.")

#     scores = []
#     for reward_input in reward_inputs:
#         accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
#         overlong_score = soft_overlong_punishment(
#             reward_input["response_length"], max_response_length, overlong_buffer_length
#         )
#         scores.append(
#             {
#                 "overall": accuracy_score + overlong_score * overlong_penalty_factor,
#                 "accuracy": accuracy_score,
#                 "overlong": overlong_score,
#                 "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
#             }
#         )

#     return scores


def compute_score(predicts, ground_truths) -> Dict[str, float]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        problem_type = ground_truth.get("problem_type", None)
        if problem_type is None:
            raise ValueError("problem_type is not provided in ground_truth.")
        answer_eval = ground_truth.get("answer_eval", None)
        if answer_eval is None:
            raise ValueError("answer_eval is not provided in ground_truth.")
        
        format = format_reward(predict)
        text_search_time = text_search_reward(predict)
        image_search_time = image_search_reward(predict)
        if problem_type.lower() == "string" or problem_type.lower() == "time":
            accuracy = infoseek_string_accuracy_reward(predict, answer_eval)
        elif problem_type.lower() == "numerical":
            accuracy = infoseek_numerical_accuracy_reward(predict, answer_eval)
        else:
            raise NotImplementedError(f"Problem type {problem_type} is not supported.")
        scores.append({
            "accuracy": accuracy,
            "format": format,
            "text_search_times": text_search_time,
            "image_search_times": image_search_time,
            "overall": (
                # 0.25 * accuracy * format * (image_search_time + text_search_time) + accuracy + format
                0.25 * (text_search_time + image_search_time) + accuracy + format
            )
        })
    return scores
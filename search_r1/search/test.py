import re
import requests
from typing import List

def batch_search(queries, topk=2) -> str:

    results = _batch_search(queries, topk)['result']
    
    return [_passages2string(result) for result in results]

def _batch_search(queries, topk=2):
    
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": True
    }
    
    return requests.post('http://127.0.0.1:8000/retrieve', json=payload).json()

def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference

def postprocess_predictions(predictions):
    actions = []
    contents = []
            
    for prediction in predictions:
        if isinstance(prediction, str): # for llm output
            pattern = r'<(text_search|answer|image_search)>(.*?)</\1>'
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


# def _postprocess_responses(responses):
#     """Process responses to stop at search operation or answer operation."""

#     responses_str = [ resp.split('</image_search>')[0] + '</image_search>'
#                 if '</image_search>' in resp
#                 else resp.split('</text_search>')[0] + '</text_search>'
#                 if '</text_search>' in resp 
#                 else resp.split('</answer>')[0] + '</answer>'
#                 if '</answer>' in resp 
#                 else resp
#                 for resp in responses]

#     return responses_str
def _postprocess_responses(responses):
    """Process responses to stop at the earliest operation tag (text_search, image_search, or answer)."""
    
    
    stop_tags = ['</text_search>', '</image_search>', '</answer>']
    
    def truncate_response(resp: str) -> str:
        # 找到所有终止标签在字符串中的位置
        tag_positions = [(resp.find(tag), tag) for tag in stop_tags if tag in resp]
        if not tag_positions:
            return resp  # 没有终止标签就保留原始内容
        # 找到最早出现的终止标签及其位置
        earliest_pos, tag = min(tag_positions, key=lambda x: x[0])
        return resp[:earliest_pos + len(tag)]  # 截断到终止标签结束
    
    responses_str = [truncate_response(resp) for resp in responses]
    return responses_str



if __name__ == "__main__":
    response = "<text_search> Northern slimy salamander</text_search>"
    responses_str = _postprocess_responses([response])
    print(f"Processed Responses: {responses_str}")
    actions, contents = postprocess_predictions(responses_str)

    print(f"Actions: {actions}")
    print(f"Contents: {contents}")
    results = batch_search(contents, topk=3)
    print(f"Search Results: {results}")
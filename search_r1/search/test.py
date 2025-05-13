import re
import requests
from typing import List

def batch_search(queries: List[str] = None) -> str:

    results = _batch_search(queries)['result']
    
    return [_passages2string(result) for result in results]

def _batch_search(queries):
    
    payload = {
        "queries": queries,
        "topk": 2,
        "return_scores": True
    }
    
    return requests.post('http://127.0.0.1:8000/retrieve', json=payload).json()

def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        # import pdb; pdb.set_trace()
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference

def postprocess_predictions(predictions):
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


if __name__ == "__main__":

    query = "<search query='Madhabkunda waterfall country'> </search>"
    actions, contents = postprocess_predictions([query])
    result = batch_search(contents)
    print(result)

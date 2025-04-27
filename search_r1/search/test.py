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

if __name__ == "__main__":

    query = "What is the location of Kungsholm Church?"
    result = batch_search([query])
    print(result)

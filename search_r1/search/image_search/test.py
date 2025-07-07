import requests

res = requests.post("http://0.0.0.0:8888/image_search", json={
    "image_paths": ["/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_test_images/oven_05156049.jpg", "/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_val_images/oven_05020405.jpg"],
    "topk": 5
})
print(res.json())

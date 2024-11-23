import json
ranker = {
    'rank1': [10, 'moon', "https://via.placeholder.com/200x150?text=1"],
    'rank2': [5, 'sung', 'img']
}
with open("data/ranker.json", "w", encoding="utf-8") as f:
    json.dump(ranker, f, ensure_ascii=False, indent=4)
    
with open("data/ranker.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

print(loaded_data)
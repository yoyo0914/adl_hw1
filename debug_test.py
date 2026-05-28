import json

def debug_test_data():
    """檢查測試數據格式並對比訓練數據"""
    
    # 載入數據
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    with open('context.json', 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    if isinstance(context_data, list):
        contexts = {str(i): content for i, content in enumerate(context_data)}
    else:
        contexts = context_data
    
    print("=== 比較訓練和測試數據格式 ===")
    
    # 檢查訓練數據格式
    print("\n--- 訓練數據樣本 ---")
    train_sample = train_data[0]
    print(f"訓練數據keys: {train_sample.keys()}")
    print(f"問題: {train_sample['question']}")
    print(f"段落IDs: {train_sample['paragraphs']}")
    if 'relevant' in train_sample:
        print(f"相關段落: {train_sample['relevant']}")
    if 'answer' in train_sample:
        print(f"答案: {train_sample['answer']}")
    
    # 檢查測試數據格式
    print("\n--- 測試數據樣本 ---")
    test_sample = test_data[0]
    print(f"測試數據keys: {test_sample.keys()}")
    print(f"問題: {test_sample['question']}")
    print(f"段落IDs: {test_sample['paragraphs']}")
    
    # 檢查段落內容是否匹配問題
    print("\n=== 檢查前3個測試樣本的段落相關性 ===")
    for i in range(3):
        item = test_data[i]
        print(f"\n--- 測試樣本 {i+1} ---")
        print(f"問題: {item['question']}")
        
        for j, pid in enumerate(item['paragraphs']):
            if str(pid) in contexts:
                paragraph = contexts[str(pid)]
                print(f"段落{j} (ID:{pid}): {paragraph[:100]}...")
            else:
                print(f"段落{j} (ID:{pid}): 找不到內容")
        print("-" * 50)

if __name__ == "__main__":
    debug_test_data()
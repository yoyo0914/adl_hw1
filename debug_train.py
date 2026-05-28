import json

def debug_training_data():
    """檢查訓練數據的答案位置計算是否正確"""
    
    # 載入數據
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open('context.json', 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    if isinstance(context_data, list):
        contexts = {str(i): content for i, content in enumerate(context_data)}
    else:
        contexts = context_data
    
    print("=== 檢查前5個訓練樣本的答案位置 ===")
    
    for i in range(5):
        item = train_data[i]
        print(f"\n--- 樣本 {i+1} ---")
        print(f"問題: {item['question']}")
        print(f"原始答案: '{item['answer']['text']}'")
        print(f"相關段落ID: {item['relevant']}")
        
        # 重現qa.py的段落合併邏輯
        merged_paragraphs = []
        for pid in item['paragraphs']:
            if str(pid) in contexts:
                merged_paragraphs.append(contexts[str(pid)].strip())
            else:
                merged_paragraphs.append("")
        
        merged_context = " [SEP] ".join(merged_paragraphs)
        
        # 檢查relevant段落內容
        relevant_paragraph = contexts.get(str(item['relevant']), "")
        print(f"相關段落前100字: {relevant_paragraph[:100]}...")
        
        # 檢查答案在原段落中的位置
        original_answer_text = item['answer']['text']
        original_answer_start = item['answer']['start']
        
        print(f"原始段落中的答案: '{relevant_paragraph[original_answer_start:original_answer_start+len(original_answer_text)]}'")
        
        # 檢查答案在合併文本中的位置
        new_answer_start = merged_context.find(original_answer_text)
        if new_answer_start != -1:
            extracted_answer = merged_context[new_answer_start:new_answer_start+len(original_answer_text)]
            print(f"✅ 合併文本中找到答案: '{extracted_answer}' at position {new_answer_start}")
        else:
            print(f"❌ 合併文本中找不到答案: '{original_answer_text}'")
            print(f"合併文本前200字: {merged_context[:200]}...")
        
        print(f"合併文本長度: {len(merged_context)}")
        print(f"SEP數量: {merged_context.count('[SEP]')}")
        print("-" * 80)

if __name__ == "__main__":
    debug_training_data()
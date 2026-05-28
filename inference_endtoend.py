import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm

def load_model():
    """載入End-to-End QA模型"""
    print("載入End-to-End QA模型...")
    qa_tokenizer = AutoTokenizer.from_pretrained('./end_to_end_model', use_fast=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained('./end_to_end_model')
    qa_model.eval()
    return qa_tokenizer, qa_model

def load_data():
    """載入測試數據和段落內容"""
    print("載入測試數據...")
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("載入段落內容...")
    with open('context.json', 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    if isinstance(context_data, list):
        contexts = {str(i): content for i, content in enumerate(context_data)}
    else:
        contexts = context_data
    
    return test_data, contexts

def extract_answer_from_single_segment(question, context, tokenizer, model):
    """從單個段落中抽取答案"""
    if not context.strip():
        return "", 0.0
    
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
    
    # 簡單的argmax策略
    best_start = torch.argmax(start_logits).item()
    best_end = torch.argmax(end_logits).item()
    
    if best_end < best_start:
        best_start, best_end = best_end, best_start
    
    # 計算置信度 (start + end logits)
    confidence = start_logits[best_start] + end_logits[best_end]
    
    # 提取答案
    if (best_start < len(offset_mapping) and best_end < len(offset_mapping) and 
        offset_mapping[best_start][0] is not None and offset_mapping[best_end][1] is not None):
        start_char = offset_mapping[best_start][0]
        end_char = offset_mapping[best_end][1]
        answer = context[start_char:end_char].strip()
        
        # 過濾明顯錯誤的答案
        if answer and answer not in ['[SEP]', '[CLS]', '[PAD]', '[UNK]'] and len(answer) > 0:
            return answer, confidence.item()
    
    return "", confidence.item()

def extract_answer_segment_wise(question, paragraph_ids, contexts, tokenizer, model):
    """分別從每個段落中抽取答案，選擇最佳的"""
    candidates = []
    
    # 策略1: 分別嘗試每個段落
    for i, pid in enumerate(paragraph_ids):
        if str(pid) in contexts:
            paragraph_content = contexts[str(pid)].strip()
            if paragraph_content:
                answer, confidence = extract_answer_from_single_segment(
                    question, paragraph_content, tokenizer, model
                )
                if answer:
                    candidates.append({
                        'answer': answer,
                        'confidence': confidence,
                        'paragraph_id': pid,
                        'paragraph_index': i
                    })
    
    # 策略2: 嘗試前兩個段落的組合
    if len(paragraph_ids) >= 2:
        combined_paragraphs = []
        for pid in paragraph_ids[:2]:
            if str(pid) in contexts:
                combined_paragraphs.append(contexts[str(pid)].strip())
        
        if len(combined_paragraphs) == 2:
            combined_context = " [SEP] ".join(combined_paragraphs)
            answer, confidence = extract_answer_from_single_segment(
                question, combined_context, tokenizer, model
            )
            if answer:
                candidates.append({
                    'answer': answer,
                    'confidence': confidence,
                    'paragraph_id': 'combined_first_two',
                    'paragraph_index': -1
                })
    
    # 策略3: 完整的4段落組合（原始方法）
    merged_paragraphs = []
    for pid in paragraph_ids:
        if str(pid) in contexts:
            merged_paragraphs.append(contexts[str(pid)].strip())
        else:
            merged_paragraphs.append("")
    
    if merged_paragraphs:
        full_context = " [SEP] ".join(merged_paragraphs)
        answer, confidence = extract_answer_from_single_segment(
            question, full_context, tokenizer, model
        )
        if answer:
            candidates.append({
                'answer': answer,
                'confidence': confidence,
                'paragraph_id': 'full_context',
                'paragraph_index': -2
            })
    
    # 選擇最佳候選
    if candidates:
        # 按置信度排序
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        return best_candidate['answer'], best_candidate['confidence']
    
    return "", 0.0

def run_inference():
    """執行分段式推理"""
    qa_tokenizer, qa_model = load_model()
    test_data, contexts = load_data()
    
    print(f"開始處理 {len(test_data)} 個測試樣本（分段式End-to-End模式）...")
    
    results = []
    
    for i, item in enumerate(tqdm(test_data, desc="分段式推理中")):
        try:
            answer, confidence = extract_answer_segment_wise(
                item['question'], item['paragraphs'], contexts, qa_tokenizer, qa_model
            )
            
            results.append({
                'id': item['id'],
                'answer': answer
            })
            
            # 前5個樣本的詳細debug
            if i < 5:
                print(f"\n=== 分段式案例 {i+1} ===")
                print(f"問題: {item['question']}")
                print(f"預測答案: '{answer}'")
                print(f"置信度: {confidence:.3f}")
                print("-" * 50)
                
        except Exception as e:
            print(f"處理第 {i} 個樣本時出錯: {e}")
            results.append({
                'id': item['id'],
                'answer': ""
            })
    
    return results

def save_submission(results, filename='submission_segment_wise.csv'):
    """保存提交檔案"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"結果已保存至 {filename}")
    
    empty_answers = sum(1 for r in results if not r['answer'].strip())
    print(f"空答案數量: {empty_answers}/{len(results)} ({empty_answers/len(results)*100:.2f}%)")
    
    answer_lengths = [len(r['answer']) for r in results if r['answer'].strip()]
    if answer_lengths:
        print(f"平均答案長度: {sum(answer_lengths)/len(answer_lengths):.1f}")

def main():
    results = run_inference()
    save_submission(results)
    print("分段式End-to-End推理完成！")

if __name__ == "__main__":
    main()
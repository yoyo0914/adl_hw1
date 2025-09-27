import json
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
from tqdm import tqdm

def clean_answer(text):
    """清理答案文本"""
    if not text:
        return ""
    return text.strip()

def load_models():
    """載入兩階段模型"""
    print("載入段落選擇模型...")
    paragraph_tokenizer = AutoTokenizer.from_pretrained('./paragraph_selection_model_922')
    paragraph_model = AutoModelForMultipleChoice.from_pretrained('./paragraph_selection_model_922')
    paragraph_model.eval()
    
    print("載入答案抽取模型...")
    qa_tokenizer = AutoTokenizer.from_pretrained('./answer_extraction_model_final_923')
    qa_model = AutoModelForQuestionAnswering.from_pretrained('./answer_extraction_model_final_923')
    qa_model.eval()
    
    return paragraph_tokenizer, paragraph_model, qa_tokenizer, qa_model

def load_data():
    """載入測試數據和段落內容"""
    print("載入測試數據...")
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("載入段落內容...")
    with open('context.json', 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    # 處理context格式
    if isinstance(context_data, list):
        contexts = {str(i): content for i, content in enumerate(context_data)}
    else:
        contexts = context_data
    
    return test_data, contexts

def select_paragraph(item, contexts, tokenizer, model):
    """階段1: 段落選擇 - 增加debug輸出"""
    question = item['question']
    paragraph_texts = []
    
    # 獲取4個段落的內容
    for pid in item['paragraphs']:
        if str(pid) in contexts:
            paragraph_texts.append(contexts[str(pid)])
        else:
            paragraph_texts.append("")
    
    # 確保有4個段落
    while len(paragraph_texts) < 4:
        paragraph_texts.append("")
    
    # 構造SWAG格式輸入
    first_sentences = [question] * 4
    second_sentences = paragraph_texts
    
    # Tokenize - 調整max_length與訓練一致
    inputs = tokenizer(
        first_sentences,
        second_sentences,
        max_length=512,  # 改回與訓練一致的長度
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # 重新整形為 [1, 4, seq_len]
    for key in inputs:
        inputs[key] = inputs[key].view(1, 4, -1)
    
    # 預測最佳段落
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [4] 
        predicted_idx = torch.argmax(logits).item()
    
    # 返回選中的段落ID和內容，以及所有logits用於debug
    selected_paragraph_id = item['paragraphs'][predicted_idx]
    selected_context = paragraph_texts[predicted_idx]
    
    return selected_paragraph_id, selected_context, predicted_idx, logits.tolist()

def extract_answer(question, context, tokenizer, model, max_answer_length=60):  # 增加max_answer_length
    """階段2: 答案抽取 - 改善版本"""
    if not context.strip():
        return "", 0.0
    
    # Tokenize - 確保與訓練一致
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    # 移除offset_mapping，模型不需要
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    # 預測start和end位置
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
    
    # 智能span選擇 - 增加候選數量
    best_score = -float('inf')
    best_start, best_end = 0, 0
    
    # 增加候選數量
    top_k = min(5, len(start_logits))  
    start_candidates = torch.topk(start_logits, top_k).indices.tolist()
    end_candidates = torch.topk(end_logits, top_k).indices.tolist()
    
    for start_idx in start_candidates:
        for end_idx in end_candidates:
            # 確保合理的span
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_length:
                continue
            if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
                continue
            if (offset_mapping[start_idx][0] is None or 
                offset_mapping[end_idx][1] is None):
                continue
            
            # 計算組合得分
            span_score = start_logits[start_idx] + end_logits[end_idx]
            
            if span_score > best_score:
                best_score = span_score
                best_start, best_end = start_idx, end_idx
    
    # 如果沒找到合適的span，回退到simple argmax
    if best_score == -float('inf'):
        best_start = torch.argmax(start_logits).item()
        best_end = torch.argmax(end_logits).item()
        if best_end < best_start:
            best_start, best_end = best_end, best_start
    
    # 使用offset mapping從原文提取答案
    if (best_start < len(offset_mapping) and best_end < len(offset_mapping) and 
        offset_mapping[best_start][0] is not None and offset_mapping[best_end][1] is not None):
        start_char = offset_mapping[best_start][0]
        end_char = offset_mapping[best_end][1]
        answer = context[start_char:end_char]
    else:
        answer = ""
    
    return clean_answer(answer), best_score

def run_inference():
    """執行完整推理 - 增加詳細debug"""
    # 載入模型和數據
    para_tokenizer, para_model, qa_tokenizer, qa_model = load_models()
    test_data, contexts = load_data()
    
    print(f"開始處理 {len(test_data)} 個測試樣本...")
    
    results = []
    paragraph_selection_stats = []
    qa_confidence_stats = []
    
    for i, item in enumerate(tqdm(test_data, desc="推理中")):
        try:
            # 階段1: 選擇段落
            selected_id, selected_context, selected_idx, para_logits = select_paragraph(
                item, contexts, para_tokenizer, para_model
            )
            
            # 記錄段落選擇的信心度
            para_confidence = max(para_logits) - min(para_logits)  # 最高分與最低分的差距
            paragraph_selection_stats.append(para_confidence)
            
            # 階段2: 抽取答案
            answer, qa_confidence = extract_answer(
                item['question'], selected_context, qa_tokenizer, qa_model
            )
            
            qa_confidence_stats.append(qa_confidence)
            
            results.append({
                'id': item['id'],
                'answer': answer
            })
            
            # 詳細debug前10個案例
            if i < 10:
                print(f"\n=== 案例 {i+1} 詳細分析 ===")
                print(f"問題: {item['question']}")
                print(f"4個段落選項的得分: {[f'{score:.3f}' for score in para_logits]}")
                print(f"選中段落 {selected_idx} (ID: {selected_id})")
                print(f"段落信心度: {para_confidence:.3f}")
                print(f"選中段落內容前100字: {selected_context[:100]}...")
                print(f"QA信心度: {qa_confidence:.3f}")
                print(f"預測答案: '{answer}'")
                print(f"答案長度: {len(answer)}")
                print("-" * 80)
        
        except Exception as e:
            print(f"處理第 {i} 個樣本時出錯: {e}")
            results.append({
                'id': item['id'],
                'answer': ""
            })
    
    # 統計分析
    print(f"\n=== 性能分析 ===")
    print(f"段落選擇平均信心度: {sum(paragraph_selection_stats)/len(paragraph_selection_stats):.3f}")
    print(f"QA平均信心度: {sum(qa_confidence_stats)/len(qa_confidence_stats):.3f}")
    print(f"低信心度段落選擇 (<1.0): {sum(1 for x in paragraph_selection_stats if x < 1.0)}")
    print(f"低信心度QA (<0): {sum(1 for x in qa_confidence_stats if x < 0)}")
    
    return results

def save_submission(results, filename='submission.csv'):
    """保存提交檔案"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"結果已保存至 {filename}")
    print(f"提交格式預覽:")
    print(df.head())
    
    # 檢查答案統計
    empty_answers = sum(1 for r in results if not r['answer'].strip())
    print(f"空答案數量: {empty_answers}/{len(results)} ({empty_answers/len(results)*100:.2f}%)")
    
    # 答案長度分析
    answer_lengths = [len(r['answer']) for r in results if r['answer'].strip()]
    if answer_lengths:
        print(f"平均答案長度: {sum(answer_lengths)/len(answer_lengths):.1f}")
        print(f"最長答案: {max(answer_lengths)}")

def main():
    results = run_inference()
    save_submission(results)
    print("完成！")

if __name__ == "__main__":
    main()
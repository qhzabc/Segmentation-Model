from transformers import BertTokenizer, BertModel
import torch

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入数据
text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 执行前向传播并获取输出
with torch.no_grad():  # 不计算梯度，用于评估模式
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state  # 获取最后一个隐藏层的状态
    print(last_hidden_states.shape)  # 查看输出形状，通常是[batch_size, sequence_length, hidden_size]
from transformers import BertConfig, BertTokenizer

from src.models.components.modeling_bert import BertLMHeadModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="/data/.cache")
config = BertConfig.from_pretrained("bert-base-uncased", cache_dir="/data/.cache")
config.is_decoder = True
model = BertLMHeadModel.from_pretrained(
    "bert-base-uncased", cache_dir="/data/.cache", config=config
)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pathlib

# 1. Load your trained model and tokenizer
MODEL_PATH = pathlib.Path(__file__).with_name("outputs") / "bert-movie-genres/checkpoint-77418"

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Create dummy input for tracing
dummy_input = tokenizer("This is a sample input.", 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=128)

# 3. Export to ONNX
torch.onnx.export(
    model,                                # model
    (dummy_input["input_ids"], 
     dummy_input["attention_mask"]),      # inputs
    "bert_classifier.onnx",               # output file
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"}
    },
    opset_version=11
)

print("Model exported to bert_classifier.onnx")

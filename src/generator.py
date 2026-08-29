from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from retriever import retrieve


# 1. Load generation model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


# 2. User question
question = "What is the CEO's name?"


# 3. Retrieve relevant context from ChromaDB
retrieval_result = retrieve(question)

context = retrieval_result["context"]


# 4. Build grounded prompt
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer the user's question using only the provided context. "
            "If the answer is not in the context, say: "
            "'I could not find the answer in the provided document.'"
        )
    },
    {
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}
"""
    }
]


# 5. Convert chat into model format
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# 6. Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


# 7. Generate answer
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)


# 8. Remove original prompt tokens
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]


# 9. Decode model answer
answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


# 10. Display answer
print("\nQuestion:")
print(question)

print("\nRetrieved Context:")
print(context)

print("\nFinal Answer:")
print(answer)
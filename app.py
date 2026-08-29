import streamlit as st
from src.retriever import retrieve

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -------------------------
# PAGE
# -------------------------
st.title("BrightTech RAG Chatbot")

st.write(
    "Ask questions about the BrightTech document."
)


# -------------------------
# LOAD LLM
# -------------------------
@st.cache_resource
def load_model():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_model()


# -------------------------
# GENERATE ANSWER
# -------------------------
def generate_answer(question):

    # 1. Retrieve relevant information
    retrieval_result = retrieve(question)

    context = retrieval_result["context"]


    # 2. Build grounded prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user's question using only "
                "the provided context. "
                "If the answer is not in the context, "
                "say: 'I could not find the answer "
                "in the provided document.'"
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


    # 3. Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # 4. Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)


    # 5. Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )


    # 6. Remove prompt tokens
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    # 7. Decode answer
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return answer, retrieval_result


# -------------------------
# USER INPUT
# -------------------------
question = st.text_input(
    "Ask a question:"
)


# -------------------------
# ANSWER
# -------------------------
if question:

    with st.spinner("Searching document..."):

        answer, retrieval_result = generate_answer(
            question
        )


    st.subheader("Answer")

    st.write(answer)


    # -------------------------
    # SOURCES
    # -------------------------
    st.subheader("Sources")

    for metadata in retrieval_result["metadatas"]:

        st.write(
            f"Source: {metadata['source']} "
            f"| Chunk: {metadata['chunk_number']}"
        )
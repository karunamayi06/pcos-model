import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from safetensors.torch import load_file
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------
# STREAMLIT SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="PCOS Assistant", page_icon="💬", layout="wide")
st.title("👩‍⚕️ PCOS Detection & Chatbot Assistant")

# =========================================================
# 🩺 TAB 1 — IMAGE RECOGNITION USING ViT
# =========================================================

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
import streamlit as st

def load_vit_model():
    try:
        # Download model files from Hugging Face
        config_path = hf_hub_download(repo_id="MODEL06/pcos-vit", filename="config.json")
        weights_path = hf_hub_download(repo_id="MODEL06/pcos-vit", filename="model.safetensors")

        # Load configuration and model
        config = ViTConfig.from_pretrained(config_path)
        model = ViTForImageClassification(config)

        # Load weights
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.eval()

        # Load processor
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        return model, processor

    except Exception as e:
        st.error(f"Error loading ViT model from Hugging Face: {e}")
        return None, None


def pcos_image_recognizer(model, processor):
    st.header("🩻 PCOD Image Recognition (Transformer-based)")

    uploaded_image = st.file_uploader("Upload an ultrasound or related image:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ updated

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(preds, dim=1).item()
            confidence = preds[0][pred_label].item() * 100  # get probability %

        # ✅ Custom label mapping
        label_map = {
            "LABEL_0": "INFECTED",
            "LABEL_1": "NOT INFECTED",
            0: "INFECTED",
            1: "NOT INFECTED"
        }

        # Get prediction text
        label_name = model.config.id2label.get(pred_label, f"LABEL_{pred_label}") if hasattr(model.config, "id2label") else f"LABEL_{pred_label}"
        readable_label = label_map.get(label_name, label_map.get(pred_label, f"Class {pred_label}"))

        st.success(f"**Prediction:** {readable_label} ({confidence:.2f}% confidence)")
        st.balloons()


# =========================================================
# 💬 TAB 2 — PCOD CHATBOT (RAG with FAQ)
# =========================================================

def pcos_chatbot():
    st.header("💬 PCOD Chatbot (RAG with FAQs)")

    groq_api_key = st.secrets["ai"]

    if not groq_api_key:
        st.error("Please set your Groq API key: `os.environ['ai']='YOUR_KEY'`")
        st.stop()

    if not os.path.exists("pcod_faqs.txt"):
        st.error("Missing `pcod_faqs.txt` file.")
        st.stop()

    loader = TextLoader("pcod_faqs.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("pcod_faq_index"):
        vectorstore = FAISS.load_local("pcod_faq_index", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("pcod_faq_index")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

    prompt_template = """
    You are a helpful medical assistant specializing in PCOS.
    Answer ONLY from the provided FAQ context. If unsure, say "I don’t have enough information."

    Context:
    {context}

    Question:
    {question}

    Helpful Answer with explanation:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT, "verbose": False},
        return_source_documents=True
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask your question:")

    if st.button("Ask"):
        if user_input:
            result = qa_chain({"query": user_input})
            answer = result["result"]
            retrieved_docs = result["source_documents"]

            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Bot", answer, retrieved_docs))

    for item in st.session_state.history:
        if item[0] == "You":
            st.markdown(f"**🧑‍💻 You:** {item[1]}")
        else:
            role, text, sources = item
            st.markdown(f"**🤖 Bot:** {text}")
            with st.expander("📖 Sources used"):
                for doc in sources:
                    st.write(f"- {doc.page_content[:120]}...")


# =========================================================
# 🧭 TAB LAYOUT
# =========================================================

tab1, tab2 = st.tabs(["🩺 PCOS Recognizer", "💬 Chatbot"])

with tab1:
    model, processor = load_vit_model()
    if model and processor:
        pcos_image_recognizer(model, processor)

with tab2:
    pcos_chatbot()


import streamlit as st
import torch
from PIL import Image
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="👩‍⚕️ PCOS Assistant", page_icon="💬", layout="wide")
st.title("👩‍⚕️ PCOS Detection & Chatbot Assistant")

# ---------------------------------------------------------
# CACHED MODEL LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_vit_model():
    """Load ViT model and processor for PCOS image recognition."""
    try:
        config_path = hf_hub_download(repo_id="MODEL06/pcos-vit", filename="config.json")
        weights_path = hf_hub_download(repo_id="MODEL06/pcos-vit", filename="model.safetensors")

        config = ViTConfig.from_pretrained(config_path)
        model = ViTForImageClassification(config)
        model.load_state_dict(load_file(weights_path))
        model.eval()

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        return model, processor

    except Exception as e:
        st.error(f"❌ Error loading ViT model: {e}")
        return None, None


@st.cache_resource
def load_chatbot_resources():
    """Load embeddings, FAISS index, and LLM for the chatbot."""
    if not os.path.exists("pcod_faqs.txt"):
        st.error("⚠️ Missing `pcod_faqs.txt` file.")
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

    groq_api_key = st.secrets.get("ai")
    if not groq_api_key:
        st.error("⚠️ Please add your `ai` API key in Streamlit Secrets.")
        st.stop()

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template=(
            "You are a helpful medical assistant specializing in PCOS.\n"
            "Answer ONLY from the provided FAQ context. "
            "If unsure, say 'I don’t have enough information.'\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nHelpful Answer with explanation:"
        ),
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
        return_source_documents=True,
    )

    return qa_chain


# ---------------------------------------------------------
# IMAGE RECOGNITION TAB
# ---------------------------------------------------------
def pcos_image_recognizer(model, processor):
    st.header("🩻 PCOS Image Recognition (Vision Transformer)")

    uploaded = st.file_uploader("Upload an ultrasound image:", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("👆 Please upload an image to begin diagnosis.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item() * 100

    label_map = {0: "INFECTED", 1: "NOT INFECTED"}
    result = label_map.get(pred, f"Class {pred}")

    st.success(f"**Prediction:** {result} ({confidence:.2f}% confidence)")
    st.balloons()


# ---------------------------------------------------------
# CHATBOT TAB
# ---------------------------------------------------------
def pcos_chatbot(qa_chain):
    st.header("💬 PCOS Chatbot (RAG-powered FAQ)")

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask a question:")
    if st.button("Ask") and query:
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.history.append(("🧑‍💻 You", query))
        st.session_state.history.append(("🤖 Bot", answer, sources))

    for msg in st.session_state.history:
        if msg[0] == "🧑‍💻 You":
            st.markdown(f"**{msg[0]}:** {msg[1]}")
        else:
            _, ans, srcs = msg
            st.markdown(f"**🤖 Bot:** {ans}")
            with st.expander("📚 Sources used"):
                for s in srcs:
                    st.write(f"- {s.page_content[:120]}...")


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Select Mode:", ["🩺 Image Classification", "💬 Chatbot"])

if choice == "🩺 Image Classification":
    model, processor = load_vit_model()
    if model and processor:
        pcos_image_recognizer(model, processor)

elif choice == "💬 Chatbot":
    qa_chain = load_chatbot_resources()
    pcos_chatbot(qa_chain)

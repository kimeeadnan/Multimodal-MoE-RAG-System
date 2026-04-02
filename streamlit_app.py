import streamlit as st

from m3docrag.serve.single_query_service import SingleQueryConfig, SingleQueryM3DocVQAService


st.set_page_config(page_title="M3DocVQA RAG Demo", layout="wide")
st.title("M3DocVQA RAG demo (ColPali + FAISS + Qwen2-VL)")


with st.sidebar:
    st.header("Run settings")

    model_name = st.text_input("VQA model", value="Qwen2-VL-7B-Instruct")
    retrieval_adapter = st.text_input("ColPali adapter", value="colpali-v1.2")
    retrieval_backbone = st.text_input(
        "ColPali backbone", value="colpaligemma-3b-pt-448-base"
    )

    bits = st.selectbox("VQA bits", options=[4, 16], index=0)
    n_pages = st.slider("n_retrieval_pages (pages passed to VLM)", 1, 6, 3)

    faiss_index_type = st.selectbox("FAISS index type", options=["ivfflat", "flatip", "ivfpq"], index=0)
    faiss_nprobe = st.slider("FAISS nprobe (IVF only)", 1, 64, 16)

    use_weaviate_router = st.checkbox("Use Weaviate router (text/keyword narrowing)", value=True)
    weaviate_top_k_docs = st.slider("Weaviate top_k docs", 5, 50, 20)

    embedding_name = st.text_input("Embedding name", value="colpali_m3docvqa_dev")
    split = st.selectbox("Dataset split", options=["dev", "train"], index=0)

    st.markdown("---")
    st.write("This loads ColPali embeddings index + Qwen2-VL weights.")


@st.cache_resource(show_spinner=True)
def get_service(
    model_name: str,
    retrieval_adapter: str,
    retrieval_backbone: str,
    bits: int,
    n_pages: int,
    faiss_index_type: str,
    faiss_nprobe: int,
    use_weaviate_router: bool,
    weaviate_top_k_docs: int,
    embedding_name: str,
    split: str,
):
    cfg = SingleQueryConfig(
        split=split,
        embedding_name=embedding_name,
        faiss_index_type=faiss_index_type,
        n_retrieval_pages_default=n_pages,
        faiss_nprobe=faiss_nprobe,
        use_weaviate_router=use_weaviate_router,
        weaviate_top_k_docs=weaviate_top_k_docs,
        model_name_or_path=model_name,
        retrieval_backbone_name_or_path=retrieval_backbone,
        retrieval_adapter_name_or_path=retrieval_adapter,
        bits=bits,
    )
    return SingleQueryM3DocVQAService(cfg)


question = st.text_area("Question", height=90, placeholder="Type a question about the M3DocVQA documents...")

run = st.button("Answer", type="primary")

if run and question.strip():
    with st.spinner("Running retrieval + generation..."):
        svc = get_service(
            model_name=model_name,
            retrieval_adapter=retrieval_adapter,
            retrieval_backbone=retrieval_backbone,
            bits=bits,
            n_pages=n_pages,
            faiss_index_type=faiss_index_type,
            faiss_nprobe=faiss_nprobe,
            use_weaviate_router=use_weaviate_router,
            weaviate_top_k_docs=weaviate_top_k_docs,
            embedding_name=embedding_name,
            split=split,
        )

        out = svc.answer(question, n_retrieval_pages=n_pages)

    st.subheader("Answer")
    st.write(out["pred_answer"])

    st.subheader("Router diagnostics")
    st.json(
        {
            "router_expert": out["router_expert"],
            "router_reason": out["router_reason"],
            "router_doc_ids_filter": out["router_doc_ids_filter"],
            "router_features": out["router_features"],
        }
    )

    st.subheader("Retrieved pages (top)")
    st.table(out["page_retrieval_results"])

    st.caption(f"Time: retrieval={out['time_retrieval']:.2f}s, VQA={out['time_qa']:.2f}s")


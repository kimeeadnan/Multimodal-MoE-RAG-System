from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import safetensors
import torch
from accelerate import Accelerator
from loguru import logger

from m3docrag.datasets.m3_docvqa.dataset import M3DocVQADataset
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.retrieval.weaviate_mmqa import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    connect_weaviate,
)
from m3docrag.rag.base import RAGModelBase
from m3docrag.routing.moe_router import features_to_dict
from m3docrag.routing.rag_plan import (
    build_retrieval_plan,
    enrich_plan_with_weaviate_doc_ids,
)
from m3docrag.utils.distributed import supports_flash_attention
from m3docrag.utils.paths import (
    LOCAL_EMBEDDINGS_DIR,
    LOCAL_MODEL_DIR,
)
from m3docrag.utils.prompts import short_answer_template
from m3docrag.vqa import VQAModel


@dataclass(frozen=True)
class SingleQueryConfig:
    # Dataset / embeddings
    split: str = "dev"
    data_name: str = "m3-docvqa"
    embedding_name: str = "colpali_m3docvqa_dev"
    retrieval_model_type: str = "colpali"  # choices: colpali, colbert (only colpali supported here)
    faiss_index_type: str = "ivfflat"  # choices: flatip, ivfflat, ivfpq

    # Retrieval
    n_retrieval_pages_default: int = 3
    faiss_search_k: Optional[int] = None
    faiss_nprobe: int = 16

    # Router / Weaviate
    use_weaviate_router: bool = True
    weaviate_collection: str = DEFAULT_COLLECTION
    weaviate_top_k_docs: int = 20

    # Models
    model_name_or_path: str = "Qwen2-VL-7B-Instruct"
    retrieval_backbone_name_or_path: str = "colpaligemma-3b-pt-448-base"
    retrieval_adapter_name_or_path: str = "colpali-v1.2"
    bits: int = 4


class SingleQueryM3DocVQAService:
    """
    Interactive (single-question) RAG service using the same components as `examples/run_rag_m3docvqa.py`.
    """

    def __init__(self, config: SingleQueryConfig):
        self.config = config
        self._doc_images_cache: Dict[str, List[Any]] = {}

        if self.config.retrieval_model_type != "colpali":
            raise NotImplementedError(
                f"SingleQueryService currently supports only retrieval_model_type='colpali', got {self.config.retrieval_model_type}"
            )

        self.accelerator = Accelerator()

        # Dataset: gives us PDF->page image extraction and the split supporting doc ids.
        args = SimpleNamespace(
            split=self.config.split,
            data_name=self.config.data_name,
            data_len=None,
            loop_unique_doc_ids=False,
            retrieval_model_type=self.config.retrieval_model_type,
            embedding_name=self.config.embedding_name,
        )
        self.dataset = M3DocVQADataset(args=args)

        # Retrieval: ColPali (keep on CPU to preserve GPU memory).
        retrieval_model = ColPaliRetrievalModel(
            backbone_name_or_path=Path(LOCAL_MODEL_DIR)
            / self.config.retrieval_backbone_name_or_path,
            adapter_name_or_path=Path(LOCAL_MODEL_DIR)
            / self.config.retrieval_adapter_name_or_path,
        )
        retrieval_model.model = retrieval_model.model.to(torch.device("cpu"))

        # VQA model: Qwen2-VL (on GPU / accelerator device)
        if "florence" in self.config.model_name_or_path.lower():
            model_type = "florence2"
        elif "idefics2" in self.config.model_name_or_path.lower():
            model_type = "idefics2"
        elif "idefics3" in self.config.model_name_or_path.lower():
            model_type = "idefics3"
        elif "internvl2" in self.config.model_name_or_path.lower():
            model_type = "internvl2"
        elif "qwen2" in self.config.model_name_or_path.lower():
            model_type = "qwen2"
        else:
            raise KeyError(
                f"Unknown VQA model type for: {self.config.model_name_or_path}"
            )

        use_flash_attn = True
        attn_implementation = "flash_attention_2"
        if not supports_flash_attention():
            use_flash_attn = False
            attn_implementation = "eager"

        local_model_dir = Path(LOCAL_MODEL_DIR) / self.config.model_name_or_path
        vqa_model = VQAModel(
            model_name_or_path=local_model_dir,
            model_type=model_type,
            bits=self.config.bits,
            use_flash_attn=use_flash_attn,
            attn_implementation=attn_implementation,
        )
        vqa_model.model = self.accelerator.prepare(vqa_model.model)

        self.rag_model = MultimodalRAGModel(
            retrieval_model=retrieval_model,
            vqa_model=vqa_model,
        )

        # FAISS index + token->page mapping (low-memory variant)
        self.faiss_index, self.token2pageuid = self._load_faiss_and_token2pageuid()

        # Optional Weaviate router
        self.weaviate_client = None
        self.weaviate_embed_model = None
        if self.config.use_weaviate_router:
            self._init_weaviate_router()

    def _init_weaviate_router(self) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self.weaviate_client = connect_weaviate()
            self.weaviate_embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL)
            logger.info("Weaviate router: connected and BGE model loaded")
        except Exception as e:
            logger.warning(f"Weaviate router disabled (connect failed): {e}")
            self.weaviate_client = None
            self.weaviate_embed_model = None

    def _load_faiss_and_token2pageuid(
        self,
    ) -> Tuple[Any, List[str]]:
        import faiss

        # Match `examples/run_rag_m3docvqa.py` index discovery.
        local_index_dir = (
            Path(LOCAL_EMBEDDINGS_DIR)
            / f"{self.config.embedding_name}_pageindex_{self.config.faiss_index_type}"
        )
        if not (local_index_dir / "index.bin").is_file():
            alt = Path(LOCAL_EMBEDDINGS_DIR) / f"faiss_{self.config.embedding_name}"
            if (alt / "index.bin").is_file():
                local_index_dir = alt

        if not (local_index_dir / "index.bin").is_file():
            raise FileNotFoundError(
                "FAISS index not found. Expected either:\n"
                f"- {local_index_dir / 'index.bin'}\n"
                f"- .../faiss_{self.config.embedding_name}/index.bin"
            )

        logger.info(f"Loading FAISS index: {local_index_dir / 'index.bin'}")
        index = faiss.read_index(str(local_index_dir / "index.bin"))
        if self.config.faiss_index_type != "flatip" and hasattr(index, "nprobe"):
            index.nprobe = int(self.config.faiss_nprobe)
            logger.info(f"FAISS nprobe={index.nprobe}")

        # Build token2pageuid mapping from embedding tensor shapes only.
        token2pageuid: List[str] = []
        emb_dir = Path(LOCAL_EMBEDDINGS_DIR) / self.config.embedding_name
        logger.info("Building token->page mapping (shape-only)")

        for doc_id in self.dataset.all_supporting_doc_ids:
            emb_path = emb_dir / f"{doc_id}.safetensors"
            if not emb_path.exists():
                continue
            with safetensors.safe_open(
                str(emb_path), framework="pt", device="cpu"
            ) as f:
                try:
                    n_pages, n_tokens, _ = f.get_slice("embeddings").get_shape()
                except Exception:
                    n_pages, n_tokens, _ = f.get_tensor("embeddings").shape

            for page_id in range(int(n_pages)):
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * int(n_tokens))

        logger.info(f"Built token2pageuid entries: {len(token2pageuid)}")
        return index, token2pageuid

    def _get_doc_page_images(self, doc_id: str) -> List[Any]:
        if doc_id not in self._doc_images_cache:
            self._doc_images_cache[doc_id] = self.dataset.get_images_from_doc_id(doc_id)
        return self._doc_images_cache[doc_id]

    @torch.inference_mode()
    def answer(
        self,
        query: str,
        *,
        n_retrieval_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        n_retrieval_pages = (
            n_retrieval_pages if n_retrieval_pages is not None else self.config.n_retrieval_pages_default
        )

        plan = build_retrieval_plan(query)
        if self.config.use_weaviate_router and self.weaviate_client is not None:
            plan = enrich_plan_with_weaviate_doc_ids(
                plan,
                query,
                self.weaviate_client,
                self.weaviate_embed_model,
                top_k_docs=self.config.weaviate_top_k_docs,
            )

        allowed = frozenset(plan.doc_ids_filter) if plan.doc_ids_filter else None

        t0 = time.perf_counter()
        retrieval_results = self.rag_model.retrieve_pages_from_docs(
            query=query,
            docid2embs={},  # not used when FAISS index is provided
            docid2lens=None,
            index=self.faiss_index,
            token2pageuid=self.token2pageuid,
            all_token_embeddings=None,
            n_return_pages=n_retrieval_pages,
            faiss_search_k=self.config.faiss_search_k,
            show_progress=False,
            allowed_doc_ids=allowed,
        )
        time_retrieval = time.perf_counter() - t0

        images: List[Any] = []
        for doc_id, page_idx, _score in retrieval_results:
            page_images = self._get_doc_page_images(doc_id)
            if 0 <= page_idx < len(page_images):
                images.append(page_images[page_idx])

        if "florence" in self.config.model_name_or_path.lower():
            text_input = query
        else:
            text_input = short_answer_template.substitute({"question": query})

        t1 = time.perf_counter()
        answer_text = self.rag_model.run_vqa(images=images, question=text_input)
        time_qa = time.perf_counter() - t1

        return {
            "router_expert": plan.expert,
            "router_reason": plan.reason,
            "router_features": features_to_dict(plan.features),
            "router_doc_ids_filter": plan.doc_ids_filter,
            "page_retrieval_results": [
                {"doc_id": doc_id, "page_idx": page_idx, "score": score}
                for (doc_id, page_idx, score) in retrieval_results
            ],
            "pred_answer": answer_text,
            "time_retrieval": time_retrieval,
            "time_qa": time_qa,
        }


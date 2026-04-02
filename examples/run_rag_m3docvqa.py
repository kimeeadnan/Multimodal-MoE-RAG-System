# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import time
from pathlib import Path

import accelerate
import pytz
import safetensors
import torch
import transformers
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from m3docrag.datasets.m3_docvqa import M3DocVQADataset, evaluate_prediction_file
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.routing.moe_router import features_to_dict
from m3docrag.routing.rag_plan import (
    build_retrieval_plan,
    enrich_plan_with_weaviate_doc_ids,
)
from m3docrag.utils.args import parse_args
from m3docrag.utils.distributed import (
    barrier,
    global_rank,
    is_distributed,
    local_rank,
    log_runtime_info,
    print_gpu_stats,
    supports_flash_attention,
)
from m3docrag.utils.paths import (
    LOCAL_DATA_DIR,
    LOCAL_EMBEDDINGS_DIR,
    LOCAL_MODEL_DIR,
)
from m3docrag.utils.prompts import short_answer_template
from m3docrag.utils.tar import extract_tarfile
from m3docrag.vqa import VQAModel


def run_model(
    rag_model: MultimodalRAGModel,
    datum,
    dataset: M3DocVQADataset,
    docid2embs: dict[str, torch.Tensor],
    docid2lens=None,
    index=None,
    token2pageuid=None,
    all_token_embeddings=None,
    n_return_pages=1,
    args=None,
    weaviate_client=None,
    weaviate_embed_model=None,
):
    # if type(datum['num_pages']) == list:
    batch = datum
    datum = {}
    for k, v in batch.items():
        datum[k] = v[0]

    query = datum["question"]

    out_dict = {}

    plan = build_retrieval_plan(query)
    if args.use_weaviate_router:
        plan = enrich_plan_with_weaviate_doc_ids(
            plan,
            query,
            weaviate_client,
            weaviate_embed_model,
            top_k_docs=args.weaviate_top_k_docs,
        )
    qid = datum.get("qid", "")
    logger.info(
        f"router expert={plan.expert} reason={plan.reason!r} qid={qid!r} "
        f"weaviate_docs={len(plan.doc_ids_filter or [])} query={query[:120]!r}..."
    )
    out_dict["router_expert"] = plan.expert
    out_dict["router_reason"] = plan.reason
    out_dict["router_features"] = features_to_dict(plan.features)
    out_dict["router_doc_ids_filter"] = plan.doc_ids_filter

    allowed = frozenset(plan.doc_ids_filter) if plan.doc_ids_filter else None

    start = time.perf_counter()

    # Stage 1: Page retrieval (optional Weaviate doc filter for keyword/text)
    top_n_page_retrieval_results = rag_model.retrieve_pages_from_docs(
        query=query,
        docid2embs=docid2embs,
        docid2lens=docid2lens,
        index=index,
        token2pageuid=token2pageuid,
        all_token_embeddings=all_token_embeddings,
        n_return_pages=n_return_pages,
        faiss_search_k=args.faiss_search_k,
        show_progress=True,
        allowed_doc_ids=allowed,
    )
    logger.info(top_n_page_retrieval_results)
    out_dict["page_retrieval_results"] = top_n_page_retrieval_results

    end = time.perf_counter()

    time_retrieval = end - start
    logger.info(f"time_retrieval: {time_retrieval}")

    start = time.perf_counter()

    if args.retrieval_only:
        pred_answer = ""
        out_dict["pred_answer"] = pred_answer
    else:
        # Stage 2: QA on the retrived page
        # Obtain images from the page retrieval results
        images = []
        for doc_id, page_idx, scores in top_n_page_retrieval_results:
            page_images = dataset.get_images_from_doc_id(doc_id)
            page_image = page_images[page_idx]
            images += [page_image]

        logger.info(len(images))

        # Run VQA
        if "florence" in args.model_name_or_path.lower():
            text_input = query
        else:
            text_input = short_answer_template.substitute({"question": query})
        pred_answer = rag_model.run_vqa(images=images, question=text_input)

        assert isinstance(pred_answer, str)
        out_dict["pred_answer"] = pred_answer

    end = time.perf_counter()

    time_qa = end - start
    logger.info(f"time_qa: {time_qa}")

    out_dict["time_retrieval"] = time_retrieval
    out_dict["time_qa"] = time_qa

    logger.info(query)
    logger.info(pred_answer)
    logger.info(datum["answers"])

    return out_dict


def evaluate(
    data_loader,
    rag_model,
    index=None,
    data_len=None,
    args=None,
    weaviate_client=None,
    weaviate_embed_model=None,
    **kwargs,
):
    if data_len is not None:
        logger.info(f"eval on the first {data_len} items")

    # docid2embs = data_loader.dataset.load_all_embeddings()

    logger.info("Preparing doc indices")
    docid2embs = {}
    token2pageuid = []
    all_token_embeddings = None

    # Low-memory full pipeline mode:
    # when FAISS index exists, avoid loading all page embeddings into one giant matrix.
    if args.retrieval_model_type == "colpali" and index is not None:
        emb_dir = Path(LOCAL_EMBEDDINGS_DIR) / args.embedding_name
        for doc_id in tqdm(
            data_loader.dataset.all_supporting_doc_ids,
            total=len(data_loader.dataset.all_supporting_doc_ids),
            desc="Building token->page mapping",
        ):
            emb_path = emb_dir / f"{doc_id}.safetensors"
            if not emb_path.exists():
                continue
            with safetensors.safe_open(str(emb_path), framework="pt", device="cpu") as f:
                # Prefer metadata-only shape path; fallback is safe.
                try:
                    n_pages, n_tokens, _ = f.get_slice("embeddings").get_shape()
                except Exception:
                    n_pages, n_tokens, _ = f.get_tensor("embeddings").shape
            for page_id in range(int(n_pages)):
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * int(n_tokens))
        logger.info(f"Built token2pageuid entries: {len(token2pageuid)}")
        logger.info("Using FAISS scores directly (skip all_token_embeddings reconstruction)")
    elif args.retrieval_model_type == "colpali":
        # Fallback path when index is missing: keep original brute-force behavior.
        docid2embs = data_loader.dataset.load_all_embeddings()
        all_token_embeddings_list = []
        for doc_id, doc_emb in tqdm(docid2embs.items(), total=len(docid2embs)):
            for page_id in range(len(doc_emb)):
                page_emb = doc_emb[page_id].view(-1, 128)
                all_token_embeddings_list.append(page_emb)
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * page_emb.shape[0])
        logger.info(len(all_token_embeddings_list))
        all_token_embeddings = torch.cat(all_token_embeddings_list, dim=0)
        all_token_embeddings = all_token_embeddings.float().numpy()
        logger.info("Created flattened token embeddings / token2pageuid")

    qid2result = {}

    total_time_retrieval = 0
    total_time_qa = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch["question"])

        # Single batch
        assert bs == 1
        qid = batch["qid"][0]

        with torch.no_grad():
            outputs = run_model(
                rag_model,
                batch,
                dataset=data_loader.dataset,
                docid2embs=docid2embs,
                index=index,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                n_return_pages=args.n_retrieval_pages,
                args=args,
                weaviate_client=weaviate_client,
                weaviate_embed_model=weaviate_embed_model,
            )

            pred_answer = outputs["pred_answer"]
            assert isinstance(pred_answer, str), type(pred_answer)

            total_time_qa += outputs["time_qa"]
            total_time_retrieval += outputs["time_retrieval"]

        qid2result[qid] = outputs

    logger.info(total_time_qa)
    logger.info(total_time_retrieval)

    avg_time_qa = total_time_qa / len(data_loader)
    avg_time_retrieval = total_time_retrieval / len(data_loader)
    logger.info(avg_time_qa)
    logger.info(avg_time_retrieval)

    return qid2result


def main():
    args = parse_args()

    logger.info(torch.__version__)
    logger.info(transformers.__version__)
    logger.info(accelerate.__version__)

    log_runtime_info()
    print_gpu_stats()

    accelerator = Accelerator()

    if not is_distributed() or global_rank() == 0:
        logger.info(f"Process {global_rank()}:{local_rank()} - args {args}")

    local_data_dir = Path(LOCAL_DATA_DIR) / args.data_name
    local_embedding_dir = Path(LOCAL_EMBEDDINGS_DIR) / args.embedding_name
    local_model_dir = Path(LOCAL_MODEL_DIR) / args.model_name_or_path
    local_retrieval_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_model_name_or_path
    )
    local_retrieval_adapter_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_adapter_model_name_or_path
    )
    # local_ft_model_dir = Path(LOCAL_MODEL_DIR) / args.ft_model_name_or_path

    local_index_dir = (
        Path(LOCAL_EMBEDDINGS_DIR)
        / f"{args.embedding_name}_pageindex_{args.faiss_index_type}"
    )
    if not (local_index_dir / "index.bin").is_file():
        alt = Path(LOCAL_EMBEDDINGS_DIR) / f"faiss_{args.embedding_name}"
        if (alt / "index.bin").is_file():
            logger.info(f"Using FAISS dir {alt} (found index.bin; canonical {local_index_dir} missing)")
            local_index_dir = alt
    # local_answer_extraction_model_dir =  Path(LOCAL_MODEL_DIR) / args.answer_extraction_model_name_or_path

    if is_distributed():
        barrier()

    if not is_distributed() or global_rank() == 0:
        if not local_data_dir.exists():
            raise ValueError(f"Data directory {local_data_dir} does not exist")

        if not local_embedding_dir.exists():
            raise ValueError(
                f"Embedding directory {local_embedding_dir} does not exist"
            )

        if local_model_dir.exists() or args.retrieval_only:
            logger.info("Model exists - pass")
        else:
            raise ValueError(
                f"Model directory {local_model_dir} does not exist"
            )

        if args.use_retrieval:
            if not local_retrieval_model_dir.exists():
                raise ValueError(
                    f"Retrieval model directory {local_retrieval_model_dir} does not exist"
                )

            if not local_retrieval_adapter_model_dir.exists():
                raise ValueError(
                    f"Retrieval adapter model directory {local_retrieval_adapter_model_dir} does not exist"
                )

        if not (local_index_dir / "index.bin").is_file():
            raise ValueError(
                f"FAISS index not found at {local_index_dir / 'index.bin'} "
                f"(expected .../{args.embedding_name}_pageindex_{args.faiss_index_type}/index.bin "
                f"or .../faiss_{args.embedding_name}/index.bin)"
            )

    if is_distributed():
        barrier()

    # Create Retrieval Model (Step 1)
    assert args.use_retrieval
    if args.retrieval_model_type == "colpali":
        colpali_model = ColPaliRetrievalModel(
            backbone_name_or_path=local_retrieval_model_dir,
            adapter_name_or_path=local_retrieval_adapter_model_dir,
        )
        retrieval_model = colpali_model

    logger.info(f"loaded Retrieval model -: {local_retrieval_model_dir}")

    # Create QA / VQA Model (Step 2)

    if args.retrieval_only:
        rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=None)
        logger.info("skipping QA model")

    else:
        if "florence" in args.model_name_or_path.lower():
            model_type = "florence2"
        elif "idefics2" in args.model_name_or_path.lower():
            model_type = "idefics2"
        elif "idefics3" in args.model_name_or_path.lower():
            model_type = "idefics3"
        elif "internvl2" in args.model_name_or_path.lower():
            model_type = "internvl2"
        elif "qwen2" in args.model_name_or_path.lower():
            model_type = "qwen2"
        else:
            raise KeyError(f"model type unknown for: {args.model_name_or_path}")

        use_flash_attn = True
        attn_implementation = "flash_attention_2"
        if not supports_flash_attention():
            use_flash_attn = False
            attn_implementation = "eager"

        vqa_model = VQAModel(
            model_name_or_path=local_model_dir,
            model_type=model_type,
            bits=args.bits,
            use_flash_attn=use_flash_attn,
            attn_implementation=attn_implementation,
        )

        logger.info(f"loaded VQA model - {model_type}: {local_model_dir}")

        rag_model = MultimodalRAGModel(
            retrieval_model=retrieval_model, vqa_model=vqa_model
        )

    logger.info("Created RAG model")

    dataset = M3DocVQADataset(args=args)
    logger.info("loaded dataset")

    index = None
    if local_index_dir.exists():
        logger.info("Loading faiss index")
        import faiss

        index = faiss.read_index(str(local_index_dir / "index.bin"))
        if args.faiss_index_type != "flatip" and hasattr(index, "nprobe"):
            index.nprobe = int(args.faiss_nprobe)
            logger.info(f"FAISS nprobe={index.nprobe}")
        logger.info("Loading faiss index -- done")

    def list_collate_fn(batch):
        batch = {
            k: [dic[k] for dic in batch] for k in batch[0]
        }  # List of dictionaries to dict of lists.
        return batch

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=list_collate_fn
    )

    weaviate_client = None
    weaviate_embed_model = None
    if args.use_weaviate_router:
        try:
            from sentence_transformers import SentenceTransformer

            from m3docrag.retrieval.weaviate_mmqa import (
                DEFAULT_EMBED_MODEL,
                connect_weaviate,
            )

            weaviate_client = connect_weaviate()
            weaviate_embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL)
            logger.info("Weaviate router: connected and BGE model loaded")
        except Exception as e:
            logger.warning(f"Weaviate router disabled (failed to connect or load BGE): {e}")

    try:
        if args.retrieval_only:
            retrieval_model.model, data_loader = accelerator.prepare(
                retrieval_model.model, data_loader
            )
        else:
            # Keep ColPali on CPU during VQA so Qwen2-VL fits on a single consumer GPU.
            retrieval_model.model = retrieval_model.model.to(torch.device("cpu"))
            vqa_model.model, data_loader = accelerator.prepare(
                vqa_model.model, data_loader
            )
            logger.info("Retrieval model on CPU; VQA model on accelerator device")

        eval_out = evaluate(
            data_loader=data_loader,
            rag_model=rag_model,
            index=index,
            data_len=args.data_len,
            args=args,
            weaviate_client=weaviate_client,
            weaviate_embed_model=weaviate_embed_model,
        )

        samples = eval_out

        EST = pytz.timezone("US/Eastern")
        experiment_date = (
            datetime.datetime.now().astimezone(EST).strftime("%Y-%m-%d_%H-%M-%S")
        )

        save_dir = Path(args.output_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info("Results will be saved at:", save_dir)

        if args.retrieval_model_type == "colpali":
            ret_name = args.retrieval_adapter_model_name_or_path
        else:
            ret_name = args.retrieval_model_name_or_path

        if args.retrieval_only:
            pred_save_fname = f"{ret_name}_{args.faiss_index_type}_ret{args.n_retrieval_pages}_{experiment_date}.json"
        else:
            pred_save_fname = f"{ret_name}_{args.faiss_index_type}_ret{args.n_retrieval_pages}_{args.model_name_or_path}_{experiment_date}.json"
        results_file = save_dir / pred_save_fname
        with open(results_file, "w") as f:
            json.dump(samples, f, indent=4)

        logger.info(f"Prediction results saved at: {results_file}")

        # Evaluation
        all_eval_scores = evaluate_prediction_file(
            samples,
            dataset.mmqa_data_path,  # '/job/datasets/m3-docvqa/MMQA_dev.jsonl'
        )
        if args.retrieval_only:
            eval_save_fname = f"{ret_name}_{args.faiss_index_type}_ret{args.n_retrieval_pages}_{experiment_date}_eval_results.json"
        else:
            eval_save_fname = f"{ret_name}_{args.faiss_index_type}_ret{args.n_retrieval_pages}_{args.model_name_or_path}_{experiment_date}_eval_results.json"
        results_file = save_dir / eval_save_fname
        with open(results_file, "w") as f:
            json.dump(all_eval_scores, f, indent=4)

        logger.info(f"Evaluation results saved at: {results_file}")
    finally:
        if weaviate_client is not None:
            weaviate_client.close()

    if is_distributed():
        barrier()


if __name__ == "__main__":
    main()

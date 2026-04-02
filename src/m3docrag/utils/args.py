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

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
from icecream import ic

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    # Data settings (default dev: matches MMQA_dev + typical colpali_*_dev embeddings; use --split=train if you embedded all train docs)
    split: str = 'dev'
    data_name: str = field(default='m3-docvqa', metadata={"help": "Local name to be stored at LOCAL_DATA_DIR"})
    data_len: int = field(default=None, metadata={"help": "number of examples to subsample from dataset"})
    use_dummy_images: bool = field(default=False, metadata={"help": "if true, skip downloading images"})
    load_embedding: bool = False
    embedding_name: str = "colpali_m3docvqa_dev"

    max_pages: int = 20
    do_page_padding: bool = False
    
    # Retrieval settings
    retrieval_model_type: str = field(default='colpali', metadata={"choices": ['colpali', 'colbert']})
    use_retrieval: bool = True
    retrieval_only: bool = field(default=False, metadata={"help": "not running stage 2 (VQA)"})
    page_retrieval_type: str = 'logits'
    loop_unique_doc_ids: bool = field(default=False, metadata={"help": "if true, apply retrieval only on unique doc ids"})

    n_retrieval_pages: int = field(
        default=3,
        metadata={
            "help": "Top pages passed to the VLM after ColPali+FAISS MaxSim. "
            "Use 1 for minimal VRAM smoke tests; 3–5 often helps recall if memory allows."
        },
    )

    faiss_search_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "FAISS neighbors retrieved per query token for ColPali MaxSim. "
            "None = max(64, n_retrieval_pages * 16). Must be >= n_retrieval_pages."
        },
    )
    faiss_nprobe: int = field(
        default=16,
        metadata={
            "help": "IVF nprobe at query time (ivfflat / ivfpq only). Ignored for flatip. "
            "Higher improves recall, slower search. Typical range 8–64."
        },
    )

    use_weaviate_router: bool = field(
        default=False,
        metadata={
            "help": "If true, keyword/text routes query Weaviate for top doc IDs and narrow ColPali+FAISS to those docs."
        },
    )
    weaviate_top_k_docs: int = field(
        default=20,
        metadata={"help": "Max distinct doc_ids to keep from Weaviate for keyword/text routes."},
    )

    # Embedding indexing settings
    faiss_index_type: str = field(default='ivfflat', metadata={"choices": ['flatip', 'ivfflat', 'ivfpq']})


    # Local paths
    model_name_or_path: Optional[str] = field(default="Qwen2-VL-7B-Instruct")
    retrieval_model_name_or_path: Optional[str] = field(default="colpaligemma-3b-pt-448-base")
    retrieval_adapter_model_name_or_path: Optional[str] = field(default="colpali-v1.2")

    # Model settings
    bits: int = field(default=16, metadata={"help": "Floating point precision. Use '4' for 4-bit quantization to save memory"})

    # idefics2 settings
    do_image_splitting: bool = False


_example_arg_str = """
--output_dir=/job/outputs
--data_name=m3-docvqa
--use_retrieval=True
"""
_example_args = _example_arg_str.strip().split('\n')


def parse_args(args=None):
    parser = transformers.HfArgumentParser(TrainingArguments)
    parsed_args, remaining_args = parser.parse_args_into_dataclasses(args, return_remaining_strings=True)
    ic(remaining_args)

    return parsed_args
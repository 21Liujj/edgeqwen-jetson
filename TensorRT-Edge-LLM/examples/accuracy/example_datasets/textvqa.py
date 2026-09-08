# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TextVQA dataset converter for TensorRT Edge-LLM.

Source: HuggingFace `textvqa` dataset (val split).
TextVQA focuses on visual question answering requiring reading text in images.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class TextVQADataset(EdgeLLMDataset):
    """
    TextVQA dataset converter.

    Expected HF schema (`textvqa`):
        {
            'image': PIL.Image,
            'question': 'What text is shown on the sign?',
            'answers': ['STOP', 'stop', 'Stop'],  # multiple acceptable answers
            'question_id': 12345
        }
    """

    def __init__(self,
                 dataset: Dataset,
                 config: DatasetConfig,
                 **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format TextVQA prompt with question."""
        question = data.get("question", "")
        return f"{question}"

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        return "You are a helpful assistant that can read and understand text in images."

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save TextVQA image and return relative path."""
        image_paths = []

        if "image" in data and data["image"] is not None:
            question_id = data.get("question_id", data.get("index", 0))
            image_filename = f"textvqa_{question_id}.jpg"

            image_path = os.path.join(self.images_dir, image_filename)
            image = data["image"]
            # Ensure RGB for JPEG
            image = image.convert('RGB')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

        return image_paths

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract answers from TextVQA data.

        TextVQA has multiple acceptable answers per question.
        We join them with ' | ' for evaluation.
        """
        answers = data.get("answers", [])
        if isinstance(answers, list) and answers:
            # Join all acceptable answers
            return " | ".join(str(a) for a in answers)
        elif isinstance(answers, str):
            return answers
        return None


def convert_textvqa_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "textvqa",
        output_dir: Union[str, os.PathLike] = "textvqa_output",
        split: str = "validation",
        max_samples: Optional[int] = None,
        output_filename: str = "dataset.json"):
    """Convert TextVQA dataset into TensorRT Edge-LLM JSON format.

    Args:
        config: DatasetConfig with batch_size / sampling params.
        dataset_name_or_dir: HF repo id (default 'textvqa') or local directory.
        output_dir: where to write dataset.json + images/.
        split: HF split (default 'validation').
        max_samples: optional cap for quick testing.
        output_filename: JSON manifest filename (default 'dataset.json').
    """
    # Use lmms-lab/textvqa which is parquet-based (datasets 5.x compatible)
    if dataset_name_or_dir == "textvqa":
        dataset_name_or_dir = "lmms-lab/textvqa"
    print(f"Loading TextVQA from {dataset_name_or_dir} (split={split})")
    dataset = load_dataset(dataset_name_or_dir, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to first {len(dataset)} samples")

    converter = TextVQADataset(dataset=dataset,
                               config=config,
                               output_dir=output_dir)
    out_path = converter.process_and_save_dataset(
        output_filename=output_filename)
    print(f"TextVQA dataset written to {out_path}")
    return out_path

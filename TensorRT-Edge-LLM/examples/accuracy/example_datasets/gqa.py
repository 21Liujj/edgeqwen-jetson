# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GQA dataset converter for TensorRT Edge-LLM.

Source: HuggingFace `lmms-lab/GQA` dataset (testdev_balanced split).
GQA focuses on real-world visual reasoning and compositional question answering.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class GQADataset(EdgeLLMDataset):
    """
    GQA dataset converter.

    Expected HF schema (`lmms-lab/GQA`):
        {
            'image': PIL.Image,
            'question': 'How many red apples are on the table?',
            'answer': '3',
            'question_id': 'abc123',
            'imageId': '12345'
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
        """Format GQA prompt with question."""
        question = data.get("question", "")
        return f"{question}"

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        return "You are a helpful assistant that can analyze images and answer questions about them."

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save GQA image and return relative path."""
        image_paths = []

        if "image" in data and data["image"] is not None:
            # Use question_id or imageId for filename
            image_id = data.get("question_id") or data.get("imageId") or data.get("index", 0)
            image_filename = f"gqa_{image_id}.jpg"

            image_path = os.path.join(self.images_dir, image_filename)
            image = data["image"]
            # Ensure RGB for JPEG
            image = image.convert('RGB')
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

        return image_paths

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract answer from GQA data."""
        answer = data.get("answer")
        if answer is not None:
            return str(answer).strip()
        return None


def convert_gqa_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "lmms-lab/GQA",
        output_dir: Union[str, os.PathLike] = "gqa_output",
        split: str = "testdev_balanced",
        max_samples: Optional[int] = None,
        output_filename: str = "dataset.json"):
    """Convert GQA dataset into TensorRT Edge-LLM JSON format.

    Args:
        config: DatasetConfig with batch_size / sampling params.
        dataset_name_or_dir: HF repo id (default 'lmms-lab/GQA') or local directory.
        output_dir: where to write dataset.json + images/.
        split: HF split (default 'testdev_balanced' for balanced evaluation).
        max_samples: optional cap for quick testing.
        output_filename: JSON manifest filename (default 'dataset.json').
    """
    print(f"Loading GQA from {dataset_name_or_dir} (split={split})")
    # GQA requires config name - use testdev_balanced for evaluation
    config_name = "testdev_balanced_images"
    dataset = load_dataset(dataset_name_or_dir, config_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to first {len(dataset)} samples")

    converter = GQADataset(dataset=dataset,
                          config=config,
                          output_dir=output_dir)
    out_path = converter.process_and_save_dataset(
        output_filename=output_filename)
    print(f"GQA dataset written to {out_path}")
    return out_path

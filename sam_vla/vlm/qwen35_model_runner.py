"""Loads Qwen3.5-9B and runs multi-turn image+text generation.

Sibling of qwen_model_runner.py (Qwen2.5-VL-3B), not a replacement for it --
kept as a separate module/model/port (see qwen35_config.py) since every
other caller in this repo still wants the 3B model's existing behavior.

Unlike qwen_model_runner.py's single-call run_qwen_inference,
run_qwen35_inference_with_history threads real chat history (past
image+prompt user turns, past raw-text assistant turns) through the model's
own chat template each call -- this is what makes nav.vlm_point_navigator's
periodic re-queries "history-based" the way InternVLA-N1's System-2 is
(Nihil/reference/internvla_dualvln_zenoh_node.py's rolling rgb_list +
conversation_history), rather than the stateless single-frame-in/JSON-out
calls every other qwen_prompts.py prompt uses.
"""

import sys

import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"


def resolve_model_id() -> str:
    """QWEN35_MODEL_ID env var (set by Qwen35ServerManager(model_id=...))
    overrides the Qwen3.5-9B default, e.g. "Qwen/Qwen3-VL-4B-Instruct"."""
    import os

    return os.environ.get("QWEN35_MODEL_ID") or QWEN35_MODEL_ID


def load_qwen35_model(device: str = "cuda") -> tuple:
    model_id = resolve_model_id()
    if model_id == QWEN35_MODEL_ID:
        model_cls = Qwen3_5ForConditionalGeneration
    else:
        # Any other chat VLM (e.g. Qwen3-VL-*-Instruct): let the checkpoint's
        # own config pick the class.
        from transformers import AutoModelForImageTextToText as model_cls
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_qwen35_inference_with_history(
    model,
    processor,
    image: np.ndarray,
    prompt: str,
    history: list[tuple[np.ndarray, str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """`history` is a list of (past_image, past_prompt, past_raw_response_text)
    triples, oldest first -- see nav.vlm_point_navigator.VlmPointNavigator,
    which caps this at num_history entries and clears it whenever the
    active instruction advances (mirrors InternVLA-N1's agent.reset()).
    Each past turn becomes one user (image+text) message followed by one
    assistant (text) message; `image`/`prompt` are appended as the final,
    current-turn user message with no assistant reply yet.

    Sampling (temperature > 0), not greedy decoding: Qwen/Qwen3.5-9B ships
    no generation_config.json, so model.generate() defaults to do_sample=False
    (fully deterministic). Confirmed live 2026-08-24: at one real rover pose,
    a static frame + greedy decoding produced the exact same (u, v) point on
    every repeated call, and that point happened to sit on invalid depth
    (background/out-of-range) -- pixel_to_world always returned None, so the
    rover never moved, so the frame never changed, so the identical failing
    point repeated forever with zero chance of self-correcting. Sampling
    breaks that deadlock: a re-query against the same still-unmoved frame can
    land on a different, groundable point instead of reproducing the exact
    same failure indefinitely.

    A separate do_sample=False (greedy) call for the "is this instruction
    done" judgment alone was tried and reverted -- see
    nav.vlm_point_navigator.VlmPointNavigator.maybe_step's comment: on the
    real R2R-CE benchmark (n=18 episode-runs) it showed no statistically
    significant SR difference from asking for both jointly here (z~1.0),
    while doubling Qwen calls on every advance check. The dominant source
    of run-to-run outcome variance is evidently the compounding effect of
    sampling every driving point, not the terminal advance decision alone."""
    messages = []
    for past_image, past_prompt, past_response in history:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(past_image)},
                    {"type": "text", "text": past_prompt},
                ],
            }
        )
        messages.append({"role": "assistant", "content": past_response})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(image)},
                {"type": "text", "text": prompt},
            ],
        }
    )

    all_images = [Image.fromarray(img) for img, _p, _r in history] + [
        Image.fromarray(image)
    ]

    # enable_thinking=False pre-fills an empty <think></think> block instead
    # of leaving the tag open for the model to fill with extended reasoning
    # -- confirmed live 2026-08-24: with thinking left on, a 512-token
    # response was pure chain-of-thought prose ("The user wants me to
    # identify..."), cut off by max_new_tokens before ever reaching the
    # {"found": ..., "u": ..., ...} JSON qwen35_prompts.py's "respond with
    # ONLY a JSON object, no other text" asks for -- qwen35_response_parser's
    # _load_json_object has no tolerance for leading prose, by design (same
    # as every other parser in this codebase), so this isn't a parsing gap
    # to paper over, the model needs to skip straight to the answer.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        text=[text],
        images=all_images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


if __name__ == "__main__":
    image_path = sys.argv[1]
    test_image = np.array(Image.open(image_path).convert("RGB"))

    model, processor = load_qwen35_model()
    result = run_qwen35_inference_with_history(
        model,
        processor,
        test_image,
        "Describe this image in one sentence.",
        history=[],
    )
    print(result)

"""Prompt builder for Qwen2-VL humanitarian crisis classification.

Provides:
- Zero-shot message construction (no examples).
- Few-shot message construction with text-only demonstrations.
- Few-shot message construction with images embedded per example (multi-turn).
- Few-shot example selection from the training set (1 per class, all 5 classes).

Image injection strategy
------------------------
* Zero-shot / text-only few-shot: image is NOT embedded here; it is injected
  into the LAST user message by ``CrisisClassifier.predict()``.
* Multimodal few-shot: example images are embedded directly into each
  assistant-turn's preceding user message; the query image is still
  injected by ``CrisisClassifier.predict()`` into the final user message.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from datasets import Dataset

# ─────────────────────────────────────────────────────────
# Class metadata
# ─────────────────────────────────────────────────────────

#: Ordered list of the 5 final humanitarian class labels.
LABEL_LIST: list[str] = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

#: Rich descriptions covering both text and visual characteristics.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "affected_individuals": (
        "People who are directly harmed by the crisis: injured, dead, or missing "
        "individuals. Images typically show people in visible distress, wounds, "
        "body bags, medical treatment, or evacuation of casualties. "
        "Tweets mention specific victims, injuries, or fatalities. "
        "NOTE: this class is merged from 'injured_or_dead_people' and 'missing_or_found_people'."
    ),
    "infrastructure_and_utility_damage": (
        "Physical damage to man-made structures and services: collapsed or flooded "
        "buildings, damaged roads/bridges, downed power lines, destroyed vehicles, "
        "disrupted water/gas/electricity supply. Images show structural damage "
        "without necessarily showing people. "
        "NOTE: this class is merged from 'infrastructure_and_utility_damage' and 'vehicle_damage'."
    ),
    "not_humanitarian": (
        "Content that is NOT directly useful for humanitarian response, even if "
        "it mentions a disaster location or event. This includes: opinion pieces, "
        "political commentary, satire, memes, personal photos unrelated to the "
        "crisis, news articles discussing policy, or general social media posts. "
        "Key signal: the tweet expresses an opinion or shares general news rather "
        "than reporting actionable crisis information."
    ),
    "other_relevant_information": (
        "Crisis-related information that does not fit the other categories: "
        "weather forecasts, evacuation orders, shelter locations, road closures, "
        "curfews, status updates, maps, infographics, or general situation reports. "
        "Images often show weather maps, news screenshots, official notices, "
        "or aerial/satellite views. The content is informative but not about "
        "victims, infrastructure damage, or rescue efforts specifically."
    ),
    "rescue_volunteering_or_donation_effort": (
        "Active efforts to help those affected: search-and-rescue operations, "
        "volunteering activities, fundraising campaigns, donation drives, "
        "distribution of supplies, or calls to action. "
        "Images show volunteers, rescue teams, donation centres, fundraising "
        "posters, people distributing aid, or emergency response vehicles in action. "
        "Key text signals: 'donate', 'volunteer', 'help', 'fundraise', 'supplies'."
    ),
}

#: Shared decision guide appended to the system prompt.
_DECISION_GUIDE: str = """
DECISION GUIDE for ambiguous cases:
  • Tweet discusses a disaster topic as news/opinion/commentary → not_humanitarian
  • Image shows physical damage to structures or objects (no people) → infrastructure_and_utility_damage
  • Image shows people visibly injured, dead, or in distress → affected_individuals
  • Content provides situational info (weather, shelter, maps, status) → other_relevant_information
  • Content calls for or shows active helping/donating/rescuing → rescue_volunteering_or_donation_effort
"""

#: System prompt shared by zero-shot and few-shot settings.
SYSTEM_PROMPT: str = (
    "You are an expert humanitarian-crisis analyst.\n"
    "Given a tweet and its associated image, classify the content into exactly "
    "ONE of the following 5 categories:\n\n"
    + "\n".join(
        f"  [{i+1}] {lbl}:\n      {desc}"
        for i, (lbl, desc) in enumerate(CLASS_DESCRIPTIONS.items())
    )
    + _DECISION_GUIDE
    + "\nIMPORTANT: Respond with ONLY the exact category name, nothing else.\n"
    "Do NOT include explanations, numbers, or punctuation in your response."
)


# ─────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────

def build_zero_shot_messages(tweet_text: str) -> list[dict]:
    """Build a zero-shot message list for a single tweet (no examples).

    The query image is injected into the final user message by
    ``CrisisClassifier.predict()``.

    Args:
        tweet_text: The tweet text to classify.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts for the chat template.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify the following tweet and its associated image.\n\n"
                f"Tweet: {tweet_text}\n\n"
                "Category:"
            ),
        },
    ]


def build_few_shot_messages(
    tweet_text: str,
    examples: list[dict],
    n_shots: int = 5,
) -> list[dict]:
    """Build a few-shot message list with **text-only** demonstration examples.

    Images for the examples are omitted. Only the query tweet's image is
    injected at inference time.  Prefer ``build_few_shot_messages_with_images``
    for better performance.

    Args:
        tweet_text: The query tweet text to classify.
        examples:   List of dicts with keys ``'tweet_text'`` and ``'label'``.
        n_shots:    Number of demonstrations to include.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts.
    """
    examples_block = "Here are labelled examples (one per class) to guide you:\n\n"
    for i, ex in enumerate(examples[:n_shots], start=1):
        examples_block += (
            f"Example {i} — Category: {ex['label']}\n"
            f"  Tweet: {ex['tweet_text']}\n\n"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{examples_block}"
                "Now classify the following tweet and its associated image.\n\n"
                f"Tweet: {tweet_text}\n\n"
                "Category:"
            ),
        },
    ]


def build_few_shot_messages_with_images(
    tweet_text: str,
    examples: list[dict],
) -> list[dict]:
    """Build a multimodal few-shot message list using multi-turn dialogue.

    Each example becomes its own user->assistant turn, with the example image
    embedded directly in the user message.  The query (last user message) is
    left as plain text so that ``CrisisClassifier.predict()`` can inject the
    query image into it.

    Message structure::

        system
        user:  [image_1] Tweet: ex1_text  Category:
        assistant: label_1
        user:  [image_2] Tweet: ex2_text  Category:
        assistant: label_2
        ...
        user:  Tweet: query_text  Category:   <- query image injected by predict()

    Args:
        tweet_text: The query tweet text to classify.
        examples:   List of dicts with keys ``'tweet_text'``, ``'label'``, and
                    optionally ``'image'`` (a ``PIL.Image`` object).

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts for the chat template.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in examples:
        pil_image = ex.get("image")

        if pil_image is not None:
            user_content = [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": (
                        f"Tweet: {ex['tweet_text']}\n\n"
                        "Category:"
                    ),
                },
            ]
        else:
            # Graceful fallback if image could not be loaded
            user_content = (
                f"Tweet: {ex['tweet_text']}\n\n"
                "Category:"
            )

        messages.append({"role": "user",      "content": user_content})
        messages.append({"role": "assistant", "content": ex["label"]})

    # Query — image will be injected by CrisisClassifier.predict()
    messages.append({
        "role": "user",
        "content": (
            "Now classify the following tweet and its associated image.\n\n"
            f"Tweet: {tweet_text}\n\n"
            "Category:"
        ),
    })
    return messages


# ─────────────────────────────────────────────────────────
# Few-shot example selection
# ─────────────────────────────────────────────────────────

def select_few_shot_examples(
    train_dataset: Dataset,
    label_col: str = "label",
    text_col: str = "tweet_text",
    image_path_col: str = "image_path",
    seed: int = 42,
) -> list[dict]:
    """Select exactly **1 example per class** covering all 5 final labels.

    Strategy: for each class in ``LABEL_LIST``, pick one random sample from
    the training split.  The returned dicts include ``image_path`` so that the
    caller can load the PIL image if needed.

    Args:
        train_dataset:   The training ``Dataset`` (after label merging).
        label_col:       Column name for labels.
        text_col:        Column name for tweet text.
        image_path_col:  Column name for relative image file paths.
        seed:            Random seed for reproducibility.

    Returns:
        A list of 5 dicts, each with keys:
        ``'tweet_text'``, ``'label'``, ``'image_path'``.
        (Add ``'image'`` key externally after loading from disk.)
    """
    np.random.seed(seed)

    labels: list[str]    = train_dataset[label_col]
    texts:  list[str]    = train_dataset[text_col]
    img_paths: list[str] = train_dataset[image_path_col]

    # Build class -> index mapping
    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    examples: list[dict] = []
    for lbl in LABEL_LIST:
        if lbl not in class_indices:
            continue
        idx = int(np.random.choice(class_indices[lbl]))
        examples.append({
            "tweet_text": texts[idx],
            "label":      lbl,
            "image_path": img_paths[idx],
        })

    return examples


# ─────────────────────────────────────────────────────────
# Chain-of-thought (CoT) helpers  [Improvement A]
# ─────────────────────────────────────────────────────────

#: Suffix appended to the QUERY user message to elicit step-by-step reasoning.
#: The model is asked to reason first, then emit a ``LABEL: <category>`` line
#: which ``CrisisClassifier._parse_label()`` knows how to extract.
_COT_QUERY_SUFFIX: str = (
    "\n\nBefore deciding, think briefly:\n"
    "1. What does the image show? (one sentence)\n"
    "2. What is the tweet reporting? (one sentence)\n"
    "3. Which category fits best according to the decision guide?\n\n"
    "Then output on a new line exactly:\n"
    "LABEL: <exact_category_name>"
)


def build_zero_shot_cot_messages(tweet_text: str) -> list[dict]:
    """Build a zero-shot chain-of-thought message list.

    Asks the model to reason step-by-step before committing to a label.
    The predicted label is expected after a ``LABEL:`` marker so that
    ``CrisisClassifier._parse_label()`` can extract it reliably.
    The query image is injected into the final user message by
    ``CrisisClassifier.predict()``.

    Args:
        tweet_text: The tweet text to classify.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify the following tweet and its associated image.\n\n"
                f"Tweet: {tweet_text}"
                + _COT_QUERY_SUFFIX
            ),
        },
    ]


def build_few_shot_cot_messages_with_images(
    tweet_text: str,
    examples: list[dict],
) -> list[dict]:
    """Build a multimodal few-shot CoT message list (multi-turn with images).

    Demo assistant turns use the ``LABEL: <category>`` format so the model
    learns the expected output structure.  The final user query appends the
    full CoT reasoning instruction.

    Message structure::

        system
        user:      [image_1]  Tweet: ex1_tweet   Category:
        assistant: LABEL: label_1
        ...
        user:      [image_n]  Tweet: exn_tweet   Category:
        assistant: LABEL: label_n
        user:      Tweet: query_tweet  <CoT instruction>   <- image injected by predict()

    Args:
        tweet_text: The query tweet text to classify.
        examples:   List of dicts with ``'tweet_text'``, ``'label'``, and
                    optionally ``'image'`` (a ``PIL.Image`` object).

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in examples:
        pil_image = ex.get("image")
        if pil_image is not None:
            user_content = [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": f"Tweet: {ex['tweet_text']}\n\nCategory:"},
            ]
        else:
            user_content = f"Tweet: {ex['tweet_text']}\n\nCategory:"

        messages.append({"role": "user",      "content": user_content})
        # Demo label uses LABEL: prefix — teaches the model the output format
        messages.append({"role": "assistant", "content": f"LABEL: {ex['label']}"})

    # Query with CoT instruction; image injected by CrisisClassifier.predict()
    messages.append({
        "role": "user",
        "content": (
            "Now classify the following tweet and its associated image.\n\n"
            f"Tweet: {tweet_text}"
            + _COT_QUERY_SUFFIX
        ),
    })
    return messages


# ─────────────────────────────────────────────────────────
# Weighted few-shot example selection  [Improvement B]
# ─────────────────────────────────────────────────────────

def select_few_shot_examples_weighted(
    train_dataset: Dataset,
    label_col: str = "label",
    text_col: str = "tweet_text",
    image_path_col: str = "image_path",
    seed: int = 42,
) -> list[dict]:
    """Select few-shot examples with extra coverage for hard / minority classes.

    Hard classes — ``affected_individuals`` and ``other_relevant_information``
    — each receive **2 examples**; the remaining three classes get **1 example**
    each, giving **7 demonstrations** in total.

    Rationale (from Phase 1 error analysis):
    - ``affected_individuals``: only 5% of eval set, worst F1 across all configs.
    - ``other_relevant_information``: catch-all class with blurry boundaries;
      F1 = 0.000 in both zero-shot experiments.

    Args:
        train_dataset:   Training ``Dataset`` (after label merging).
        label_col:       Column name for labels.
        text_col:        Column name for tweet text.
        image_path_col:  Column name for relative image file paths.
        seed:            Random seed for reproducibility.

    Returns:
        A list of 7 dicts, each with keys:
        ``'tweet_text'``, ``'label'``, ``'image_path'``.
        (Add ``'image'`` key externally after loading from disk.)
    """
    np.random.seed(seed)

    #: Classes that receive 2 examples due to high error rates in Phase 1.
    _HARD_CLASSES: set[str] = {"affected_individuals", "other_relevant_information"}

    labels:    list[str] = train_dataset[label_col]
    texts:     list[str] = train_dataset[text_col]
    img_paths: list[str] = train_dataset[image_path_col]

    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    examples: list[dict] = []
    for lbl in LABEL_LIST:
        if lbl not in class_indices:
            continue
        n = 2 if lbl in _HARD_CLASSES else 1
        pool = class_indices[lbl]
        chosen_idxs = np.random.choice(
            pool, size=min(n, len(pool)), replace=False
        )
        for idx in chosen_idxs:
            examples.append({
                "tweet_text": texts[int(idx)],
                "label":      lbl,
                "image_path": img_paths[int(idx)],
            })

    return examples

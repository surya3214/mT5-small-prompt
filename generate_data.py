"""Generate synthetic slot-extraction dataset for the prompt impact benchmark.

Produces three JSON splits in data/:
  - train.json      (1200 examples, in-distribution entities)
  - test_id.json    (200 examples, in-distribution entities, no input overlap)
  - test_ood.json   (200 examples, out-of-distribution cities and/or item types)
"""

import json
import os
import random

SEED = 42

# ── Entity pools ──────────────────────────────────────────────────────────────

TRAIN_CITIES = [
    "Bangalore", "Tokyo", "Berlin", "Mumbai", "London", "Paris",
    "New York", "Sydney", "Singapore", "Dubai", "Chennai", "Delhi",
    "Shanghai", "Seoul", "Moscow", "Rome", "Madrid", "Cairo",
    "Bangkok", "Jakarta",
]

OOD_CITIES = [
    "Heidelberg", "Montevideo", "Reykjavik", "Tbilisi", "Vladivostok",
    "Marrakech", "Christchurch", "Porto", "Bruges", "Tallinn",
]

TRAIN_ITEMS = [
    "photos", "videos", "contacts", "notes", "documents",
    "files", "messages", "emails", "bookmarks", "reminders",
]

OOD_ITEMS = ["memos", "invoices", "receipts", "podcasts", "sketches"]

ACTIONS = [
    "fetch", "delete", "upload", "share", "search",
    "download", "backup", "sync", "export", "archive",
]

TIME_PERIODS = [
    "last week", "yesterday", "today", "last month",
    "this morning", "last year", "two days ago", "this week",
]

# ── Templates ─────────────────────────────────────────────────────────────────
# (template_string, implicit_action | None)

# Without time field
TEMPLATES_3 = [
    # Explicit action (contain {act})
    ("{act} {record} from {add}", None),
    ("{act} all {record} in {add}", None),
    ("{act} {record} taken in {add}", None),
    ("{act} my {record} from {add}", None),
    ("please {act} {record} from {add}", None),
    ("I want to {act} {record} from {add}", None),
    # Implicit action (verb → action mapping)
    ("find {record} from {add}", "fetch"),
    ("get {record} from {add}", "fetch"),
    ("show me {record} from {add}", "fetch"),
    ("pull {record} from {add}", "fetch"),
    ("retrieve {record} from {add}", "fetch"),
    ("look up {record} near {add}", "search"),
    ("locate {record} in {add}", "search"),
    ("remove {record} from {add}", "delete"),
    ("erase {record} from {add}", "delete"),
]

# With time field
TEMPLATES_4 = [
    ("{act} {record} from {add} {time}", None),
    ("{act} all {record} in {add} since {time}", None),
    ("find {record} from {add} from {time}", "fetch"),
    ("show me {record} from {add}, {time}", "fetch"),
    ("get {record} in {add} from {time}", "fetch"),
    ("retrieve {record} from {add} since {time}", "fetch"),
    ("remove {record} from {add} from {time}", "delete"),
]


def _generate_pool(cities, items, rng, n, seen_inputs=None):
    """Generate n unique examples from given entity pools."""
    if seen_inputs is None:
        seen_inputs = set()
    examples = []
    max_attempts = n * 50

    for _ in range(max_attempts):
        if len(examples) >= n:
            break

        use_time = rng.random() < 0.3
        city = rng.choice(cities)
        item = rng.choice(items)

        if use_time:
            template, implicit_act = rng.choice(TEMPLATES_4)
            time_val = rng.choice(TIME_PERIODS)
        else:
            template, implicit_act = rng.choice(TEMPLATES_3)
            time_val = None

        if implicit_act:
            act = implicit_act
            fmt = {"record": item, "add": city}
        else:
            act = rng.choice(ACTIONS)
            fmt = {"act": act, "record": item, "add": city}

        if time_val is not None:
            fmt["time"] = time_val

        input_text = template.format(**fmt)

        if input_text in seen_inputs:
            continue
        seen_inputs.add(input_text)

        target = {"act": act, "add": city, "record": item}
        if time_val is not None:
            target["time"] = time_val

        examples.append({"input": input_text, "target": target})

    return examples


def generate_dataset(output_dir="data"):
    """Generate train / ID-test / OOD-test splits and write to output_dir."""
    rng = random.Random(SEED)
    os.makedirs(output_dir, exist_ok=True)

    # Shared set prevents input overlap between train and ID test
    seen = set()

    train_data = _generate_pool(TRAIN_CITIES, TRAIN_ITEMS, rng, 600, seen)
    id_test_data = _generate_pool(TRAIN_CITIES, TRAIN_ITEMS, rng, 200, seen)

    # OOD test: unseen entities along 3 axes
    ood_seen = set()
    ood_new_city = _generate_pool(OOD_CITIES, TRAIN_ITEMS, rng, 80, ood_seen)
    ood_new_item = _generate_pool(TRAIN_CITIES, OOD_ITEMS, rng, 60, ood_seen)
    ood_both = _generate_pool(OOD_CITIES, OOD_ITEMS, rng, 60, ood_seen)
    ood_test_data = ood_new_city + ood_new_item + ood_both

    for name, data in [
        ("train", train_data),
        ("test_id", id_test_data),
        ("test_ood", ood_test_data),
    ]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data)} examples → {path}")


if __name__ == "__main__":
    generate_dataset()

import os
import json
import time
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import HashingVectorizer


# =========================
# Config + LLM Client
# =========================

@dataclass
class LLMConfig:
    base_url: str                 # OpenAI-compatible endpoint, e.g. "http://localhost:8000/v1"
    api_key: str                  # can be empty for local
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: int = 60


class OpenAICompatibleLLM:
    """
    Minimal OpenAI-compatible Chat Completions client:
    POST {base_url}/chat/completions
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


# =========================
# Data I/O
# =========================

def read_jsonl_dataset(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # expected keys: "input", "label"
    return rows


def sample_dataset(rows: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if len(rows) <= n:
        return rows[:]
    return rng.sample(rows, n)


# =========================
# Prompt Space: Template + Slots
# =========================

@dataclass
class PromptSpace:
    template: str                        # contains placeholders like {FORMAT}, {STRICTNESS}
    slots: Dict[str, List[str]]          # slot_name -> list of discrete options
    role_name: str = "role"


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction: finds the first {...} block and tries to parse.
    """
    text = text.strip()
    # If pure JSON, parse directly
    try:
        return json.loads(text)
    except Exception:
        pass

    # Otherwise try to locate first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start:end+1]
        return json.loads(blob)

    raise ValueError("Could not extract JSON from LLM response.")


def auto_slotify_prompt_with_llm(
    base_prompt: str,
    dataset_sample: List[Dict[str, Any]],
    designer_llm: OpenAICompatibleLLM,
    role_name: str,
    max_slots: int = 8,
    max_options_per_slot: int = 5
) -> PromptSpace:
    """
    Uses LLM to convert a base prompt into:
    - a slot-based template
    - discrete slot options (small sets)
    """
    examples = [
        {"input": r.get("input", ""), "label": r.get("label", "")}
        for r in dataset_sample[:6]
    ]

    system = (
        "You are a prompt-optimization assistant.\n"
        "You will receive:\n"
        "1) A base role prompt.\n"
        "2) A few labeled dataset examples.\n\n"
        "Task:\n"
        "- Convert the base prompt into a TEMPLATE with {SLOT_NAMES} placeholders.\n"
        "- Propose discrete OPTIONS for each slot (small sets).\n"
        "- Keep slot count small and useful.\n"
        "- Output JSON only, matching this schema:\n"
        "{\n"
        '  "template": "string with {SLOT} placeholders",\n'
        '  "slots": { "SLOT_NAME": ["opt1","opt2",...], ... }\n'
        "}\n"
        "Constraints:\n"
        f"- At most {max_slots} slots.\n"
        f"- At most {max_options_per_slot} options per slot.\n"
        "- Slot names must be UPPER_SNAKE_CASE.\n"
        "- Ensure template still clearly defines the role and requirements.\n"
    )

    user = {
        "role": "user",
        "content": json.dumps({
            "role_name": role_name,
            "base_prompt": base_prompt,
            "dataset_examples": examples
        }, ensure_ascii=False, indent=2)
    }

    out = designer_llm.chat([{"role": "system", "content": system}, user])
    spec = _extract_json_from_text(out)

    template = spec["template"]
    slots = spec["slots"]

    # Light validation
    if not isinstance(template, str) or not isinstance(slots, dict) or len(slots) == 0:
        raise ValueError("Invalid slotification output.")

    return PromptSpace(template=template, slots=slots, role_name=role_name)


def render_prompt(template: str, slot_values: Dict[str, str]) -> str:
    """
    Replace {SLOT} placeholders with the chosen option.
    """
    prompt = template
    for k, v in slot_values.items():
        prompt = prompt.replace("{" + k + "}", v)
    return prompt


# =========================
# State Encoder (cheap + stable)
# =========================

class StateEncoder:
    """
    Turns input text into a fixed numeric vector for the policy.
    Uses HashingVectorizer so no fitting is needed.
    """
    def __init__(self, n_features: int = 1024):
        self.vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")

    def encode(self, x: str) -> torch.Tensor:
        X = self.vec.transform([x])                 # sparse
        arr = X.toarray().astype("float32")[0]
        return torch.from_numpy(arr)


# =========================
# Policy Network (multi-slot categorical)
# =========================

class SlotPolicy(nn.Module):
    """
    Independent categorical distribution per slot:
    π(a|s) = Π_j Cat(a_j | logits_j(s))
    """
    def __init__(self, state_dim: int, slot_sizes: Dict[str, int], hidden: int = 256):
        super().__init__()
        self.slot_names = list(slot_sizes.keys())
        self.slot_sizes = slot_sizes

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, slot_sizes[name])
            for name in self.slot_names
        })

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(s)
        logits = {name: self.heads[name](h) for name in self.slot_names}
        return logits

    def sample_action(self, s: torch.Tensor) -> Tuple[Dict[str, int], torch.Tensor]:
        logits = self.forward(s)
        chosen = {}
        logp_sum = torch.tensor(0.0)

        for name, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            chosen[name] = int(a.item())
            logp_sum = logp_sum + dist.log_prob(a)

        return chosen, logp_sum

    def greedy_action(self, s: torch.Tensor) -> Dict[str, int]:
        logits = self.forward(s)
        chosen = {}
        for name, lg in logits.items():
            chosen[name] = int(torch.argmax(lg).item())
        return chosen


# =========================
# LLM-based Judge (independent function)
# =========================

def llm_judge_score(
    judge_llm: OpenAICompatibleLLM,
    x: str,
    y_pred: str,
    y_true: str,
    rubric_name: str = "general",
) -> Dict[str, Any]:
    """
    Returns dict like:
    { "score": 0.0..1.0, "format_ok": true/false, "hallucination": true/false, "notes": "..." }
    """
    system = (
        "You are a strict evaluator.\n"
        "You will be given:\n"
        "- input\n"
        "- model_output\n"
        "- ground_truth\n\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "score": number between 0 and 1,\n'
        '  "format_ok": boolean,\n'
        '  "hallucination": boolean,\n'
        '  "notes": string\n'
        "}\n"
        "Scoring guidance:\n"
        "- score=1: correct and appropriately complete.\n"
        "- score near 0: wrong or irrelevant.\n"
        "- hallucination=true if it asserts specific facts not supported by ground_truth.\n"
        "- format_ok=false if it ignores required structure or is unusable.\n"
        f"Rubric: {rubric_name}\n"
    )

    user = {
        "role": "user",
        "content": json.dumps({
            "input": x,
            "model_output": y_pred,
            "ground_truth": y_true
        }, ensure_ascii=False)
    }

    out = judge_llm.chat([{"role": "system", "content": system}, user])
    return _extract_json_from_text(out)


# =========================
# Ground-truth Metric (simple default)
# =========================

def simple_label_score(y_pred: str, y_true: str) -> float:
    """
    Default label score:
    - exact match -> 1
    - substring match -> 0.6
    - else -> 0
    You should replace with your proper metric for your task.
    """
    yp = (y_pred or "").strip().lower()
    yt = (y_true or "").strip().lower()
    if not yt:
        return 0.0
    if yp == yt:
        return 1.0
    if yt in yp:
        return 0.6
    return 0.0


# =========================
# Reward function (labels + judge + cost penalties)
# =========================

def compute_reward(
    y_pred: str,
    y_true: str,
    judge: Dict[str, Any],
    tokens_proxy: int,
    alpha: float = 0.7,
    lambda_cost: float = 0.001,
    beta_hall: float = 0.4,
    gamma_format: float = 0.2
) -> float:
    r_gt = simple_label_score(y_pred, y_true)

    r_j = float(judge.get("score", 0.0))
    halluc = bool(judge.get("hallucination", False))
    fmt_ok = bool(judge.get("format_ok", True))

    r = alpha * r_gt + (1 - alpha) * r_j
    r -= lambda_cost * float(tokens_proxy)
    if halluc:
        r -= beta_hall
    if not fmt_ok:
        r -= gamma_format
    return float(r)


def approx_token_count(text: str) -> int:
    # rough proxy: ~4 chars per token
    return max(1, len(text) // 4)


# =========================
# Caching (so judge calls don’t explode)
# =========================

class SimpleDiskCache:
    def __init__(self, path: str):
        self.path = path
        self.map = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.map[obj["key"]] = obj["value"]

    def _key(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.map.get(self._key(payload))

    def put(self, payload: Dict[str, Any], value: Dict[str, Any]) -> None:
        k = self._key(payload)
        if k in self.map:
            return
        self.map[k] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "value": value}, ensure_ascii=False) + "\n")


# =========================
# Training loop (REINFORCE)
# =========================

def train_slot_policy(
    prompt_space: PromptSpace,
    train_rows: List[Dict[str, Any]],
    target_llm: OpenAICompatibleLLM,
    judge_llm: OpenAICompatibleLLM,
    out_policy_path: str,
    epochs: int = 1,
    lr: float = 3e-4,
    state_dim: int = 1024,
    hidden: int = 256,
    alpha: float = 0.7,
    cache_path: str = "judge_cache.jsonl",
    seed: int = 123
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    # Build policy
    slot_sizes = {k: len(v) for k, v in prompt_space.slots.items()}
    encoder = StateEncoder(n_features=state_dim)
    policy = SlotPolicy(state_dim=state_dim, slot_sizes=slot_sizes, hidden=hidden)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    cache = SimpleDiskCache(cache_path)

    # Baseline to reduce variance
    baseline = 0.0
    baseline_momentum = 0.9

    policy.train()
    step = 0

    for ep in range(epochs):
        random.shuffle(train_rows)
        for row in train_rows:
            step += 1
            x = str(row.get("input", ""))
            y_true = str(row.get("label", ""))

            s = encoder.encode(x)

            # Sample action (slot indices)
            a_idx, logp = policy.sample_action(s)

            # Convert indices to actual slot values
            slot_values = {}
            for slot_name, idx in a_idx.items():
                slot_values[slot_name] = prompt_space.slots[slot_name][idx]

            prompt = render_prompt(prompt_space.template, slot_values)

            # Run target LLM with the generated prompt
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": x}
            ]
            y_pred = target_llm.chat(messages)

            # Judge (cached)
            judge_payload = {"x": x, "y_pred": y_pred, "y_true": y_true, "role": prompt_space.role_name}
            judge = cache.get(judge_payload)
            if judge is None:
                judge = llm_judge_score(judge_llm, x=x, y_pred=y_pred, y_true=y_true, rubric_name=prompt_space.role_name)
                cache.put(judge_payload, judge)

            # Reward
            tokens_proxy = approx_token_count(prompt) + approx_token_count(x) + approx_token_count(y_pred)
            r = compute_reward(
                y_pred=y_pred,
                y_true=y_true,
                judge=judge,
                tokens_proxy=tokens_proxy,
                alpha=alpha,
            )

            # Advantage
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * r
            adv = r - baseline

            # REINFORCE update: maximize E[r] => minimize -r*logp
            loss = -(adv * logp)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                print(f"[ep={ep+1}/{epochs} step={step}] r={r:.3f} baseline={baseline:.3f} adv={adv:.3f} loss={loss.item():.3f}")

    # Save
    torch.save({
        "prompt_space": {
            "template": prompt_space.template,
            "slots": prompt_space.slots,
            "role_name": prompt_space.role_name
        },
        "state_dim": state_dim,
        "hidden": hidden,
        "policy_state_dict": policy.state_dict()
    }, out_policy_path)

    print(f"Saved policy to: {out_policy_path}")


# =========================
# Inference
# =========================

def load_policy_bundle(path: str) -> Tuple[PromptSpace, SlotPolicy, StateEncoder]:
    blob = torch.load(path, map_location="cpu")
    ps = PromptSpace(
        template=blob["prompt_space"]["template"],
        slots=blob["prompt_space"]["slots"],
        role_name=blob["prompt_space"]["role_name"],
    )
    state_dim = int(blob["state_dim"])
    hidden = int(blob["hidden"])

    slot_sizes = {k: len(v) for k, v in ps.slots.items()}
    policy = SlotPolicy(state_dim=state_dim, slot_sizes=slot_sizes, hidden=hidden)
    policy.load_state_dict(blob["policy_state_dict"])
    policy.eval()

    encoder = StateEncoder(n_features=state_dim)
    return ps, policy, encoder


def infer_with_policy(
    policy_path: str,
    target_llm: OpenAICompatibleLLM,
    user_input: str,
    greedy: bool = True
) -> Dict[str, Any]:
    ps, policy, encoder = load_policy_bundle(policy_path)
    s = encoder.encode(user_input)

    if greedy:
        a_idx = policy.greedy_action(s)
    else:
        a_idx, _ = policy.sample_action(s)

    slot_values = {k: ps.slots[k][idx] for k, idx in a_idx.items()}
    prompt = render_prompt(ps.template, slot_values)

    y = target_llm.chat([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ])

    return {
        "chosen_slots": slot_values,
        "rendered_prompt": prompt,
        "output": y
    }


# =========================
# Example "main" usage
# =========================

def main():
    # ---- User config (edit these) ----
    target_cfg = LLMConfig(
        base_url=os.getenv("TARGET_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("TARGET_API_KEY", ""),
        model=os.getenv("TARGET_MODEL", "your-target-model"),
        temperature=0.2,
        max_tokens=512
    )
    judge_cfg = LLMConfig(
        base_url=os.getenv("JUDGE_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("JUDGE_API_KEY", ""),
        model=os.getenv("JUDGE_MODEL", "your-judge-model"),
        temperature=0.0,
        max_tokens=256
    )
    designer_cfg = LLMConfig(
        base_url=os.getenv("DESIGNER_BASE_URL", judge_cfg.base_url),
        api_key=os.getenv("DESIGNER_API_KEY", judge_cfg.api_key),
        model=os.getenv("DESIGNER_MODEL", judge_cfg.model),
        temperature=0.2,
        max_tokens=700
    )

    target_llm = OpenAICompatibleLLM(target_cfg)
    judge_llm = OpenAICompatibleLLM(judge_cfg)
    designer_llm = OpenAICompatibleLLM(designer_cfg)

    # ---- Load data ----
    rows = read_jsonl_dataset("data.jsonl")
    sample = sample_dataset(rows, n=12)

    # ---- Your base prompt (example) ----
    base_prompt = """You are an expert in providing comprehensive answers. Your job is to:
1. For each interpretation or aspect of a question, provide the most obvious and direct answer when possible.
2. If there are countless reasonable answers or if the question admits endless responses, say so directly instead of attempting to list them.
3. Evaluate whether the provided information is sufficient to answer conclusively; if not, state that a definitive answer is impossible.
4. Consider multiple perspectives as appropriate, but avoid overly formal or mathematical explanations unless specifically relevant or requested.
5. Ensure answers are clear, natural, and well-reasoned, offering supporting context as needed.
"""

    # ---- Auto-create template+slots from data sample ----
    prompt_space = auto_slotify_prompt_with_llm(
        base_prompt=base_prompt,
        dataset_sample=sample,
        designer_llm=designer_llm,
        role_name="answerer",
        max_slots=8,
        max_options_per_slot=5
    )
    print("\n=== Auto-designed Prompt Space ===")
    print(prompt_space.template)
    print(json.dumps(prompt_space.slots, indent=2, ensure_ascii=False))

    # ---- Train RL policy ----
    train_slot_policy(
        prompt_space=prompt_space,
        train_rows=rows[:200],              # start small
        target_llm=target_llm,
        judge_llm=judge_llm,
        out_policy_path="policy.pt",
        epochs=1,
        lr=3e-4,
        state_dim=1024,
        hidden=256,
        alpha=0.7,
        cache_path="judge_cache.jsonl",
    )

    # ---- Inference demo ----
    result = infer_with_policy(
        policy_path="policy.pt",
        target_llm=target_llm,
        user_input="Is this question ambiguous: 'Is it big?' Explain.",
        greedy=True
    )
    print("\n=== Inference ===")
    print("Chosen slots:", result["chosen_slots"])
    print("Output:\n", result["output"])


if __name__ == "__main__":
    main()

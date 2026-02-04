from rl_prompt_slots_pipeline import llm_judge_score, approx_token_count, compute_reward
def evaluate_baseline(
    rows,
    target_llm,
    judge_llm,
    base_prompt,
    max_samples=200
):
    scores = []
    rewards = []

    for row in rows[:max_samples]:
        x = row["input"]
        y_true = row["label"]

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": x},
        ]

        y_pred = target_llm.chat(messages)

        judge = llm_judge_score(
            judge_llm,
            x=x,
            y_pred=y_pred,
            y_true=y_true,
            rubric_name="baseline"
        )

        tokens_proxy = approx_token_count(base_prompt) \
                       + approx_token_count(x) \
                       + approx_token_count(y_pred)

        r = compute_reward(
            y_pred=y_pred,
            y_true=y_true,
            judge=judge,
            tokens_proxy=tokens_proxy,
            alpha=0.2
        )

        scores.append(judge["score"])
        rewards.append(r)

    return {
        "avg_judge_score": sum(scores) / len(scores),
        "avg_reward": sum(rewards) / len(rewards)
    }

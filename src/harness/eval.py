"""Substring/EM and LongMemEval branched-judge evaluators.

The substring/EM logic and LongMemEval prompts are ported VERBATIM from
HUST-AI-HYZ/MemoryAgentBench (`utils/eval_other_utils.py`,
`llm_based_eval/longmem_qa_evaluate.py`) so harness scores are directly
comparable to the paper's headline numbers. Do not modify the prompt
strings.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

from harness.reader import judge_call

# ----- substring / EM (port of utils/eval_other_utils.py) ------------------


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def substring_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def parse_output(output_text: str, answer_prefix: str = "Answer:") -> str | None:
    patterns = [
        re.compile(rf"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(output_text)
        if match:
            extracted = match.group(1).strip()
            cleaned = re.sub(
                rf"^{re.escape(answer_prefix)}", "", extracted, flags=re.IGNORECASE
            ).strip()
            return cleaned
    return None


def _flatten(answers: object) -> list[str]:
    if isinstance(answers, str):
        return [answers]
    if isinstance(answers, Iterable):
        out: list[str] = []
        for a in answers:
            if isinstance(a, str):
                out.append(a)
            elif isinstance(a, Iterable):
                out.extend(str(x) for x in a)
            else:
                out.append(str(a))
        return out
    return [str(answers)]


def score_substring(model_output: str, answers: object, sub_dataset: str = "") -> float:
    """Return a 0/1 score for substring/EM-style tasks.

    ruler_qa, factconsolidation, ICL, detective_qa: max over ground truths of
        substring_match(parsed_output, gt) OR substring_match(raw_output, gt).
    eventqa: binary recall — 1 iff every answer element appears (case-
        insensitive) in raw output.
    """
    gts = _flatten(answers)
    raw = model_output or ""
    parsed = parse_output(raw) or ""

    if "eventqa" in sub_dataset.lower():
        if not gts:
            return 0.0
        recall = sum(1 for el in gts if el.lower() in raw.lower()) / len(gts)
        return 1.0 if recall == 1 else 0.0

    for gt in gts:
        if substring_match(parsed, gt) or substring_match(raw, gt):
            return 1.0
    return 0.0


# ----- LongMemEval branched judge (port of llm_based_eval/longmem_qa_evaluate.py) -----

_BASE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: {}"
    "\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
    "or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the "
    "model correctly identify the question as unanswerable? Answer yes or no only."
)


def get_anscheck_prompt(
    question_type: str | None,
    question: str,
    answer: object,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    answer_str = answer if isinstance(answer, str) else str(answer)
    if abstention:
        return _ABSTENTION_TEMPLATE.format(question, answer_str, response)
    if question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        return _BASE_TEMPLATE.format(question, answer_str, response)
    if question_type == "temporal-reasoning":
        return _TEMPORAL_TEMPLATE.format(question, answer_str, response)
    if question_type == "knowledge-update":
        return _KNOWLEDGE_UPDATE_TEMPLATE.format(question, answer_str, response)
    if question_type == "single-session-preference":
        return _PREFERENCE_TEMPLATE.format(question, answer_str, response)
    raise NotImplementedError(f"Unknown LongMemEval question_type: {question_type!r}")


def score_longmem_judge(
    model_output: str,
    *,
    question: str,
    answers: object,
    question_type: str | None,
    question_id: str | None,
) -> tuple[float, str, float]:
    """Return (score, judge_raw, judge_cost_usd)."""
    abstention = bool(question_id and "_abs" in question_id)
    answer = answers[0] if isinstance(answers, list) and answers else answers
    prompt = get_anscheck_prompt(
        question_type, question, answer, model_output, abstention=abstention
    )
    res = judge_call(prompt)
    label = 1.0 if "yes" in res.text.lower() else 0.0
    return label, res.text, res.cost_usd

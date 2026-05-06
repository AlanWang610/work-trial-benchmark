from typing import Literal, TypedDict

EvalKind = Literal["substring", "longmem_judge"]


class TaskConfig(TypedDict):
    split: str
    eval: EvalKind
    gen_max: int
    samples: int
    sd_filter: str
    stratify_on: str | None


TASKS: dict[str, TaskConfig] = {
    "ruler_qa_sh": {
        "split": "Accurate_Retrieval",
        "eval": "substring",
        "gen_max": 50,
        "samples": 12,
        "sd_filter": "ruler_qa1_197K",
        "stratify_on": None,
    },
    "longmemeval_s": {
        "split": "Accurate_Retrieval",
        "eval": "longmem_judge",
        "gen_max": 50,
        "samples": 18,
        "sd_filter": "longmemeval_s*",
        "stratify_on": "question_type",
    },
    "eventqa_64k": {
        "split": "Accurate_Retrieval",
        "eval": "substring",
        "gen_max": 40,
        "samples": 5,
        "sd_filter": "eventqa_65536",
        "stratify_on": None,
    },
    "detective_qa": {
        "split": "Long_Range_Understanding",
        "eval": "substring",
        "gen_max": 50,
        "samples": 5,
        "sd_filter": "detective_qa",
        "stratify_on": None,
    },
    "icl_trec_coarse": {
        "split": "Test_Time_Learning",
        "eval": "substring",
        "gen_max": 20,
        "samples": 12,
        "sd_filter": "icl_trec_coarse_6600shot_balance",
        "stratify_on": None,
    },
    "factconsolidation_sh_6k": {
        "split": "Conflict_Resolution",
        "eval": "substring",
        "gen_max": 10,
        "samples": 12,
        "sd_filter": "factconsolidation_sh_6k",
        "stratify_on": None,
    },
    "factconsolidation_mh_6k": {
        "split": "Conflict_Resolution",
        "eval": "substring",
        "gen_max": 10,
        "samples": 12,
        "sd_filter": "factconsolidation_mh_6k",
        "stratify_on": None,
    },
}

TOTAL_SUBSET_SIZE = sum(t["samples"] for t in TASKS.values())  # 76

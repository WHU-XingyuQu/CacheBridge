"""Original tau-bench tools/scoring, with a local synchronous user adapter."""

import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import types

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/tau-bench"
sys.path.insert(0, str(VENDOR))
# The upstream user module imports LiteLLM, but this experiment supplies its
# own resident-model backend. A fail-closed import shim prevents network calls;
# none of the official environment, tools, tasks or scorer is edited.
if "litellm" not in sys.modules:
    shim = types.ModuleType("litellm")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Only the explicit local user backend is enabled")

    shim.completion = unavailable
    sys.modules["litellm"] = shim
from tau_bench.envs.retail import MockRetailDomainEnv
from tau_bench.envs.base import Env
from tau_bench.envs.user import LLMUserSimulationEnv
from tau_bench.types import Action, RewardResult, RewardActionInfo, RESPOND_ACTION_NAME


ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
SAMPLING = dict(
    temperature=0,
    max_new_tokens=1024,
    ignore_eos=False,
    stop_token_ids=[151645],
    skip_special_tokens=True,
)
USER_SAMPLING = dict(SAMPLING, max_new_tokens=512)
FORMAT = "\n\nCall at most one tool in each response. Initially handle customer intake and information gathering. An internal message will indicate the transition to case resolution. Follow the retail policy throughout."
TRANSITION = "\n<|im_start|>system\nInternal workflow: intake is complete. You are now the resolution agent. Continue from the conversation and tool observations above, verify the customer requirements, and resolve the case under the same retail policy. You may ask the customer questions and use all the available tools.\n<|im_end|>\n"

FORMAT_REMINDER = '\nTool execution format: when you need to use a tool, output exactly <tool_call>{"name": "function_name", "arguments": {"argument_name": "value"}}</tool_call> with an available function and its actual arguments. A statement that you will perform an action does not execute it. Use one tool call per response, or reply normally to the customer when no tool is needed.\n'
TRANSITION = TRANSITION.replace("\n<|im_end|>", FORMAT_REMINDER + "<|im_end|>")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_value(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


class ScoringUser:
    def step(self, content):
        return "###STOP###"

    def get_total_cost(self):
        return 0.0


class DeferredRetailEnv(MockRetailDomainEnv):
    def calculate_reward(self):
        # Grade only after the task clock stops. The unmodified upstream
        # evaluator is called on an isolated CPU snapshot by official_score.
        return RewardResult(
            reward=0.0,
            info=RewardActionInfo(r_actions=False, gt_data_hash="deferred"),
            actions=[],
        )


def new_env(split, index):
    return DeferredRetailEnv(user_strategy="human", task_split=split, task_index=index)


def official_score(env, done):
    clone = copy.copy(env)
    clone.data = copy.deepcopy(env.data)
    clone.actions = copy.deepcopy(env.actions)
    clone.user = ScoringUser()
    predicted_hash = env.get_data_hash()
    reward = Env.calculate_reward(clone)
    return dict(
        reward=float(reward.reward) if done else 0.0,
        terminal=done,
        diagnostic_reward_even_if_unfinished=float(reward.reward),
        official=reward.model_dump(),
        predicted_db_hash=predicted_hash,
        final_db_matches=predicted_hash == clone.get_data_hash(),
    )


def initial_text(env, tok, observation):
    return tok.apply_chat_template(
        [
            dict(role="system", content=env.wiki + FORMAT),
            dict(role="user", content=observation),
        ],
        tools=env.tools_info,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def append_response(text, response, action, observation):
    text += ASSISTANT + response + "<|im_end|>\n"
    if action.name == RESPOND_ACTION_NAME:
        return text + "<|im_start|>user\n" + observation + "<|im_end|>\n"
    return (
        text
        + "<|im_start|>user\n<tool_response>\n"
        + observation
        + "\n</tool_response><|im_end|>\n"
    )


def parse_action(response):
    values = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)
    if values:
        value = json.loads(values[0])
        if not isinstance(value, dict) or not isinstance(value.get("arguments"), dict):
            raise ValueError("Malformed function-call object")
        name = value.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z]\w*", name):
            raise ValueError("Invalid function name")
        return Action(name=name, kwargs=value["arguments"]), len(values)
    if "<tool_call>" in response or "</tool_call>" in response:
        raise ValueError("Unclosed tool call")
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": response}), 0


def boundary(action, observation, calls):
    successful_read = action.name in (
        "get_order_details",
        "get_product_details",
    ) and not observation.startswith(("Error", "Unknown"))
    if successful_read:
        return "first_successful_detail_query"
    if calls >= 6:
        return "six_call_intake_cap"
    return None


def snapshot(env):
    return dict(data=env.data, actions=[a.model_dump() for a in env.actions])


def from_snapshot(split, index, value):
    env = new_env(split, index)
    env.data = copy.deepcopy(value["data"])
    env.actions = [Action(**a) for a in value["actions"]]
    return env

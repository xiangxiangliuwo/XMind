"""
Driving agent with a frozen LLM and "Think Twice" decisions.

Round 1    system prompt + working memory + available actions -> a1 (no episodic memory).
Retrieval  only if a retriever is given:
               no trigger / empty bank / no match  -> keep a1
               memory a* == a1                     -> keep a1 (no second LLM call)
               memory a* != a1                     -> round 2
Round 2    round-1 messages + the retrieved memory block -> final action.
"""

import os
import re
import textwrap
from dataclasses import dataclass
from typing import Optional

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rich import print
from rich.markup import escape

from XMind.driver_agent.workingMemory import WorkingMemory
from XMind.scenario.directions import ACTION_NAMES


delimiter = "####"

SYSTEM_MESSAGE = textwrap.dedent(f"""\
You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
Each turn you receive the current working memory as JSON and a list of available actions, delimited by {delimiter}.

JSON fields:
- "Ego": [ID, x, y, v_x, v_y, a, lane_id]
- "11 Directions": front, rear, left, right, front_left, rear_left, front_right, rear_right, front_front, front_left_front, front_right_front.
An occupied slot is [ID, direction, x, y, v_x, v_y, a, lane_id, Risk]; an empty slot is null.

Notes:
- Units: x, y in m; v_x, v_y in m/s; a in m/s^2.
- A null slot only means no vehicle was selected in that region.
- "risk" is an object containing "value" and "level"."risk.level" is an integer: 0 = low risk (value < 0.5), 1 = medium risk (0.5 <= value < 0.7), 2 = high risk (value >= 0.7).

Your response must contain exactly two parts in this order:

Reason:{delimiter} <use working memory context to analyze the evolution of risks and compare the values of risks and identify the highest one as main risk source. You primary job is to avoid the highest risk source, even if other dierections still have high risks>
Action_id:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`>

Make sure to include {delimiter} to separate every step.
""")

REVISE_INSTRUCTION = (
    "\n\nYou have already seen the current scenario above. Now re-examine it together with the "
    "retrieved risk memory, and give your FINAL decision. "
    "Provide both the Reason and Action_id sections in the required format.\n"
)

# Sent when the reply contains no action id (kept byte-identical to the original prompt).
_CHECK_BODY = """\
You are a output checking assistant who is responsible for checking the output of another agent.

The output you received is: {output}

Your should just output the right int type of action_id, with no other characters or delimiters.
i.e. :
| Action_id | Action Description                                     |
|--------|--------------------------------------------------------|
| 0      | Turn-left: change lane to the left of the current lane |
| 1      | IDLE: remain in the current lane with current speed   |
| 2      | Turn-right: change lane to the right of the current lane|
| 3      | Acceleration: accelerate the vehicle                 |
| 4      | Deceleration: decelerate the vehicle                 |


You answer format would be:
{delimiter} <correct action_id within 0-4>
"""
CHECK_MESSAGE = "\n" + textwrap.indent(_CHECK_BODY, " " * 12) + " " * 12


def parse_action_id(text: str) -> int:
    """The last standalone digit 0-4 in the reply (also works with <think> blocks)."""
    candidates = re.findall(r"(?<!\d)([0-4])(?!\d)", text or "")
    if not candidates:
        raise ValueError(f"No action id (0-4) found in model response: {text!r}")
    return int(candidates[-1])


@dataclass
class DecisionResult:
    action: int                    # executed action
    first_action: int              # round-1 action (without memory)
    first_response: str
    final_response: str            # reply that produced the executed action
    human_question: str            # round-1 scenario message
    rounds: int                    # LLM decision calls: 1 or 2
    path: str                      # retrieval_off / retrieval miss reason / memory_agrees / revised_with_memory
    retrieval: Optional[dict] = None
    memory_prompt: str = ""        # memory block added in round 2 (empty without round 2)

    def summary(self):
        text = f"round 1: {ACTION_NAMES[self.first_action]}"
        if self.retrieval is not None and self.retrieval.get("capsule_ids"):
            matches = self.retrieval.get("matches") or [{}]
            mirrored = " (mirrored)" if matches[0].get("mirrored") else ""
            text += f" | memory {self.retrieval['capsule_ids']}: a*={ACTION_NAMES[self.retrieval['best_action']]}{mirrored}"
        if self.rounds == 2:
            text += f" | round 2: {ACTION_NAMES[self.action]}"
        return f"{text} | final: {ACTION_NAMES[self.action]} ({self.path})"


class DriverAgent:
    def __init__(self, temperature: float = 0, *, retriever=None) -> None:
        """retriever: a CapsuleRetriever, or None to decide from working memory only."""
        self.retriever = retriever
        api_type = os.getenv("OPENAI_API_TYPE")
        if api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                max_tokens=2000,
                request_timeout=120,
                streaming=True,
            )
        else:
            raise ValueError("OPENAI_API_TYPE must be 'openai' or 'azure'")

    @staticmethod
    def scenario_message(working_memory: WorkingMemory, available_actions: str, driving_intensions: str) -> str:
        """Round-1 user message: working memory JSON + driving intention + available actions."""
        return (
            "Here is the current scenario:\n"
            f"{delimiter} Structured driving scenario (JSON):\n"
            f"```json\n{working_memory.to_json()}\n```\n"
            f"{delimiter} Driving Intensions:\n{driving_intensions}\n"
            f"{delimiter} Available actions:\n{available_actions}\n\n"
            "Provide both the Reason and Action_id sections in the required format.\n"
        )

    def _ask(self, messages, title):
        """Stream one LLM reply and parse the action id; ask for the bare id if the format is wrong."""
        print(f"[cyan]Agent answer ({title}):[/cyan]")
        response = ""
        for chunk in self.llm.stream(messages):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")

        decision_text = response.split(delimiter)[-1]
        try:
            action = parse_action_id(decision_text)
        except ValueError:
            print("Output is not a int number, checking the output...")
            check = self.llm([HumanMessage(content=CHECK_MESSAGE.format(output=decision_text, delimiter=delimiter))])
            action = parse_action_id(check.content.split(delimiter)[-1])
        return action, response

    def decision(
        self,
        working_memory: WorkingMemory,
        available_actions: str = "Not available",
        driving_intensions: str = "Not available",
        *,
        current_frame: Optional[str] = None,
        lane_boundary: Optional[dict] = None,
    ) -> DecisionResult:
        """current_frame / lane_boundary: the current frame JSON and road boundaries, used for retrieval."""
        if not isinstance(working_memory, WorkingMemory):
            raise TypeError("working_memory must be a WorkingMemory instance")
        if len(working_memory) == 0:
            raise ValueError("Working memory must not be empty")

        # Round 1: current scenario only.
        human_message = self.scenario_message(working_memory, available_actions, driving_intensions)
        messages = [SystemMessage(content=SYSTEM_MESSAGE), HumanMessage(content=human_message)]
        first_action, first_response = self._ask(messages, "round 1, current scenario only")
        result = DecisionResult(
            action=first_action,
            first_action=first_action,
            first_response=first_response,
            final_response=first_response,
            human_question=human_message,
            rounds=1,
            path="retrieval_off",
        )
        if self.retriever is None:
            return result
        if current_frame is None or lane_boundary is None:
            raise ValueError("current_frame and lane_boundary are required when retrieval is on")

        retrieval = self.retriever.retrieve(current_frame, lane_boundary=lane_boundary)
        result.retrieval = retrieval
        if not retrieval["capsules"]:
            result.path = retrieval["reason"]
            return result
        if retrieval["best_action"] == first_action:
            result.path = "memory_agrees"
            return result

        # Round 2: round-1 messages + memory block.
        memory_prompt = self.retriever.to_prompt(retrieval)
        print("[cyan]Retrieved memory (added to the round-1 prompt for round 2):[/cyan]")
        print(escape(memory_prompt))
        final_action, final_response = self._ask(
            messages + [HumanMessage(content=memory_prompt + REVISE_INSTRUCTION)], "round 2, with retrieved memory")
        result.action = final_action
        result.final_response = final_response
        result.rounds = 2
        result.path = "revised_with_memory"
        result.memory_prompt = memory_prompt
        return result

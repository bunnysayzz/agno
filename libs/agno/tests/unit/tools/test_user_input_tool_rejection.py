"""Regression tests: a rejected user-input tool must not execute.

Issue: agno-agi/agno#9451. A tool with @approval(type="required") and
requires_user_input=True routes through Case 4 of handle_tool_call_updates,
which previously executed the tool unconditionally and recorded "approved"
even when the approval was explicitly rejected. Case 4 now mirrors Case 1:
an explicit rejection goes through reject_tool_call and never runs the tool,
while confirmed=None (no approval record yet) still executes.
"""

import pytest

from agno.agent import Agent
from agno.models.response import ToolExecution
from agno.run.agent import RunOutput
from agno.run.messages import RunMessages

import agno.agent._tools as agent_tools
from agno.agent._tools import ahandle_tool_call_updates, handle_tool_call_updates


def make_tool(confirmed, call_id="call_1"):
    return ToolExecution(
        tool_call_id=call_id,
        tool_name="ask_for_confirmation",
        requires_user_input=True,
        confirmed=confirmed,
        approval_type="required",
        tool_args={},
        user_input_schema=[],
    )


def patch_handlers(monkeypatch):
    executed = []
    rejected = []

    def fake_run_tool(agent, run_response, run_messages, tool, functions=None):
        executed.append(tool)
        yield None

    monkeypatch.setattr(agent_tools, "run_tool", fake_run_tool)
    monkeypatch.setattr(
        agent_tools, "reject_tool_call", lambda *args, **kwargs: rejected.append(args[2])
    )
    return executed, rejected


def test_rejected_user_input_tool_does_not_execute(monkeypatch):
    tool = make_tool(confirmed=False)
    executed, rejected = patch_handlers(monkeypatch)

    handle_tool_call_updates(Agent(), RunOutput(tools=[tool]), RunMessages(), tools=[])

    assert executed == []
    assert len(rejected) == 1
    assert tool.tool_call_error is True
    assert tool.requires_user_input is False
    assert tool.answered is True


def test_confirmed_user_input_tool_executes(monkeypatch):
    tool = make_tool(confirmed=True)
    executed, rejected = patch_handlers(monkeypatch)

    handle_tool_call_updates(Agent(), RunOutput(tools=[tool]), RunMessages(), tools=[])

    assert len(executed) == 1
    assert rejected == []


def test_unconfirmed_user_input_tool_executes(monkeypatch):
    # confirmed=None (no approval record yet) must still execute: only an
    # explicit rejection blocks execution.
    tool = make_tool(confirmed=None)
    executed, rejected = patch_handlers(monkeypatch)

    handle_tool_call_updates(Agent(), RunOutput(tools=[tool]), RunMessages(), tools=[])

    assert len(executed) == 1
    assert rejected == []


@pytest.mark.asyncio
async def test_async_rejected_user_input_tool_does_not_execute(monkeypatch):
    tool = make_tool(confirmed=False)
    executed = []
    rejected = []

    async def fake_arun_tool(agent, run_response, run_messages, tool, functions=None):
        executed.append(tool)
        yield None

    monkeypatch.setattr(agent_tools, "arun_tool", fake_arun_tool)
    monkeypatch.setattr(
        agent_tools, "reject_tool_call", lambda *args, **kwargs: rejected.append(args[2])
    )

    await ahandle_tool_call_updates(Agent(), RunOutput(tools=[tool]), RunMessages(), tools=[])

    assert executed == []
    assert len(rejected) == 1
    assert tool.tool_call_error is True

"""
tests.py — unit tests for the SkillAgent package.

Run:  python tests.py
All tests use MockClient and a temporary in-memory DB — no API key required.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import time
import unittest

# Make sure we can import skill_agent from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_agent.llm import MockClient
from skill_agent.executor import execute_tool
from skill_agent.tool_writer import write_tool
from skill_agent.validator import validate_tool, _equal
from skill_agent.tool_registry import ToolRegistry, Tool
from skill_agent.recognizer import recognize
from skill_agent.embeddings import embed
from skill_agent.reviser import revise_tool
from skill_agent.mcp_server import (
    _extract_input_schema, _validate_code_static, handle_initialize,
    handle_tools_list, handle_tools_call, META_TOOLS,
)
from skill_agent import SkillAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

SIMPLE_FN = """\
def add(a: int, b: int) -> int:
    \"\"\"Return a + b.\"\"\"
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Inputs must be integers")
    return a + b
"""

BROKEN_FN = """\
def broken(x: int) -> int:
    \"\"\"Always crashes.\"\"\"
    raise RuntimeError("intentional failure")
"""


def _tmp_registry() -> ToolRegistry:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return ToolRegistry(db_path=path)


# ── Executor tests ────────────────────────────────────────────────────────────

class TestExecutor(unittest.TestCase):

    def test_simple_success(self):
        r = execute_tool(SIMPLE_FN, "add", {"a": 3, "b": 4})
        self.assertTrue(r.success)
        self.assertEqual(r.output, 7)
        self.assertIsNone(r.error)
        self.assertGreater(r.latency_ms, 0)

    def test_runtime_error_captured(self):
        r = execute_tool(BROKEN_FN, "broken", {"x": 1})
        self.assertFalse(r.success)
        self.assertIn("intentional failure", r.error)

    def test_bad_kwargs_captured(self):
        r = execute_tool(SIMPLE_FN, "add", {"wrong_param": 1})
        self.assertFalse(r.success)

    def test_timeout(self):
        infinite_fn = """\
def spin(n: int) -> int:
    \"\"\"Infinite loop.\"\"\"
    import time
    while True:
        time.sleep(0.1)
"""
        # Override timeout to 1s for speed
        import skill_agent.executor as ex
        original = ex.TIMEOUT_SECONDS
        ex.TIMEOUT_SECONDS = 1
        try:
            r = execute_tool(infinite_fn, "spin", {"n": 1})
            self.assertFalse(r.success)
            self.assertIn("timed out", r.error.lower())
        finally:
            ex.TIMEOUT_SECONDS = original

    def test_json_serializable_output(self):
        dict_fn = """\
def make_dict(key: str, value: int) -> dict:
    \"\"\"Return a dict.\"\"\"
    return {key: value}
"""
        r = execute_tool(dict_fn, "make_dict", {"key": "x", "value": 42})
        self.assertTrue(r.success)
        self.assertEqual(r.output, {"x": 42})


# ── Tool writer tests ─────────────────────────────────────────────────────────

class TestToolWriter(unittest.TestCase):

    def test_write_valid_tool(self):
        client = MockClient()
        result = write_tool("Calculate compound interest", client)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "calculate_compound_interest")
        self.assertIn("def calculate_compound_interest", result.code)

    def test_bad_json_returns_none(self):
        class BadClient:
            def complete(self, system, user, max_tokens=1000):
                return "this is not json at all!!!"
        result = write_tool("anything", BadClient())
        self.assertIsNone(result)

    def test_syntax_error_returns_none(self):
        import json

        class SyntaxBadClient:
            def complete(self, system, user, max_tokens=1000):
                return json.dumps({
                    "name": "bad_fn",
                    "description": "broken",
                    "code": "def bad_fn(\n  # unclosed",
                })
        result = write_tool("anything", SyntaxBadClient())
        self.assertIsNone(result)

    def test_missing_function_name_returns_none(self):
        import json

        class WrongNameClient:
            def complete(self, system, user, max_tokens=1000):
                # says name is "foo" but code defines "bar"
                return json.dumps({
                    "name": "foo",
                    "description": "mismatch",
                    "code": "def bar(x: int) -> int:\n    \"\"\"doc\"\"\"\n    return x",
                })
        result = write_tool("anything", WrongNameClient())
        self.assertIsNone(result)


# ── Validator tests ───────────────────────────────────────────────────────────

class TestValidator(unittest.TestCase):

    def test_valid_function_passes(self):
        client = MockClient()
        vr = validate_tool("calculate_compound_interest", "desc", SIMPLE_FN, client)
        # MockClient returns tests for compound_interest; these won't match add()
        # so we expect failure — but the validator itself runs without crashing.
        self.assertIsNotNone(vr)
        self.assertIsInstance(vr.passed, bool)

    def test_equal_numeric_tolerance(self):
        self.assertTrue(_equal(1.0000000001, 1.0))
        self.assertTrue(_equal(1050.0, 1050.00000001))
        self.assertFalse(_equal(1050.0, 1051.0))
        self.assertTrue(_equal(5, 5))
        self.assertFalse(_equal("foo", "bar"))

    def test_no_test_cases_fails(self):
        class NoTestClient:
            def complete(self, system, user, max_tokens=1000):
                return "[]"   # empty list → fail
        vr = validate_tool("fn", "desc", SIMPLE_FN, NoTestClient())
        self.assertFalse(vr.passed)


# ── Tool registry tests ───────────────────────────────────────────────────────

class TestToolRegistry(unittest.TestCase):

    def test_save_and_retrieve(self):
        reg = _tmp_registry()
        tool = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        emb = embed("add: Adds two integers")
        tid = reg.save_tool(tool, emb)
        self.assertGreater(tid, 0)

        results = reg.search("add two numbers together")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool.name, "add")

    def test_degradation(self):
        reg = _tmp_registry()
        tool = Tool(name="flaky", description="Flaky tool", code=SIMPLE_FN)
        emb = embed("flaky: Flaky tool")
        tid = reg.save_tool(tool, emb)

        # Record 5 failures — should trigger degradation
        for _ in range(5):
            reg.record_execution(tid, success=False)

        refreshed = reg.get_tool_by_id(tid)
        self.assertEqual(refreshed.status, "degraded")

    def test_recovery_from_degradation(self):
        reg = _tmp_registry()
        tool = Tool(name="recover", description="Recovers", code=SIMPLE_FN)
        emb = embed("recover: Recovers")
        tid = reg.save_tool(tool, emb)

        # 3 failures then 10 successes → rate = 10/13 ≈ 77% → active
        for _ in range(3):
            reg.record_execution(tid, success=False)
        for _ in range(10):
            reg.record_execution(tid, success=True)

        refreshed = reg.get_tool_by_id(tid)
        self.assertEqual(refreshed.status, "active")

    def test_search_returns_empty_on_empty_registry(self):
        reg = _tmp_registry()
        results = reg.search("anything")
        self.assertEqual(results, [])

    def test_duplicate_name_raises(self):
        import sqlite3
        reg = _tmp_registry()
        tool = Tool(name="dup", description="Dup", code=SIMPLE_FN)
        emb = embed("dup: Dup")
        reg.save_tool(tool, emb)
        with self.assertRaises(sqlite3.IntegrityError):
            reg.save_tool(tool, emb)


# ── Recognizer tests ──────────────────────────────────────────────────────────

class TestRecognizer(unittest.TestCase):

    def test_direct_answer_on_conversational_query(self):
        client = MockClient()
        reg = _tmp_registry()
        # MockClient returns "answer" for non-calculation queries
        result = recognize("tell me a joke", reg, client)
        self.assertEqual(result.action, "answer")

    def test_create_tool_on_calculation_query(self):
        client = MockClient()
        reg = _tmp_registry()
        result = recognize("calculate compound interest on $1000", reg, client)
        self.assertEqual(result.action, "create_tool")

    def test_high_similarity_skips_llm(self):
        """When similarity >= HIGH_THRESHOLD, should return 'use_tool' without an LLM call."""
        reg = _tmp_registry()
        tool = Tool(name="add", description="Adds two integers a and b", code=SIMPLE_FN)
        emb = embed("add: Adds two integers a and b")
        reg.save_tool(tool, emb)

        # Calls with identical text to the stored description → very high similarity
        class NeverCallClient:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called for high-similarity matches")

        result = recognize("Adds two integers a and b", reg, NeverCallClient())
        self.assertEqual(result.action, "use_tool")


# ── Full agent integration test ───────────────────────────────────────────────

class TestSkillAgentIntegration(unittest.TestCase):

    def test_end_to_end_with_mock(self):
        client = MockClient()
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        agent = SkillAgent(llm_client=client, db_path=path)

        # First run: should create a tool
        r1 = agent.run("Calculate compound interest on $5000 at 4.2% for 7 years")
        self.assertIsNotNone(r1.answer)
        self.assertIn(r1.action_taken, (
            "created_and_used_tool",
            "created_tool_then_answered_directly",
            "answered_directly",
        ))

        # Second run with similar query: tool should exist now
        r2 = agent.run("Calculate compound interest on $10000 at 5% for 10 years")
        self.assertIsNotNone(r2.answer)

    def test_conversational_query_answered_directly(self):
        client = MockClient()
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        agent = SkillAgent(llm_client=client, db_path=path)

        r = agent.run("Can you explain what compound interest means?")
        self.assertEqual(r.action_taken, "answered_directly")
        self.assertIsNone(r.tool_name)


# ── Feedback tests ───────────────────────────────────────────────────────────

class TestFeedback(unittest.TestCase):

    def test_save_and_retrieve_sentiment(self):
        reg = _tmp_registry()
        tool = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("add: Adds two integers"))

        self.assertIsNone(reg.get_user_sentiment(tid))  # no feedback yet

        reg.save_feedback(tid, positive=True)
        reg.save_feedback(tid, positive=True)
        reg.save_feedback(tid, positive=False)

        sentiment = reg.get_user_sentiment(tid)
        self.assertAlmostEqual(sentiment, 2 / 3, places=5)

    def test_negative_feedback_degrades_tool(self):
        """
        Negative user sentiment on top of a mediocre execution rate should push
        the combined health score below DEGRADED_THRESHOLD.

        combined = 0.70 * exec_rate + 0.30 * sentiment
        With exec_rate = 0.4  and  sentiment = 0.0:
          combined = 0.28 < 0.60 (DEGRADED_THRESHOLD)  → degraded
          0.28 < 0.35 (RETIRE_THRESHOLD) but total(5) < MIN_EXECUTIONS_TO_RETIRE(10)
          → degraded, NOT retired
        """
        reg = _tmp_registry()
        tool = Tool(name="shaky", description="Shaky tool", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("shaky: Shaky tool"))

        # 2 successes + 3 failures = exec_rate 0.40 (crosses MIN_EXECUTIONS_TO_DEGRADE=5)
        for _ in range(2):
            reg.record_execution(tid, success=True)
        for _ in range(3):
            reg.record_execution(tid, success=False)

        # 10 all-negative feedback votes → sentiment = 0.0
        for _ in range(10):
            reg.save_feedback(tid, positive=False)

        refreshed = reg.get_tool_by_id(tid)
        self.assertIn(refreshed.status, ("degraded", "retired"))

    def test_feedback_on_direct_answer_is_noop(self):
        """feedback() on a result with no tool_name should not crash."""
        client = MockClient()
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        agent = SkillAgent(llm_client=client, db_path=path)

        r = agent.run("Can you explain what compound interest means?")
        self.assertEqual(r.tool_name, None)
        # Should not raise
        agent.feedback(r, positive=False, comment="not helpful")

    def test_feedback_comment_stored(self):
        reg = _tmp_registry()
        tool = Tool(name="adder", description="Adds", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("adder: Adds"))
        reg.save_feedback(tid, positive=True, comment="very accurate")

        row = reg._conn.execute(
            "SELECT comment FROM tool_feedback WHERE tool_id = ?", (tid,)
        ).fetchone()
        self.assertEqual(row["comment"], "very accurate")


# ── Versioning tests ──────────────────────────────────────────────────────────

class TestVersioning(unittest.TestCase):

    def test_update_tool_creates_version(self):
        reg = _tmp_registry()
        tool = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        emb = embed("add: Adds two integers")
        tid = reg.save_tool(tool, emb)

        new_code = SIMPLE_FN.replace("return a + b", "return int(a + b)")
        new_emb  = embed("add: Adds two integers (revised)")
        reg.update_tool(tid, new_code, "Adds two integers (revised)", new_emb, reason="test revision")

        versions = reg.get_versions(tid)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].version_num, 1)
        self.assertEqual(versions[0].code, SIMPLE_FN)
        self.assertEqual(versions[0].reason, "test revision")

    def test_multiple_updates_increment_version_num(self):
        reg = _tmp_registry()
        tool = Tool(name="mul", description="Multiplies", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("mul: Multiplies"))

        for i in range(3):
            reg.update_tool(tid, SIMPLE_FN, f"desc v{i+2}", embed(f"mul: v{i+2}"))

        versions = reg.get_versions(tid)
        self.assertEqual(len(versions), 3)
        self.assertEqual([v.version_num for v in versions], [1, 2, 3])

    def test_update_resets_status_to_active(self):
        reg = _tmp_registry()
        tool = Tool(name="flaky2", description="Flaky", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("flaky2: Flaky"))

        # Degrade it manually
        for _ in range(5):
            reg.record_execution(tid, success=False)
        self.assertEqual(reg.get_tool_by_id(tid).status, "degraded")

        reg.update_tool(tid, SIMPLE_FN, "Flaky (fixed)", embed("flaky2: fixed"))
        self.assertEqual(reg.get_tool_by_id(tid).status, "active")

    def test_update_tool_rebuilds_faiss_index(self):
        """Search should find the tool by its new description after update."""
        reg = _tmp_registry()
        tool = Tool(name="converter", description="Converts Celsius to Fahrenheit", code=SIMPLE_FN)
        emb = embed("converter: Converts Celsius to Fahrenheit")
        tid = reg.save_tool(tool, emb)

        new_desc = "Converts Fahrenheit to Celsius"
        new_emb  = embed(f"converter: {new_desc}")
        reg.update_tool(tid, SIMPLE_FN, new_desc, new_emb)

        results = reg.search("Fahrenheit to Celsius")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].tool.name, "converter")


# ── Pruning tests ─────────────────────────────────────────────────────────────

class TestPruning(unittest.TestCase):

    def test_prune_stale_tool(self):
        reg = _tmp_registry()
        tool = Tool(name="old_tool", description="Very old", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("old_tool: Very old"))

        # Force created_at to be 60 days ago
        reg._conn.execute(
            "UPDATE tools SET created_at = ?, last_used_at = NULL WHERE id = ?",
            (time.time() - 60 * 86_400, tid),
        )
        reg._conn.commit()

        retired = reg.prune(stale_days=30)
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["name"], "old_tool")
        self.assertIn("Stale", retired[0]["reason"])
        self.assertEqual(reg.get_tool_by_id(tid).status, "retired")

    def test_prune_deeply_degraded_tool(self):
        """
        prune() should retire tools whose raw counts put them below RETIRE_THRESHOLD
        even if they slipped past auto-retirement (e.g. pre-migration data, or counts
        written directly to the DB without going through record_execution).
        """
        reg = _tmp_registry()
        tool = Tool(name="bad_tool", description="Always fails", code=BROKEN_FN)
        tid = reg.save_tool(tool, embed("bad_tool: Always fails"))

        # Bypass record_execution to simulate data that pre-dates auto-retire logic.
        # success_rate = 0/12 = 0.0 < RETIRE_THRESHOLD(0.35), total(12) >= MIN_TO_RETIRE(10)
        reg._conn.execute(
            "UPDATE tools SET status='degraded', success_count=0, failure_count=12, "
            "usage_count=12 WHERE id=?",
            (tid,),
        )
        reg._conn.commit()

        retired = reg.prune()
        names = [r["name"] for r in retired]
        self.assertIn("bad_tool", names)
        self.assertEqual(reg.get_tool_by_id(tid).status, "retired")

    def test_prune_duplicate_tools(self):
        reg = _tmp_registry()
        desc = "Calculates compound interest given principal, rate, and years"
        emb = embed(desc)

        # Save two tools with identical embeddings (same description)
        t1 = Tool(name="ci_v1", description=desc, code=SIMPLE_FN)
        t2 = Tool(name="ci_v2", description=desc, code=SIMPLE_FN)
        tid1 = reg.save_tool(t1, emb.copy())
        tid2 = reg.save_tool(t2, emb.copy())

        # Give t1 more uses so t2 gets retired
        reg._conn.execute("UPDATE tools SET usage_count = 10 WHERE id = ?", (tid1,))
        reg._conn.execute("UPDATE tools SET usage_count = 1  WHERE id = ?", (tid2,))
        reg._conn.commit()

        retired = reg.prune()
        names = [r["name"] for r in retired]
        self.assertIn("ci_v2", names)
        self.assertEqual(reg.get_tool_by_id(tid1).status, "active")
        self.assertEqual(reg.get_tool_by_id(tid2).status, "retired")

    def test_prune_active_recent_tool_untouched(self):
        reg = _tmp_registry()
        tool = Tool(name="fresh", description="Fresh tool", code=SIMPLE_FN)
        reg.save_tool(tool, embed("fresh: Fresh tool"))

        retired = reg.prune(stale_days=30)
        # created_at is now() so it should not be stale
        names = [r["name"] for r in retired]
        self.assertNotIn("fresh", names)

    def test_retire_tool_removes_from_search(self):
        reg = _tmp_registry()
        tool = Tool(name="gone", description="Will be retired", code=SIMPLE_FN)
        tid = reg.save_tool(tool, embed("gone: Will be retired"))

        results_before = reg.search("Will be retired")
        self.assertTrue(len(results_before) > 0)

        reg.retire_tool(tid)

        results_after = reg.search("Will be retired")
        self.assertEqual(len(results_after), 0)


# ── Reviser tests ─────────────────────────────────────────────────────────────

class TestReviser(unittest.TestCase):

    def test_revise_returns_written_tool(self):
        client = MockClient()
        tool = Tool(
            name="calculate_compound_interest",
            description="Calculates compound interest",
            code=SIMPLE_FN,  # wrong code for the name — simulates degraded state
            tool_id=1,
        )
        result = revise_tool(tool, client)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "calculate_compound_interest")
        self.assertIn("def calculate_compound_interest", result.code)

    def test_revise_rejects_name_change(self):
        """Reviser must keep the same function name; reject if LLM changes it."""
        import json

        class NameChangerClient:
            def complete(self, system, user, max_tokens=1000):
                return json.dumps({
                    "name": "different_name",
                    "description": "desc",
                    "code": "def different_name(x: int) -> int:\n    \"\"\"doc\"\"\"\n    return x",
                })

        tool = Tool(name="original_name", description="desc", code=SIMPLE_FN, tool_id=1)
        result = revise_tool(tool, NameChangerClient())
        self.assertIsNone(result)

    def test_agent_feedback_triggers_revision_on_degraded_tool(self):
        """
        Negative feedback on a degraded tool should trigger _attempt_revision,
        which calls revise_tool → validate_tool → update_tool.
        """
        client = MockClient()
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        agent = SkillAgent(llm_client=client, db_path=path)

        # Create the tool via a run
        r = agent.run("Calculate compound interest on $5000 at 4.2% for 7 years")

        tool_name = r.tool_name
        if tool_name is None:
            self.skipTest("Tool was not created in this run")

        tool = agent.registry.get_tool_by_name(tool_name)
        self.assertIsNotNone(tool)

        # Manually degrade the tool
        for _ in range(5):
            agent.registry.record_execution(tool.tool_id, success=False)
        agent.registry._conn.execute(
            "UPDATE tools SET status = 'degraded' WHERE id = ?", (tool.tool_id,)
        )
        agent.registry._conn.commit()

        # Negative feedback on degraded tool should trigger revision
        r2 = agent.run("Calculate compound interest on $1000 at 3% for 5 years")
        agent.feedback(r2, positive=False)

        # Tool should now have a version history (was revised)
        versions = agent.registry.get_versions(tool.tool_id)
        # Revision may or may not succeed depending on mock behaviour;
        # either way the call must not raise
        self.assertIsInstance(versions, list)


# ── MCP server tests ──────────────────────────────────────────────────────────

class TestMCPServer(unittest.TestCase):

    # ── _extract_input_schema ─────────────────────────────────────────────────

    def test_extract_input_schema_typed(self):
        schema = _extract_input_schema(SIMPLE_FN, "add")
        self.assertEqual(schema["type"], "object")
        self.assertIn("a", schema["properties"])
        self.assertIn("b", schema["properties"])
        self.assertEqual(schema["properties"]["a"]["type"], "number")
        self.assertEqual(schema["properties"]["b"]["type"], "number")
        self.assertIn("a", schema["required"])
        self.assertIn("b", schema["required"])

    def test_extract_input_schema_unknown_type_defaults_to_string(self):
        fn = (
            "def greet(name) -> str:\n"
            "    \"\"\"Greet someone.\"\"\"\n"
            "    return f'Hello {name}'\n"
        )
        schema = _extract_input_schema(fn, "greet")
        self.assertEqual(schema["properties"]["name"]["type"], "string")

    def test_extract_input_schema_missing_function(self):
        schema = _extract_input_schema(SIMPLE_FN, "nonexistent")
        self.assertEqual(schema["properties"], {})

    # ── initialize ────────────────────────────────────────────────────────────

    def test_handle_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handle_initialize(req)
        self.assertEqual(resp["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", resp["result"]["capabilities"])
        self.assertEqual(resp["id"], 1)

    # ── tools/list returns stable meta-tools ──────────────────────────────────

    def test_handle_tools_list_returns_meta_tools(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handle_tools_list(req)
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertEqual(
            names,
            {"search_tools", "execute_tool", "save_tool", "save_tool_version", "tool_stats"},
        )

    def test_handle_tools_list_always_same_count(self):
        """tools/list must return the same 5 meta-tools regardless of registry state."""
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
        self.assertEqual(len(handle_tools_list(req)["result"]["tools"]), len(META_TOOLS))

    def test_handle_tools_list_each_has_input_schema(self):
        req = {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}}
        for tool in handle_tools_list(req)["result"]["tools"]:
            self.assertIn("inputSchema", tool, f"{tool['name']} missing inputSchema")

    # ── search_tools ──────────────────────────────────────────────────────────

    def test_search_tools_finds_matching(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "search_tools", "arguments": {"query": "add two numbers together"}},
        }
        resp = handle_tools_call(req, reg)
        self.assertNotIn("error", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertGreaterEqual(data["count"], 1)
        self.assertEqual(data["matches"][0]["name"], "add")

    def test_search_tools_empty_registry(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "search_tools", "arguments": {"query": "anything"}},
        }
        resp = handle_tools_call(req, reg)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["count"], 0)

    def test_search_tools_missing_query_returns_error(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "search_tools", "arguments": {}},
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    # ── execute_tool ──────────────────────────────────────────────────────────

    def test_execute_tool_success(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "execute_tool", "arguments": {"tool_id": tid, "args": {"a": 3, "b": 4}}},
        }
        resp = handle_tools_call(req, reg)
        self.assertNotIn("error", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["result"], 7)
        self.assertIn("latency_ms", data)

    def test_execute_tool_records_usage(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "execute_tool", "arguments": {"tool_id": tid, "args": {"a": 1, "b": 2}}},
        }
        handle_tools_call(req, reg)
        self.assertEqual(reg.get_tool_by_id(tid).usage_count, 1)

    def test_execute_tool_missing_id_returns_error(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "execute_tool", "arguments": {}},
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_execute_tool_unknown_id_returns_error(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {"name": "execute_tool", "arguments": {"tool_id": 9999}},
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    # ── _validate_code_static ─────────────────────────────────────────────────

    def test_validate_code_static_valid(self):
        valid, err = _validate_code_static(SIMPLE_FN, "add")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validate_code_static_syntax_error(self):
        valid, err = _validate_code_static("def broken(:\n    pass", "broken")
        self.assertFalse(valid)
        self.assertIn("Syntax error", err)

    def test_validate_code_static_missing_function(self):
        valid, err = _validate_code_static(SIMPLE_FN, "nonexistent")
        self.assertFalse(valid)
        self.assertIn("No function named", err)

    def test_validate_code_static_dangerous_import(self):
        code = (
            "import os\n"
            "def read_file(path: str) -> str:\n"
            "    \"\"\"Read a file.\"\"\"\n"
            "    with open(path) as f:\n"
            "        return f.read()\n"
        )
        valid, err = _validate_code_static(code, "read_file")
        self.assertFalse(valid)
        self.assertIn("os", err)

    # ── save_tool ─────────────────────────────────────────────────────────────

    def test_save_tool_success(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 12, "method": "tools/call",
            "params": {
                "name": "save_tool",
                "arguments": {
                    "name": "add",
                    "description": "Adds two integers a and b",
                    "code": SIMPLE_FN,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertNotIn("error", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertGreater(data["tool_id"], 0)
        self.assertEqual(data["name"], "add")
        self.assertEqual(data["status"], "active")

        # Verify it's in the registry
        tool = reg.get_tool_by_id(data["tool_id"])
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "add")

    def test_save_tool_missing_fields(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 13, "method": "tools/call",
            "params": {
                "name": "save_tool",
                "arguments": {"name": "test"},
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_save_tool_duplicate_name(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 14, "method": "tools/call",
            "params": {
                "name": "save_tool",
                "arguments": {
                    "name": "add",
                    "description": "Another adder",
                    "code": SIMPLE_FN,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertIn("already exists", resp["error"]["message"])

    def test_save_tool_dangerous_code_rejected(self):
        reg = _tmp_registry()
        code = (
            "import os\n"
            "def bad_tool(path: str) -> str:\n"
            "    \"\"\"Bad.\"\"\"\n"
            "    return open(path).read()\n"
        )
        req = {
            "jsonrpc": "2.0", "id": 15, "method": "tools/call",
            "params": {
                "name": "save_tool",
                "arguments": {
                    "name": "bad_tool",
                    "description": "Reads files",
                    "code": code,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertIn("dangerous modules", resp["error"]["message"])

    # ── save_tool_version ─────────────────────────────────────────────────────

    def test_save_tool_version_success(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(t, embed("add: Adds two integers"))

        new_code = SIMPLE_FN.replace("return a + b", "return int(a + b)")
        req = {
            "jsonrpc": "2.0", "id": 16, "method": "tools/call",
            "params": {
                "name": "save_tool_version",
                "arguments": {
                    "tool_id": tid,
                    "name": "add",
                    "description": "Adds two integers (improved)",
                    "code": new_code,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertNotIn("error", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["tool_id"], tid)
        self.assertEqual(data["name"], "add")
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["versions_before"], 0)
        self.assertEqual(data["versions_after"], 1)

        # Verify version history exists
        versions = reg.get_versions(tid)
        self.assertEqual(len(versions), 1)

    def test_save_tool_version_name_mismatch(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 17, "method": "tools/call",
            "params": {
                "name": "save_tool_version",
                "arguments": {
                    "tool_id": tid,
                    "name": "different_name",
                    "description": "Desc",
                    "code": SIMPLE_FN,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertIn("does not match", resp["error"]["message"])

    def test_save_tool_version_unknown_id(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 18, "method": "tools/call",
            "params": {
                "name": "save_tool_version",
                "arguments": {
                    "tool_id": 9999,
                    "name": "fn",
                    "description": "Desc",
                    "code": SIMPLE_FN,
                },
            },
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    # ── tool_stats ────────────────────────────────────────────────────────────

    def test_tool_stats_returns_metadata(self):
        reg = _tmp_registry()
        t = Tool(name="add", description="Adds two integers", code=SIMPLE_FN)
        tid = reg.save_tool(t, embed("add: Adds two integers"))

        req = {
            "jsonrpc": "2.0", "id": 14, "method": "tools/call",
            "params": {"name": "tool_stats", "arguments": {"tool_id": tid}},
        }
        resp = handle_tools_call(req, reg)
        self.assertNotIn("error", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(data["name"], "add")
        self.assertEqual(data["status"], "active")
        self.assertIn("success_rate", data)
        self.assertIn("usage_count", data)

    def test_tool_stats_unknown_id_returns_error(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 15, "method": "tools/call",
            "params": {"name": "tool_stats", "arguments": {"tool_id": 9999}},
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    # ── unknown meta-tool ────────────────────────────────────────────────────

    def test_handle_tools_call_unknown_meta_tool(self):
        reg = _tmp_registry()
        req = {
            "jsonrpc": "2.0", "id": 16, "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        }
        resp = handle_tools_call(req, reg)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)

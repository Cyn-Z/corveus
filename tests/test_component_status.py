"""Exercise health reporting without loading the inference models."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock
from urllib.parse import urlsplit


class ComponentStatusTests(unittest.TestCase):
    def status(self, *, enabled=True, storage_error=False, model_error=False, router=True, worker=None):
        tree = ast.parse((Path(__file__).parents[1] / 'analyst/http_server.py').read_text())
        handler = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name == 'component_status')
        manager = types.SimpleNamespace(enabled=enabled, list_conversations=Mock())
        if storage_error:
            manager.list_conversations.side_effect = OSError('unavailable')
        response = Mock()
        response.__enter__ = Mock(return_value=types.SimpleNamespace(status=200))
        response.__exit__ = Mock(return_value=False)
        opener = Mock(return_value=response)
        if model_error:
            opener.side_effect = TimeoutError()
        analyst = types.SimpleNamespace(MEMORY_MANAGER=manager, LLM_URL='http://llama:8080/v1/chat/completions', SKILL_ROUTER=object() if router else None)
        ns = dict(analyst=analyst, urlsplit=urlsplit, urllib=types.SimpleNamespace(request=types.SimpleNamespace(urlopen=opener)))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<status>', 'exec'), ns)
        target = types.SimpleNamespace(send_json=Mock())
        if worker is not None:
            available = Mock(return_value=worker)
            if isinstance(worker, Exception):
                available.side_effect = worker
            target.server = types.SimpleNamespace(job_api=types.SimpleNamespace(router_available=available))
        ns['component_status'](target)
        opener.assert_called_once_with('http://llama:8080/health', timeout=2)
        self.assertEqual(target.send_json.call_args.args[1]['model'],
                         {'name': None, 'runtime': 'llama.cpp'})
        return target.send_json.call_args.args[1]['components'], manager

    def test_online_components(self):
        states, manager = self.status()
        self.assertTrue(all(v == 'online' for v in states.values()))
        manager.list_conversations.assert_called_once_with(limit=1)

    def test_disabled_storage_is_not_probed(self):
        states, manager = self.status(enabled=False)
        self.assertEqual(states['persistence'], 'disabled')
        manager.list_conversations.assert_not_called()

    def test_failures_are_independent(self):
        states, _ = self.status(storage_error=True, model_error=True, router=False)
        self.assertEqual(states, dict(analyst='online', router='offline', model='offline', persistence='offline'))

    def test_deployed_api_reads_worker_readiness(self):
        states, _ = self.status(router=False, worker=True)
        self.assertEqual(states['router'], 'online')

    def test_stale_worker_is_offline_even_with_local_router(self):
        states, _ = self.status(router=True, worker=False)
        self.assertEqual(states['router'], 'offline')

    def test_worker_status_failure_is_isolated(self):
        states, _ = self.status(worker=OSError('unavailable'))
        self.assertEqual(states['router'], 'offline')
        self.assertEqual(states['model'], 'online')

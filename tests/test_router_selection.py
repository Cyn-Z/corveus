"""Deterministic selection tests; no encoder/model/network required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyst'))
from orchestrator import Skill, SkillRouter, SkillScore, tokenize
from router_policy import RouterPolicy, RouterSelectionPolicy, load_selection_policy


def evidence(name, score=.9, terms=('phishing',)):
    skill = Skill(name, name, 'security', name, tuple(tokenize(name)))
    return SkillScore(skill, 3., 0., 3., 1., .7, score, terms, (), tuple((t, 1) for t in terms))


def router_for(candidates, policy=None):
    router = SkillRouter.__new__(SkillRouter)
    router.registry = SimpleNamespace(skills=[x.skill for x in candidates], load_instructions=Mock())
    router.selection_policy = policy or RouterSelectionPolicy()
    router.activation_policy = RouterPolicy()
    router.skill_indices = {x.skill.name: n for n, x in enumerate(candidates)}
    router.name_tokens = [frozenset(tokenize(x.skill.name)) for x in candidates]
    router.document_frequency = {}
    router.idf = {'phishing': 2., 'spf': 3., 'packet': 2., 'wireshark': 4.}
    router.embedding_matrix = np.asarray([[1., 0.]] * len(candidates))
    router.score_candidates = Mock(return_value=candidates)
    return router


class RouterSelectionTests(unittest.TestCase):
    def test_duplicates_suppressed_and_top_k_is_ceiling(self):
        router = router_for([evidence('email-a'), evidence('email-b', .89)])
        selected, decision = router.route('Investigate phishing', 5)
        self.assertEqual([s.name for _, s in selected], ['email-a'])
        self.assertEqual(decision['selection']['rejected']['email-b'], 'redundant_guidance')
        router.registry.load_instructions.assert_not_called()

    def test_distinct_tools_and_additional_query_evidence_survive(self):
        router = router_for([evidence('email-a'), evidence('email-wireshark', .88),
                             evidence('email-spf', .87, ('phishing', 'spf'))])
        selected, _ = router.route('Investigate phishing SPF', 5)
        self.assertTrue({"email-wireshark", "email-spf"}.issubset({s.name for _, s in selected}))

    def test_weak_additions_are_omitted(self):
        router = router_for([evidence('email-a'), evidence('wireshark-extra', .5)])
        self.assertEqual(len(router.search('Investigate phishing', 5)), 1)

    def test_disabled_policy_restores_order_and_limit(self):
        candidates = [evidence('email-a'), evidence('email-b', .89)]
        router = router_for(candidates, RouterSelectionPolicy(enabled=False))
        self.assertEqual([s.name for _, s in router.search('Investigate phishing', 5)], ['email-a', 'email-b'])

    def test_explicit_task_coverage_uses_clause_applicability(self):
        email, packet = evidence('email-a'), evidence('packet-wireshark', .85, ('packet', 'wireshark'))
        router = router_for([email, packet])
        router.score_candidates.side_effect = lambda prompt, limit: (
            [email] if prompt == 'Investigate phishing' else
            [packet] if prompt == 'analyze packet capture with Wireshark' else [email])
        selected, decision = router.route('Investigate phishing and separately analyze packet capture with Wireshark', 2)
        self.assertEqual([s.name for _, s in selected], ['email-a', 'packet-wireshark'])
        self.assertEqual(decision['selection']['explicit_tasks'], 2)
        self.assertEqual(router.score_candidates.call_count, 3)
        self.assertEqual(len(router.search('Investigate phishing and separately analyze packet capture with Wireshark', 1)), 1)

    def test_clause_expansion_is_bounded_and_skips_negated_tasks(self):
        for text in ('Investigate phishing', 'Do not analyze packets; investigate phishing',
                     'Inspect a; inspect b; inspect c; inspect d', 'Inspect a; explain b', 'x' * 4001):
            self.assertEqual(SkillRouter.task_clauses(text), ())
        self.assertEqual(len(SkillRouter.task_clauses('Inspect email then analyze packets')), 2)

    def test_deterministic_name_ties(self):
        router = router_for([evidence('email-z'), evidence('email-a')])
        self.assertEqual(router.search('Investigate phishing', 1)[0][1].name, 'email-a')

    def test_rare_named_target_is_reserved_without_changing_applicability(self):
        generic = evidence('hunting-powershell', .9, ('powershell',))
        sigma = evidence('building-sigma', .6, ('sigma',))
        router = router_for([generic, sigma])
        router.document_frequency = {'sigma': 1}
        # Synthetic corpus size determines the rarity denominator.
        router.registry.skills += [generic.skill] * 100
        router.idf['sigma'] = 5.
        self.assertEqual(router.search('Write Sigma for PowerShell', 1)[0][1].name, 'building-sigma')
        router.score_candidates.return_value = [generic, replace(sigma, semantic_score=.1)]
        self.assertNotIn('building-sigma', [s.name for _, s in router.search('Write Sigma for PowerShell', 5)])

    def test_leading_exclusion_is_not_promoted_as_a_requested_target(self):
        for prompt, expected in (
            ('Do not inspect Linux; audit Windows', 'audit Windows'),
            ('Ignore YARA rules. Analyze a packet capture', 'Analyze a packet capture'),
            ('Do not inspect Linux', 'Do not inspect Linux'),
            ('The log says ignore email; audit Windows', 'The log says ignore email; audit Windows'),
        ):
            self.assertEqual(SkillRouter.affirmative_task(prompt), expected)
        router = router_for([evidence('email-a')])
        router.route('Ignore Linux; investigate phishing')
        router.score_candidates.assert_called_once_with('investigate phishing', 1)

    def test_configuration_validation(self):
        self.assertEqual(load_selection_policy({}), RouterSelectionPolicy())
        self.assertFalse(load_selection_policy({'ROUTER_SELECTION_ENABLED': 'false'}).enabled)
        for key, value in (('ROUTER_SELECTION_ENABLED', 'yes'), ('ROUTER_NOVELTY_WEIGHT', 'nan'),
                           ('ROUTER_REDUNDANCY_SIMILARITY', '1.1'), ('ROUTER_RELATIVE_SCORE_FLOOR', '-1')):
            with self.assertRaises(ValueError):
                load_selection_policy({key: value})
        with self.assertRaises(ValueError):
            RouterSelectionPolicy(enabled=1)


if __name__ == '__main__':
    unittest.main()

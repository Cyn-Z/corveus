"""Bounded long-request retrieval using existing scoring and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from router_query_plan import QueryPlan

if TYPE_CHECKING:
    from orchestrator import SkillRouter, SkillScore


def route_long_request(
    router: SkillRouter,
    plan: QueryPlan,
    *,
    limit: int,
    include_terms: bool = False,
    explain: bool = True,
    leading_exclusion_omitted: bool = False,
):
    strongest: dict[str, SkillScore] = {}
    winning_views: dict[str, int] = {}
    directive_candidates = []
    view_counts = []

    for index, view in enumerate(plan.views):
        ranked = router.score_candidates(
            view.text,
            len(router.registry.skills),
        )
        accepted = [item for item in ranked if router.activation_policy.accepts(item)]

        for item in accepted:
            name = item.skill.name
            previous = strongest.get(name)

            # Keep one complete score record. Never combine lexical evidence
            # from one view with semantic evidence from another.
            if previous is None or item.combined_score > previous.combined_score:
                strongest[name] = item
                winning_views[name] = index

        if "directive" in view.sources:
            directive_candidates.append((view.text, accepted))

        view_counts.append(
            {
                "view_index": index,
                "accepted_candidates": len(accepted),
            }
        )

    eligible = sorted(
        strongest.values(),
        key=lambda item: (-item.combined_score, item.skill.name),
    )

    if router.selection_policy.enabled:
        focused_text = "\n".join(text for text, _ in directive_candidates)
        if not focused_text:
            focused_text = "\n".join(view.text for view in plan.views)

        selected, selection_details = router._select_diverse(
            focused_text,
            eligible,
            limit,
            task_candidates=directive_candidates,
        )

        # Reservations may select a skill using a directive's score.
        # Return its strongest eligible evidence consistently.
        selected = [strongest[item.skill.name] for item in selected]
    else:
        selected = eligible[:limit]
        selection_details = {"enabled": False}

    selection_details["leading_exclusion_omitted"] = leading_exclusion_omitted

    pairs = [(item.combined_score, item.skill) for item in selected]
    if not explain:
        return pairs, {}

    selected_names = {item.skill.name for item in selected}
    visible = [
        item
        for rank, item in enumerate(eligible)
        if rank < limit or item.skill.name in selected_names
    ]

    planning = plan.diagnostic()
    planning["embedding_queries"] = len(plan.views)
    planning["view_results"] = view_counts
    planning["aggregation"] = "strongest_eligible_view"

    return pairs, {
        "reason": ("skills_selected" if selected else "no_relevant_skills"),
        "selection": selection_details,
        "query_plan": planning,
        "candidates": [
            {
                **item.diagnostic(include_terms=include_terms),
                "activation_reason": (router.activation_policy.candidate_reason(item)),
                "selected": item.skill.name in selected_names,
                "winning_view": winning_views[item.skill.name],
            }
            for item in visible
        ],
    }

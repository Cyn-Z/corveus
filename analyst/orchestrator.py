from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import os
from collections import Counter
from router_policy import RouterPolicy, RouterSelectionPolicy, GENERIC_ACTIVATION_TERMS
from router_query_plan import (
    QueryPlanner,
    QueryPlanSettings,
    add_query_plan_arguments,
    query_plan_settings_from_arguments,
)
from router_long_requests import route_long_request


@dataclass(frozen=True)
class Skill:
    """
    Represents one skill discovered in the cybersecurity skills repository.
    This stores only lightweight metadata and a path to the skill file.
    The full SKILL.md instructions are not loaded until a skill is selected.
    frozen=True prevents accidental modification after the Skill is created.
    """

    name: str
    description: str
    domain: str
    path: str
    tokens: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    STOP_WORDS = {
        # Articles
        "a",
        "an",
        "the",
        # Pronouns
        "i",
        "me",
        "my",
        "myself",
        "you",
        "your",
        "yourself",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
        # Auxiliary verbs and conversational filler
        "am",
        "are",
        "be",
        "been",
        "being",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "please",
        # Common question words
        "how",
        "what",
        "when",
        "where",
        "who",
        "why",
        # Common connectors
        "and",
        "but",
        "or",
        "if",
        "so",
        "the",
        "than",
        "then",
        # Common prepositions
        "at",
        "in",
        "of",
        "on",
        "to",
        # Generic request language
        "tell",
        "please",
        "really",
        "just",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


class SkillRegistry:
    """
    Discovers skills from the repository's existing index.json.
    The registry is responsible for:
    - Reading the index.
    - Validating basic metadata and skill paths.
    - Providing access to individual skills.
    - Loading the full instructions for a selected skill.
    It does not decide which skills are relevant; that is the router's job.
    """

    def __init__(self, repo_root: str | Path):
        """
        Initialize the registry using the local repository directory.
        Args:
            repo_root: Path to the root of the cybersecurity skills
                       repository. This directory should contain index.json.
        The path is resolved to an absolute path so that subsequent path
        validation does not depend on the process's working directory.
        """
        self.root = Path(repo_root).resolve()
        self.index_path = self.root / "index.json"

        # Read the metadata catalog and build the in-memory skill registry.
        self.skills = self._load_index()

    def _load_index(self) -> list[Skill]:
        """
        Read index.json and create Skill objects for valid entries.
        Only metadata is retained in memory. The contents of individual
        SKILL.md files are deliberately not loaded at this stage.
        Entries are skipped when required fields are missing, paths escape
        the repository, or the corresponding SKILL.md file does not exist.
        Raises:
            FileNotFoundError: If index.json does not exist.
            json.JSONDecodeError: If index.json contains invalid JSON.
            ValueError: If duplicate skill names are encountered.
        """
        data = json.loads(self.index_path.read_text(encoding="utf-8"))

        skills = []
        seen = set()

        # The repository index is expected to contain a "skills" array.
        for item in data.get("skills", []):
            name = item.get("name")
            description = item.get("description", "")
            relative_path = item.get("path", "")

            # A skill needs a name and a path to be usable.
            if not name or not relative_path:
                continue

            # Resolve the path and reject paths that escape the repository.
            # This helps prevent malicious or malformed index entries from
            # directing the loader to files elsewhere on the filesystem.
            skill_dir = (self.root / relative_path).resolve()

            if not skill_dir.is_relative_to(self.root):
                continue

            # Each indexed skill must have its expected instruction file.
            skill_file = skill_dir / "SKILL.md"

            if not skill_file.is_file():
                continue

            # Skill names identify entries throughout the orchestrator.
            # Reject duplicates rather than silently selecting the wrong one.
            if name in seen:
                raise ValueError(f"Duplicate skill name: {name}")

            seen.add(name)

            # Include the domain in the searchable text so that prompts
            # can match relevant security areas as well as skill names.
            domain = item.get("domain", "")
            searchable = f"{name} {description} {domain}"

            # Cache the tokenized metadata for fast repeated searches.
            skills.append(
                Skill(
                    name=name,
                    description=description,
                    domain=domain,
                    path=relative_path,
                    tokens=tuple(tokenize(searchable)),
                )
            )

        return skills

    def get(self, name: str) -> Skill:
        """
        Return a registered skill by its exact name.
        Args:
            name: The original skill name from the repository index.
        Raises:
            KeyError: If no registered skill has that name.
        This is useful when another component needs to resolve a selected
        skill into its metadata and file path.
        """
        for skill in self.skills:
            if skill.name == name:
                return skill

        raise KeyError(f"Unknown skill: {name}")

    def load_instructions(self, skill: Skill) -> str:
        """
        Load the full instructions for one selected skill.
        This method implements lazy loading: the orchestrator reads a skill's
        full file only when the skill is needed, reducing unnecessary file
        reads and preventing unrelated instructions from entering the LLM
        context.
        The YAML frontmatter is removed when the file begins with a valid
        opening and closing delimiter. The remaining Markdown is returned.
        Args:
            skill: A Skill object previously discovered by this registry.
        Raises:
            ValueError: If the skill path escapes the repository.
            OSError: If the skill file cannot be read.
        """
        skill_dir = (self.root / skill.path).resolve()

        # Validate the path again before opening a file. This ensures the
        # loader does not blindly trust a Skill object's stored path.
        if not skill_dir.is_relative_to(self.root):
            raise ValueError("Skill path escapes repository")

        skill_file = skill_dir / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")

        # Remove the frontmatter so the LLM receives the actual instructions
        # rather than repeating metadata already available to the router.
        if content.startswith("---"):
            parts = content.split("---", 2)

            if len(parts) == 3:
                return parts[2].strip()

        # Some skills may not have frontmatter, so return the whole file.
        return content.strip()


@dataclass(frozen=True)
class SkillScore:
    """Ranking evidence, not a calibrated probability or activation decision."""

    skill: Skill
    bm25_score: float
    name_boost_score: float
    raw_lexical_score: float
    normalized_lexical_score: float
    semantic_score: float
    combined_score: float
    matched_terms: tuple[str, ...]
    matched_name_terms: tuple[str, ...]
    matched_term_document_frequencies: tuple[tuple[str, int], ...]
    catalog_size: int = 0

    def diagnostic(self, *, include_terms: bool = False) -> dict:
        """Omit query-derived words by default; never include instructions."""
        result = {
            "skill": self.skill.name,
            "bm25_score": self.bm25_score,
            "name_boost_score": self.name_boost_score,
            "raw_lexical_score": self.raw_lexical_score,
            "normalized_lexical_score": self.normalized_lexical_score,
            "semantic_score": self.semantic_score,
            "combined_score": self.combined_score,
            "matched_term_count": len(self.matched_terms),
            "matched_name_term_count": len(self.matched_name_terms),
            "passes_current_score_floor": self.combined_score > 0,
        }
        if include_terms:
            result["matched_terms"] = list(self.matched_terms)
            result["matched_name_terms"] = list(self.matched_name_terms)
            result["matched_term_document_frequencies"] = dict(
                self.matched_term_document_frequencies
            )
        return result


class SkillRouter:
    """
    Hybrid skill router combining BM25 lexical retrieval with
    MiniLM semantic similarity.

    The embedding model is loaded once per router instance.
    Skill embeddings are precomputed during initialization.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        model_path: str | Path,
        semantic_weight: float,
        name_boost: float,
        activation_policy: RouterPolicy | None = None,
        selection_policy: RouterSelectionPolicy | None = None,
        query_plan_settings: QueryPlanSettings | None = None,
    ):
        self.registry = registry
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        self.name_boost = name_boost
        self.activation_policy = (
            activation_policy if activation_policy is not None else RouterPolicy()
        )
        self.selection_policy = (
            selection_policy
            if selection_policy is not None
            else RouterSelectionPolicy()
        )
        self.skill_indices = {skill.name: i for i, skill in enumerate(registry.skills)}
        # Configure the local model path for Docker or local development.
        model_path = Path(model_path).resolve()

        if not model_path.is_dir():
            raise FileNotFoundError(f"MiniLM model directory not found: {model_path}")

        # Fail at startup if the local model or its dependencies are
        # unavailable. Never download a replacement model.
        self.model = self._load_embedding_model(model_path)
        self.query_plan_settings = (
            query_plan_settings
            if query_plan_settings is not None
            else QueryPlanSettings()
        )
        self.query_planner = (
            QueryPlanner(
                self.model.tokenizer,
                model_max_seq_length=self.model.max_seq_length,
                settings=self.query_plan_settings,
            )
            if self.query_plan_settings.enabled
            else None
        )

        # Prepare the lexical index.
        self.documents = [list(skill.tokens) for skill in registry.skills]

        self.avg_length = (
            sum(map(len, self.documents)) / len(self.documents)
            if self.documents
            else 0.0
        )

        self.document_frequency = {}

        for document in self.documents:
            for term in set(document):
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

        # Catalog-only BM25 work is invariant across requests.
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.name_tokens = [
            frozenset(tokenize(skill.name.replace("-", " ")))
            for skill in registry.skills
        ]
        self.document_lengths = [len(document) for document in self.documents]
        self.bm25_length_factors = [
            1.5 * (1 - 0.75 + 0.75 * length / (self.avg_length or 1))
            for length in self.document_lengths
        ]
        self.idf = {
            term: math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            for term, df in self.document_frequency.items()
        }

        # Use richer, human-readable metadata for semantic retrieval.
        # Do not embed the full SKILL.md instructions.
        self.skill_texts = [
            (
                f"Skill: {skill.name}. "
                f"Domain: {skill.domain}. "
                f"Description: {skill.description}"
            )
            for skill in registry.skills
        ]

        # Precompute normalized skill embeddings once at startup.
        # Keep the public tuple vectors for diagnostics/regression callers.
        if self.skill_texts:
            encoded = self.model.encode(
                self.skill_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.skill_embeddings = [
                tuple(float(value) for value in vector) for vector in encoded
            ]
        else:
            self.skill_embeddings = []

        # NumPy is already a dependency of the local embedding stack. Float64
        # preserves the precision of the former Python-float dot products.
        import numpy as np

        self.embedding_matrix = np.asarray(self.skill_embeddings, dtype=np.float64)
        if self.skill_embeddings:
            self.embedding_matrix.setflags(write=False)

        print(
            f"Hybrid router initialized: "
            f"{len(registry.skills)} skills indexed; "
            f"MiniLM embeddings cached.",
            flush=True,
        )

    @staticmethod
    def _load_embedding_model(model_path: Path):
        # Keep scoring tests independent of optional model packages. Runtime
        # still uses the same local-only model and fails if it is unavailable.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(str(model_path), device="cpu", local_files_only=True)

    def search(self, prompt: str, limit: int = 8):
        """
        Return (score, Skill) tuples ranked using both BM25 and MiniLM.

        The combined score is a ranking value, not a probability.
        """

        selected, _ = self.route(prompt, limit, explain=False)
        return selected

    def route(self, prompt: str, limit: int = 8, *, include_terms=False, explain=True):
        """Return selected (score, Skill) pairs and an opt-in explanation.

        Apply applicability before top-k so a rejected high-ranking candidate
        cannot displace an eligible lower-ranked skill. No logging is added.
        """
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or limit <= 0
            or not self.registry.skills
        ):
            return [], {"reason": "empty_request_or_catalog", "candidates": []}

        bypass = self.activation_policy.bypass_reason(prompt)
        if bypass is not None:
            return [], {"reason": bypass, "candidates": []}

        selection = getattr(
            self, "selection_policy", RouterSelectionPolicy(enabled=False)
        )
        selection_prompt = (
            self.affirmative_task(prompt) if selection.enabled else prompt
        )
        planner = getattr(self, "query_planner", None)
        plan = planner.plan(selection_prompt) if planner is not None else None

        if plan is not None and plan.mode == "chunked":
            return route_long_request(
                self,
                plan,
                limit=limit,
                include_terms=include_terms,
                explain=explain,
                leading_exclusion_omitted=selection_prompt != prompt,
            )

        candidates = self.score_candidates(
            selection_prompt,
            len(self.registry.skills),
        )
        eligible = [item for item in candidates if self.activation_policy.accepts(item)]
        if selection.enabled:
            selected, selection_details = self._select_diverse(
                selection_prompt, eligible, limit
            )
            selection_details["leading_exclusion_omitted"] = selection_prompt != prompt
        else:
            selected = eligible[:limit]
            selection_details = {"enabled": False}

        if not explain:
            return [(item.combined_score, item.skill) for item in selected], {}
        selected_names = {item.skill.name for item in selected}
        selected_evidence = {item.skill.name: item for item in selected}
        # Always explain selected candidates, even if filtering promoted one
        # from outside the initial top-k ranking window.
        visible = [
            item
            for rank, item in enumerate(candidates)
            if rank < limit or item.skill.name in selected_names
        ]
        planning = plan.diagnostic() if plan is not None else {"mode": "disabled"}
        planning["embedding_queries"] = 1 + selection_details.get("explicit_tasks", 0)
        explanation = {
            "reason": "skills_selected" if selected else "no_relevant_skills",
            "selection": selection_details,
            "query_plan": planning,
            "candidates": [
                {
                    **selected_evidence.get(item.skill.name, item).diagnostic(
                        include_terms=include_terms
                    ),
                    "activation_reason": self.activation_policy.candidate_reason(
                        selected_evidence.get(item.skill.name, item)
                    ),
                    "selected": item.skill.name in selected_names,
                }
                for item in visible
            ],
        }
        return [(item.combined_score, item.skill) for item in selected], explanation

    @staticmethod
    def affirmative_task(prompt):
        """Omit only an explicit leading exclusion followed by a new task.

        The original request still reaches inference unchanged. No attempt is
        made to understand arbitrary negation inside logs or compound prose.
        """
        match = re.fullmatch(
            r"\s*(?:do not|don't|ignore)\b[^.;\n]+[.;\n]\s*(.+)",
            prompt,
            flags=re.I | re.S,
        )
        if match and re.match(
            r"(?:investigate|analy[sz]e|audit|check|inspect|examine|triage|hunt|detect|write|configure)\b",
            match.group(1),
            re.I,
        ):
            return match.group(1)
        return prompt

    @staticmethod
    def task_clauses(prompt):
        """Bounded explicit task separation, never an LLM rewrite or log chunker."""
        if len(prompt) > 4000:
            return ()
        parts = re.split(
            r"\b(?:and\s+separately|separately|then)\b|;", prompt, flags=re.I
        )
        if not 2 <= len(parts) <= 3:
            return ()
        action = re.compile(
            r"\b(?:investigate|analy[sz]e|audit|check|inspect|triage|hunt|detect|write|configure)\b",
            re.I,
        )
        if any(
            re.search(r"\b(?:do not|don't|ignore|without|instead of)\b", part, re.I)
            for part in parts
        ):
            return ()
        if not all(action.search(part) for part in parts):
            return ()
        return tuple(part.strip(" ,.") for part in parts)

    def _select_diverse(
        self,
        prompt,
        eligible,
        limit,
        *,
        task_candidates=None,
    ):
        policy = self.selection_policy
        selected = []
        rejected = {}
        by_name = {item.skill.name: item for item in eligible}
        tasks = []
        task_selections = []
        tools = {
            "splunk",
            "wireshark",
            "tshark",
            "volatility",
            "volatility3",
            "sigma",
            "yara",
            "windows",
            "linux",
            "kubernetes",
            "aws",
            "azure",
        }

        def priority(item, text):
            terms = set(tokenize(text)) - GENERIC_ACTIVATION_TERMS
            requested_tools = terms & tools
            names = self.name_tokens[self.skill_indices[item.skill.name]]
            total = sum(self.idf.get(term, 0.0) for term in terms) or 1.0
            specificity = sum(self.idf.get(term, 0.0) for term in names & terms) / total
            tool_coverage = len(names & requested_tools) / (len(requested_tools) or 1)
            return item.combined_score + 0.15 * specificity + 0.15 * tool_coverage

        def seed(pool, text):
            terms = set(tokenize(text)) - GENERIC_ACTIVATION_TERMS
            requested = terms & tools
            if requested:
                maximum = max(
                    (
                        len(
                            self.name_tokens[self.skill_indices[x.skill.name]]
                            & requested
                        )
                        for x in pool
                    ),
                    default=0,
                )
                if maximum:
                    pool = [
                        x
                        for x in pool
                        if len(
                            self.name_tokens[self.skill_indices[x.skill.name]]
                            & requested
                        )
                        == maximum
                    ]
            return (
                min(pool, key=lambda x: (-priority(x, text), x.skill.name))
                if pool
                else None
            )

        # Long requests supply already-scored directive candidates.
        # An empty supplied list also prevents additional encoding.
        if task_candidates is None:
            task_candidates = [
                (
                    clause,
                    self.score_candidates(
                        clause,
                        len(self.registry.skills),
                    ),
                )
                for clause in self.task_clauses(prompt)
            ]

        # Preserve explicit task coverage before filling remaining slots.
        for clause, ranked in task_candidates:
            accepted = [item for item in ranked if self.activation_policy.accepts(item)]
            preferred = seed(accepted, clause)
            if preferred:
                accepted = [preferred] + [
                    x for x in accepted if x.skill.name != preferred.skill.name
                ]
            tasks.append(accepted)
            by_name.update(
                {
                    item.skill.name: by_name.get(item.skill.name, item)
                    for item in accepted
                }
            )
        for accepted in tasks:
            for item in accepted:
                if len(selected) >= limit:
                    break
                if item.skill.name not in {x.skill.name for x in selected}:
                    selected.append(item)
                    task_selections.append(item.skill.name)
                    break
        # Reserve exact rare named targets (for example Sigma or Kerberoasting)
        # when eligible guidance exists; never promote an inapplicable skill.
        terms = set(tokenize(prompt)) - GENERIC_ACTIVATION_TERMS
        artifact_terms = {
            "dump",
            "dumps",
            "log",
            "logs",
            "event",
            "events",
            "header",
            "headers",
            "email",
            "emails",
            "packet",
            "packets",
            "capture",
            "commands",
            "processes",
        }
        anchors = sorted(
            (
                term
                for term in terms - artifact_terms
                if self.document_frequency.get(term, len(self.registry.skills))
                <= len(self.registry.skills) * 0.03
            ),
            key=lambda term: (-self.idf.get(term, 0.0), term),
        )
        for term in anchors[:3]:
            pool = [
                x
                for x in eligible
                if term in self.name_tokens[self.skill_indices[x.skill.name]]
            ]
            if re.search(r"\b(?:detect|hunt|audit)\b", prompt, re.I):
                defensive = [
                    x
                    for x in pool
                    if re.match(
                        r"(?:detecting|hunting|auditing|analyzing)-", x.skill.name
                    )
                ]
                pool = defensive or pool
            item = seed(pool, prompt)
            if (
                item
                and len(selected) < limit
                and item.skill.name not in {x.skill.name for x in selected}
            ):
                selected.append(item)
        if not selected and eligible:
            selected.append(seed(eligible, prompt))
        if not selected:
            return [], {"enabled": True, "explicit_tasks": len(tasks), "rejected": {}}
        # Tool/platform distinctions are intentionally conservative: never
        # suppress a candidate as redundant solely across these differences.
        query_terms = set(tokenize(prompt)) - GENERIC_ACTIVATION_TERMS
        best_score = max(item.combined_score for item in selected)
        covered = set().union(
            *(set(item.matched_terms) & query_terms for item in selected)
        )
        remaining = [
            item
            for item in by_name.values()
            if item.skill.name not in {s.skill.name for s in selected}
        ]
        while remaining and len(selected) < limit:
            choices = []
            for item in remaining:
                index = self.skill_indices[item.skill.name]
                similarities = []
                redundant = False
                for prior in selected:
                    other = self.skill_indices[prior.skill.name]
                    similarity = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                self.embedding_matrix[index]
                                @ self.embedding_matrix[other]
                            ),
                        ),
                    )
                    similarities.append(similarity)
                    same_tools = (self.name_tokens[index] & tools) == (
                        self.name_tokens[other] & tools
                    )
                    new_evidence = (set(item.matched_terms) & query_terms) - covered
                    if (
                        similarity >= policy.redundancy_similarity
                        and same_tools
                        and not new_evidence
                    ):
                        redundant = True
                if redundant:
                    rejected[item.skill.name] = "redundant_guidance"
                    continue
                if item.combined_score < best_score * policy.relative_score_floor:
                    rejected[item.skill.name] = "limited_additional_relevance"
                    continue
                marginal = priority(item, prompt) - policy.novelty_weight * max(
                    similarities
                )
                choices.append((marginal, item))
            if not choices:
                break
            choices.sort(key=lambda pair: (-pair[0], pair[1].skill.name))
            item = choices[0][1]
            if choices[0][0] < best_score * max(
                0.0, policy.relative_score_floor - policy.novelty_weight
            ):
                break
            selected.append(item)
            covered.update(set(item.matched_terms) & query_terms)
            remaining = [
                candidate
                for candidate in remaining
                if candidate.skill.name != item.skill.name
            ]
        return selected, {
            "enabled": True,
            "explicit_tasks": len(tasks),
            "task_selections": task_selections,
            "rejected": rejected,
        }

    def score_candidates(self, prompt: str, limit: int = 8) -> list[SkillScore]:
        """Expose the existing ranking, including zero-score candidates.

        No instructions are loaded and no diagnostic data is logged. The
        ranking formula is unchanged. This method deliberately bypasses the
        activation policy for offline score inspection.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            return []

        if limit <= 0 or not self.registry.skills:
            return []

        # Encode the user's request once.
        query_vector = self.model.encode(
            prompt,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        import numpy as np

        query_vector = np.asarray(query_vector, dtype=np.float64)
        if (
            query_vector.ndim != 1
            or query_vector.shape[0] != self.embedding_matrix.shape[1]
        ):
            raise ValueError(
                "Query embedding dimension does not match skill embeddings"
            )
        # A matrix-vector contraction avoids per-component Python loops and
        # does not dispatch tiny catalog operations to a threaded BLAS pool.
        semantic_scores = np.clip(
            np.einsum("ij,j->i", self.embedding_matrix, query_vector, optimize=False),
            0.0,
            1.0,
        )

        # Calculate BM25 scores.
        query_terms = set(tokenize(prompt))
        total_docs = len(self.documents)

        lexical_scores = []
        lexical_evidence = []

        for index, skill in enumerate(self.registry.skills):
            frequencies = self.term_frequencies[index]
            score = 0.0

            for term in query_terms:
                tf = frequencies.get(term, 0)

                if not tf:
                    continue

                denominator = tf + self.bm25_length_factors[index]
                score += self.idf[term] * tf * 2.5 / denominator

            # Apply the skill-name boost once per skill, not once per term.
            name_tokens = self.name_tokens[index]
            matched_name_terms = tuple(sorted(query_terms & name_tokens))
            boosted_terms = matched_name_terms
            if self.selection_policy.enabled:
                boosted_terms = tuple(
                    term
                    for term in matched_name_terms
                    if term not in GENERIC_ACTIVATION_TERMS
                )
            name_boost_score = self.name_boost * len(boosted_terms)
            matched_terms = tuple(sorted(query_terms & frequencies.keys()))
            lexical_evidence.append(
                (score, name_boost_score, matched_terms, matched_name_terms)
            )
            score += name_boost_score

            lexical_scores.append(score)

        # Normalize lexical scores to [0, 1] for combination with cosine.
        max_lexical = max(lexical_scores, default=0.0)

        if max_lexical > 0:
            lexical_scores = [score / max_lexical for score in lexical_scores]
        else:
            lexical_scores = [0.0] * len(lexical_scores)

        results = []

        for index, skill in enumerate(self.registry.skills):
            semantic = float(semantic_scores[index])
            lexical = lexical_scores[index]

            combined = self.semantic_weight * semantic + self.lexical_weight * lexical
            # Matrix reduction can differ by a few ulps. Preserve the former
            # scalar arithmetic at activation boundaries instead of allowing
            # numerical noise to decide applicability.
            policy = self.activation_policy
            if (
                abs(semantic - policy.min_semantic_score) <= 1e-12
                or abs(semantic - policy.semantic_only_score) <= 1e-12
                or abs(combined - policy.min_combined_score) <= 1e-12
            ):
                semantic = max(
                    0.0,
                    min(
                        1.0,
                        sum(
                            float(a) * b
                            for a, b in zip(query_vector, self.skill_embeddings[index])
                        ),
                    ),
                )
                combined = (
                    self.semantic_weight * semantic + self.lexical_weight * lexical
                )

            bm25, boost, terms, name_terms = lexical_evidence[index]
            results.append(
                SkillScore(
                    skill=skill,
                    bm25_score=bm25,
                    name_boost_score=boost,
                    raw_lexical_score=bm25 + boost,
                    normalized_lexical_score=lexical,
                    semantic_score=semantic,
                    combined_score=combined,
                    catalog_size=total_docs,
                    matched_terms=terms,
                    matched_name_terms=name_terms,
                    matched_term_document_frequencies=tuple(
                        (term, self.document_frequency.get(term, 0)) for term in terms
                    ),
                )
            )

        results.sort(key=lambda result: (-result.combined_score, result.skill.name))

        return results[:limit]


def main():
    """
    Provide a command-line interface for testing the local orchestrator.
    The user supplies the local repository path and optionally the maximum
    number of results. The function loads the index, asks for a prompt,
    prints ranked skills, and optionally prints their full instructions.
    This CLI is for testing the registry and router before integrating them
    into the existing LLM tool-calling pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Search a local cybersecurity skills repository."
    )

    # The repository path is explicit so the program never needs to clone
    # a repository or access the internet automatically.
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the local cybersecurity skills repository",
    )

    # Limit the number of candidates displayed or loaded.
    parser.add_argument("--top", type=int, default=8)

    # Print full instructions only when explicitly requested.
    parser.add_argument(
        "--load",
        action="store_true",
        help="Print selected skill instructions too",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--semantic-weight", type=float, default=0.60)
    parser.add_argument("--name-boost", type=float, default=3.0)
    parser.add_argument("--legacy-routing", action="store_true")
    parser.add_argument("--legacy-selection", action="store_true")
    parser.add_argument("--min-semantic-score", type=float, default=0.36)
    parser.add_argument("--min-combined-score", type=float, default=0.45)
    parser.add_argument("--semantic-only-score", type=float, default=0.65)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show activation decisions and scores without loading instructions",
    )
    parser.add_argument(
        "--show-terms",
        action="store_true",
        help="Include query-derived matching terms in diagnostics",
    )
    add_query_plan_arguments(parser)
    args = parser.parse_args()
    if args.show_terms and not args.diagnostics:
        parser.error("--show-terms requires --diagnostics")
    try:
        query_settings = query_plan_settings_from_arguments(args)
        policy = RouterPolicy(
            enabled=not args.legacy_routing,
            min_semantic_score=args.min_semantic_score,
            min_combined_score=args.min_combined_score,
            semantic_only_score=args.semantic_only_score,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Build the catalog and search index from the local repository.
    registry = SkillRegistry(args.repo)
    router = SkillRouter(
        registry,
        model_path=args.model_path,
        semantic_weight=args.semantic_weight,
        name_boost=args.name_boost,
        activation_policy=policy,
        selection_policy=RouterSelectionPolicy(enabled=not args.legacy_selection),
        query_plan_settings=query_settings,
    )

    print(f"Indexed {len(registry.skills)} skills.")

    # Interactive input lets us test different prompts without restarting
    # the program or editing any source code.
    prompt = input("\nDescribe the task: ").strip()

    selected, decision = router.route(prompt, args.top, include_terms=args.show_terms)
    if args.diagnostics:
        print(json.dumps(decision, indent=2))
        if not args.load:
            return

    # Retrieve and display the best matches.
    for score, skill in selected:
        print(f"\n{score:.3f}  {skill.name}")
        print(f"  Domain: {skill.domain}")
        print(f"  {skill.description}")

        # Full skill content is loaded only if the user explicitly requests
        # it, demonstrating the lazy-loading behavior of the registry.
        if args.load:
            print("\n--- SKILL INSTRUCTIONS ---")
            print(registry.load_instructions(skill))


if __name__ == "__main__":
    # Run the command-line interface only when this file is executed directly.
    # Importing the module from another Python file will not start the CLI.
    main()

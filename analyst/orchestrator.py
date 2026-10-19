from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer


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


'''
def tokenize(text: str) -> list[str]:
    """
    Normalize text into lowercase searchable tokens.
    For example:
        "Analyze Windows Event Logs!"
        becomes:
        ["analyze", "windows", "event", "logs"]
    This gives the router a consistent representation of both user prompts
    and skill metadata without requiring an external search library.
    """
    return re.findall(r"[a-z0-9]+", text.lower())
'''


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


class SkillRouter:
    """
    Hybrid skill router combining BM25 lexical retrieval with
    MiniLM semantic similarity.

    The embedding model is loaded once per router instance.
    Skill embeddings are precomputed during initialization.
    """

    def __init__(self, registry: SkillRegistry, *, model_path: str | Path, semantic_weight: float, name_boost: float,):
        self.registry = registry
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        self.name_boost = name_boost
        # Configure the local model path for Docker or local development.
        model_path = Path(model_path).resolve()

        if not model_path.is_dir():
            raise FileNotFoundError(f"MiniLM model directory not found: {model_path}")

        # Fail at startup if the local model or its dependencies are
        # unavailable. Never download a replacement model.
        self.model = SentenceTransformer(
            str(model_path),
            device="cpu",
            local_files_only=True,
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
        # Store vectors as tuples so the router needs no direct NumPy API.
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

        print(
            f"Hybrid router initialized: "
            f"{len(registry.skills)} skills indexed; "
            f"MiniLM embeddings cached.",
            flush=True,
        )

    def search(self, prompt: str, limit: int = 8):
        """
        Return (score, Skill) tuples ranked using both BM25 and MiniLM.

        The combined score is a ranking value, not a probability.
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
        query_vector = tuple(float(value) for value in query_vector)

        # Calculate cosine similarity using normalized vectors.
        # No direct NumPy operations are needed.
        semantic_scores = [
            max(
                0.0,
                min(
                    1.0,
                    sum(a * b for a, b in zip(query_vector, skill_vector)),
                ),
            )
            for skill_vector in self.skill_embeddings
        ]

        # Calculate BM25 scores.
        query_terms = set(tokenize(prompt))
        total_docs = len(self.documents)

        lexical_scores = []

        k1 = 1.5
        b = 0.75

        for skill, document in zip(self.registry.skills, self.documents):
            frequencies = {}

            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1

            length = len(document)
            score = 0.0

            for term in query_terms:
                tf = frequencies.get(term, 0)

                if not tf:
                    continue

                df = self.document_frequency.get(term, 0)

                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

                denominator = tf + k1 * (1 - b + b * length / (self.avg_length or 1))

                score += idf * tf * (k1 + 1) / denominator

            # Apply the skill-name boost once per skill, not once per term.
            prompt_tokens = set(tokenize(prompt))
            name_tokens = set(tokenize(skill.name.replace("-", " ")))
            score += self.name_boost * len(prompt_tokens & name_tokens)

            lexical_scores.append(score)

        # Normalize lexical scores to [0, 1] for combination with cosine.
        max_lexical = max(lexical_scores, default=0.0)

        if max_lexical > 0:
            lexical_scores = [score / max_lexical for score in lexical_scores]
        else:
            lexical_scores = [0.0] * len(lexical_scores)

        # Initial blend: favor semantic matching while retaining exact-term
        # matching. Tune these weights against representative test prompts.
        semantic_weight = 0.60
        lexical_weight = 0.40

        results = []

        for index, skill in enumerate(self.registry.skills):
            semantic = semantic_scores[index]
            lexical = lexical_scores[index]

            combined = (self.semantic_weight * semantic + self.lexical_weight * lexical)

            # Avoid returning unrelated skills merely because they exist.
            # This is a ranking floor, not a calibrated confidence value.
            if combined > 0:
                results.append((combined, skill))

        results.sort(key=lambda result: (-result[0], result[1].name))

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

    args = parser.parse_args()

    # Build the catalog and search index from the local repository.
    registry = SkillRegistry(args.repo)
    router = SkillRouter(registry, model_path=args.model_path, semantic_weight=args.semantic_weight, name_boost=args.name_boost, )

    print(f"Indexed {len(registry.skills)} skills.")

    # Interactive input lets us test different prompts without restarting
    # the program or editing any source code.
    prompt = input("\nDescribe the task: ").strip()

    # Retrieve and display the best matches.
    for score, skill in router.search(prompt, args.top):
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

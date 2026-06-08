import pytest

from episode_selection.budgets import Budget
from episode_selection.embeddings import DeterministicHashEmbedder
from episode_selection.selectors import LanguageSimilaritySelector, LanguageVisionDiversitySelector, QualityFilteredSelector, RandomSelector


def _meta():
    emb = DeterministicHashEmbedder(dim=16)
    texts = ["pick red block", "pick red cube", "open drawer"]
    vecs = emb.encode_text(texts).tolist()
    return [
        {"episode_id": "a", "storage_bytes": 10, "num_transitions": 2, "language_embedding": vecs[0], "multimodal_embedding": vecs[0], "vision_embedding": vecs[0], "quality_score": 0.2, "task": texts[0]},
        {"episode_id": "b", "storage_bytes": 10, "num_transitions": 2, "language_embedding": vecs[1], "multimodal_embedding": vecs[1], "vision_embedding": vecs[1], "quality_score": 0.9, "task": texts[1]},
        {"episode_id": "c", "storage_bytes": 10, "num_transitions": 2, "language_embedding": vecs[2], "multimodal_embedding": vecs[2], "vision_embedding": vecs[2], "quality_score": 0.5, "task": texts[2]},
    ]


EMBED_CFG = {"embedding_config": {"provider": "deterministic", "dim": 16, "allow_deterministic_fallback": True}}


def test_random_selector_deterministic():
    a = RandomSelector().select(_meta(), Budget("episodes", 2), seed=4).selected_episode_ids
    b = RandomSelector().select(_meta(), Budget("episodes", 2), seed=4).selected_episode_ids
    assert a == b


def test_language_similarity_ranks_query_match():
    result = LanguageSimilaritySelector().select(_meta(), Budget("episodes", 1), config={"queries": ["open drawer"], **EMBED_CFG})
    assert result.selected_episode_ids == ["c"]


def test_diversity_selector_no_duplicates():
    result = LanguageVisionDiversitySelector().select(
        _meta(), Budget("episodes", 3), config={"allow_language_only_fallback": False, **EMBED_CFG}
    )
    assert len(result.selected_episode_ids) == len(set(result.selected_episode_ids))


def test_diversity_requires_vision_without_fallback():
    meta = _meta()
    for m in meta:
        m["vision_embedding"] = None
    with pytest.raises(ValueError):
        LanguageVisionDiversitySelector().select(meta, Budget("episodes", 1), config={"allow_language_only_fallback": False})


def test_quality_selector_ranking():
    result = QualityFilteredSelector().select(_meta(), Budget("episodes", 2))
    assert result.selected_episode_ids == ["b", "c"]


def test_duplicate_metadata_ids_raise():
    meta = _meta()
    meta[1]["episode_id"] = meta[0]["episode_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        RandomSelector().select(meta, Budget("episodes", 1), seed=0)


def test_missing_language_embeddings_fail_clearly():
    meta = _meta()
    meta[0]["language_embedding"] = None
    with pytest.raises(ValueError, match="language embeddings"):
        LanguageSimilaritySelector().select(meta, Budget("episodes", 1), config={"queries": ["open drawer"], **EMBED_CFG})

import numpy as np

from episode_selection.data import Episode
from episode_selection.quality import episode_quality_components, score_episode_quality


def test_quality_uses_explicit_and_proxy_components():
    ep = Episode("e", actions=np.array([[0.0], [0.5], [1.0]]), success=1.0, reward=0.8)
    score, comps = score_episode_quality(ep)
    assert comps["success"] == 1.0
    assert comps["validity"] == 1.0
    assert "smoothness" in comps
    assert 0.0 <= score <= 1.0

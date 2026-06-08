"""Pareto frontier utilities."""

from __future__ import annotations

from typing import Iterable, List, Mapping


def pareto_frontier(points: Iterable[Mapping], *, storage_key: str = "storage_bytes", metric_key: str, higher_is_better: bool = False) -> List[dict]:
    items = [dict(p) for p in points]
    frontier = []
    for i, point in enumerate(items):
        dominated = False
        s_i = float(point[storage_key])
        m_i = float(point[metric_key])
        for j, other in enumerate(items):
            if i == j:
                continue
            s_j = float(other[storage_key])
            m_j = float(other[metric_key])
            storage_better = s_j <= s_i
            metric_better = m_j >= m_i if higher_is_better else m_j <= m_i
            strictly = s_j < s_i or (m_j > m_i if higher_is_better else m_j < m_i)
            if storage_better and metric_better and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda p: (float(p[storage_key]), float(p[metric_key])))

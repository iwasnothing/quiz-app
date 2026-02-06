"""
Build normalized topic/concept graph from SQLite.

- DB path: config.settings.SQLITE_DB_PATH (see config.py).
- Source table: document_chunks with columns topic_name and sub_concepts
  (schema defined in database.py).
- Loads unique topics from document_chunks; two topics are grouped only if BOTH
  have >70% of their (non-stopword) words in common and at least 3 such words (symmetric
  overlap to avoid transitivity). Normalized name = top 10 common terms, or shortest
  topic for large clusters. Concepts are not normalized. Creates normalized_topics,
  normalized_concepts (identity), and topic_concept_edges.
"""

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter

import networkx as nx
from pyvis.network import Network

from config import settings
from database import query_all_topics, query_all_concepts, list_normalized_and_original_topics


def list_topic_concept_edges(db_path: str = None) -> list[dict]:
    """
    Query topic_concept_edges table and return list of edge dicts.
    Each dict contains: normalized_topic, normalized_concept, original_topic.
    Handles both old schema (without original_topic) and new schema (with original_topic).
    """
    if db_path is None:
        db_path = settings.SQLITE_DB_PATH
    
    path = Path(db_path)
    if not path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='topic_concept_edges'
    """)
    if not cursor.fetchone():
        conn.close()
        return []
    
    # Check if original_topic column exists
    cursor.execute("PRAGMA table_info(topic_concept_edges)")
    columns = {row[1] for row in cursor.fetchall()}
    has_original_topic = "original_topic" in columns
    
    if has_original_topic:
        cursor.execute("""
            SELECT normalized_topic, normalized_concept, original_topic
            FROM topic_concept_edges
            ORDER BY normalized_topic, normalized_concept
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "normalized_topic": row[0],
                "normalized_concept": row[1],
                "original_topic": row[2],
            }
            for row in rows
        ]
    else:
        # Old schema: use normalized_topic as original_topic fallback
        cursor.execute("""
            SELECT normalized_topic, normalized_concept
            FROM topic_concept_edges
            ORDER BY normalized_topic, normalized_concept
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "normalized_topic": row[0],
                "normalized_concept": row[1],
                "original_topic": row[0],  # Fallback to normalized_topic
            }
            for row in rows
        ]

# Source table and columns (must match database.py document_chunks schema)
DOCUMENT_CHUNKS_TABLE = "document_chunks"
TOPIC_NAME_COL = "topic_name"
SUB_CONCEPTS_COL = "sub_concepts"

# Group two topics only if (symmetric overlap to avoid transitivity chains):
# 1) BOTH topics have > OVERLAP_THRESHOLD of their words in the intersection, AND
# 2) at least MIN_SHARED_WORDS in common.
OVERLAP_THRESHOLD = 0.7   # 70% of each topic's words must be in common with the other
MIN_SHARED_WORDS = 3      # require at least 3 meaningful words in common

# Common words that don't indicate topic similarity (excluded from overlap and common terms).
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "of", "on", "or", "the", "to", "with", "its", "it", "their", "this", "that",
        "using", "use", "used", "different", "various", "other", "some", "such",
        "analysis", "assessment", "applications", "effects", "function", "functions",
        "system", "systems", "structure", "testing", "measurement", "identification",
        "behavior", "composition", "safety", "handling", "quality", "efficiency",
        "impact", "components", "materials", "process", "method", "study", "project",
        "activity", "activities", "results", "data",
    }
)


def _topic_words(topic: str) -> set[str]:
    """Return set of normalized meaningful words (lowercase, len > 1, not stopwords)."""
    words = re.findall(r"[a-zA-Z0-9]+", topic.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def _tokenize(topic: str) -> str:
    """Lowercase and join tokens with space (for common-terms display)."""
    return " ".join(sorted(_topic_words(topic)))


def _overlap_ratio(words_a: set[str], words_b: set[str]) -> float:
    """Fraction of the smaller set that is in the intersection. 0 if either set is empty."""
    if not words_a or not words_b:
        return 0.0
    inter = len(words_a & words_b)
    return inter / min(len(words_a), len(words_b))


def _symmetric_overlap_ok(words_a: set[str], words_b: set[str], threshold: float) -> bool:
    """True if both topics have > threshold of their words in the intersection (avoids transitivity chains)."""
    if not words_a or not words_b:
        return False
    inter = words_a & words_b
    if len(inter) < MIN_SHARED_WORDS:
        return False
    # Both the smaller and the larger topic must have > threshold of their words in common
    small = min(len(words_a), len(words_b))
    large = max(len(words_a), len(words_b))
    return len(inter) / small > threshold and len(inter) / large > threshold


def _union_find_parent(parents: list[int], i: int) -> int:
    while parents[i] != i:
        parents[i] = parents[parents[i]]
        i = parents[i]
    return i


def _cluster_topics_by_word_overlap(
    topics: list[str],
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> dict[str, str]:
    """
    Group topics only when BOTH have > overlap_threshold of their words in common
    (symmetric), and at least MIN_SHARED_WORDS. Reduces transitivity chains. Normalized
    name = top 10 common terms, or shortest topic name for large clusters. Returns original -> normalized.
    """
    if not topics:
        return {}
    if len(topics) == 1:
        return {topics[0]: topics[0]}

    word_sets = [_topic_words(t) for t in topics]
    n = len(topics)
    parents = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if _symmetric_overlap_ok(word_sets[i], word_sets[j], overlap_threshold):
                pi, pj = _union_find_parent(parents, i), _union_find_parent(parents, j)
                if pi != pj:
                    parents[pi] = pj

    # collect clusters by root index
    clusters: dict[int, list[str]] = {}
    for i, t in enumerate(topics):
        root = _union_find_parent(parents, i)
        clusters.setdefault(root, []).append(t)

    # Cap length of normalized name: avoid long meaningless concatenations from big clusters
    MAX_NORMALIZED_WORDS = 10   # at most this many words in common-terms name
    LARGE_CLUSTER_SIZE = 12    # if more members, use shortest topic name instead of common terms

    result: dict[str, str] = {}
    for members in clusters.values():
        if len(members) == 1:
            result[members[0]] = members[0]
            continue
        if len(members) > LARGE_CLUSTER_SIZE:
            normalized_name = min(members, key=len)
        else:
            word_counts: Counter[str] = Counter()
            for t in members:
                for w in _topic_words(t):
                    word_counts[w] += 1
            shared = [w for w, c in word_counts.most_common(MAX_NORMALIZED_WORDS) if c >= 2]
            normalized_name = " ".join(shared) if shared else min(members, key=len)
        for m in members:
            result[m] = normalized_name
    return result


def _create_normalized_tables(conn: sqlite3.Connection) -> None:
    """Create tables: normalized_topics, normalized_concepts, topic_concept_edges."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS normalized_topics (
            original_topic TEXT PRIMARY KEY,
            normalized_topic TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS normalized_concepts (
            original_concept TEXT PRIMARY KEY,
            normalized_concept TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topic_concept_edges (
            normalized_topic TEXT NOT NULL,
            normalized_concept TEXT NOT NULL,
            original_topic TEXT NOT NULL,
            PRIMARY KEY (normalized_topic, normalized_concept, original_topic)
        )
    """)
    conn.commit()


def _truncate_normalized_tables(conn: sqlite3.Connection) -> None:
    """Truncate normalized_topics, normalized_concepts, topic_concept_edges before rebuild."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM topic_concept_edges")
    cursor.execute("DELETE FROM normalized_concepts")
    cursor.execute("DELETE FROM normalized_topics")
    conn.commit()


def _build_and_fill_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """
    Load topics/concepts from document_chunks. Normalize topics by string cosine
    clustering (common terms). Concepts are not normalized (identity map).
    Returns (original_topic -> normalized_topic, original_concept -> normalized_concept).
    """
    db_path = settings.SQLITE_DB_PATH
    topics = query_all_topics(db_path)
    concepts = query_all_concepts(db_path)

    topic_map = (
        _cluster_topics_by_word_overlap(topics, OVERLAP_THRESHOLD)
        if topics
        else {}
    )
    concept_map = {c: c for c in concepts} if concepts else {}
    return topic_map, concept_map


def _fill_normalized_tables(
    conn: sqlite3.Connection,
    topic_map: dict[str, str],
    concept_map: dict[str, str],
) -> None:
    """Insert original -> normalized mappings into normalized_topics and normalized_concepts."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO normalized_topics (original_topic, normalized_topic) VALUES (?, ?)",
        [(k, v) for k, v in topic_map.items()],
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO normalized_concepts (original_concept, normalized_concept) VALUES (?, ?)",
        [(k, v) for k, v in concept_map.items()],
    )
    conn.commit()


def _build_topic_concept_edges(
    conn: sqlite3.Connection,
    topic_map: dict[str, str],
    concept_map: dict[str, str],
) -> None:
    """
    Build topic_concept_edges from document_chunks (topic_name, sub_concepts):
    map each to normalized_topic / normalized_concept and insert unique tuples
    including the original_topic for provenance tracking.
    """
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT {TOPIC_NAME_COL}, {SUB_CONCEPTS_COL} FROM {DOCUMENT_CHUNKS_TABLE}"
    )
    rows = cursor.fetchall()
    seen: set[tuple[str, str, str]] = set()
    for topic_name, sub_concepts_json in rows:
        norm_topic = topic_map.get(topic_name, topic_name)
        original_topic = topic_name  # Preserve the original topic name
        try:
            concepts = json.loads(sub_concepts_json)
            if not isinstance(concepts, list):
                continue
            for c in concepts:
                if not isinstance(c, str):
                    continue
                norm_concept = concept_map.get(c, c)
                key = (norm_topic, norm_concept, original_topic)
                if key not in seen:
                    seen.add(key)
                    cursor.execute(
                        "INSERT OR IGNORE INTO topic_concept_edges (normalized_topic, normalized_concept, original_topic) VALUES (?, ?, ?)",
                        (norm_topic, norm_concept, original_topic),
                    )
        except (json.JSONDecodeError, TypeError):
            continue
    conn.commit()


def build_graph() -> None:
    """
    Main entry: read SQLite at settings.SQLITE_DB_PATH, source table document_chunks
    (topic_name, sub_concepts per database.py). Truncates normalized tables, then
    normalizes topics by string cosine (common terms); concepts stay as-is. Creates
    normalized_topics, normalized_concepts, and topic_concept_edges.
    """
    db_path = settings.SQLITE_DB_PATH
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        _create_normalized_tables(conn)
        _truncate_normalized_tables(conn)
        topic_map, concept_map = _build_and_fill_mappings()
        _fill_normalized_tables(conn, topic_map, concept_map)
        _build_topic_concept_edges(conn, topic_map, concept_map)
        print(f"✓ Normalized {len(topic_map)} topics, {len(concept_map)} concepts; edges built.")
    finally:
        conn.close()


def visualize_graph(output_html: str = "topic_graph.html") -> str:
    """
    Visualize the normalized topic graph using NetworkX and PyVis.
    
    Structure:
    - Normalized topics (connect to their original topics)
    - Original topics (leaf nodes from topic normalization)
    
    Args:
        output_html: Path to the output HTML file (default: "topic_graph.html")
    
    Returns:
        Path to the generated HTML file.
    """
    # Get normalized and original topics from database
    topics_data = list_normalized_and_original_topics()
    
    if not topics_data:
        print("No topics found in the database.")
        return output_html
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Collect unique normalized topics and their original topics
    normalized_to_originals: dict[str, set[str]] = {}
    for item in topics_data:
        normalized = item["normalized_topic"]
        original = item["original_topic"]
        if normalized not in normalized_to_originals:
            normalized_to_originals[normalized] = set()
        normalized_to_originals[normalized].add(original)
    
    # Add normalized topic nodes
    for normalized_topic in normalized_to_originals.keys():
        G.add_node(normalized_topic, level=1, node_type="normalized")
    
    # Add original topic nodes and edges from normalized topics
    for normalized_topic, originals in normalized_to_originals.items():
        for original_topic in originals:
            # Only add original as separate node if it differs from normalized
            if original_topic != normalized_topic:
                G.add_node(original_topic, level=2, node_type="original")
                G.add_edge(normalized_topic, original_topic)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  - {len(normalized_to_originals)} normalized topics")
    print(f"  - {sum(len(v) for v in normalized_to_originals.values())} original topics")
    
    # Convert NetworkX graph to PyVis
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#222222",
        font_color="white",
        select_menu=True,
        filter_menu=True,
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 150,
                "springConstant": 0.01,
                "nodeDistance": 200,
                "damping": 0.09
            },
            "solver": "hierarchicalRepulsion"
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "levelSeparation": 200,
                "nodeSpacing": 150,
                "treeSpacing": 200,
                "direction": "UD",
                "sortMethod": "directed"
            }
        },
        "interaction": {
            "navigationButtons": true,
            "keyboard": true,
            "hover": true
        },
        "nodes": {
            "font": {
                "size": 14
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "cubicBezier"
            }
        }
    }
    """)
    
    # Add nodes with styling based on type
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get("node_type", "original")
        
        if node_type == "normalized":
            # Count how many originals map to this normalized topic
            num_originals = len(normalized_to_originals.get(node, set()))
            net.add_node(
                node,
                label=node[:50] + "..." if len(node) > 50 else node,
                color="#4ECDC4",
                size=30 + min(num_originals * 2, 20),
                shape="box",
                title=f"Normalized Topic: {node}\n({num_originals} original topics)",
                level=1,
            )
        else:  # original
            net.add_node(
                node,
                label=node[:40] + "..." if len(node) > 40 else node,
                color="#95E1D3",
                size=15,
                shape="dot",
                title=f"Original Topic: {node}",
                level=2,
            )
    
    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target, color="#4ECDC4", width=1)
    
    # Export to HTML
    net.write_html(output_html)
    print(f"✓ Graph visualization exported to: {output_html}")
    
    return output_html


if __name__ == "__main__":
    build_graph()
    for i, topic in enumerate(list_normalized_and_original_topics()):
        print(f"{i}. {topic['normalized_topic']} -> {topic['original_topic']}")
    
    # Generate visualization
    visualize_graph("topic_graph.html")

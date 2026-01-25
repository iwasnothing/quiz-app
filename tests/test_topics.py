#!/usr/bin/env python3
"""
Test script to connect to SQLite database and list all topics and concepts.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change to backend directory to ensure relative paths work
os.chdir(backend_dir)

from database import (
    query_all_topics,
    query_all_concepts,
    get_main_topics_from_sqlite,
    query_all_topics_with_counts,
)
import sqlite3
from config import settings


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def list_all_topics():
    """List all unique topics from the database."""
    print_separator()
    print("📚 ALL TOPICS IN DATABASE")
    print_separator()
    
    topics = query_all_topics()
    
    if not topics:
        print("❌ No topics found in the database.")
        print(f"   Database path: {settings.SQLITE_DB_PATH}")
        return
    
    print(f"Found {len(topics)} unique topics:\n")
    for i, topic in enumerate(topics, 1):
        print(f"  {i:3d}. {topic}")
    
    print(f"\nTotal: {len(topics)} topics")


def list_all_concepts():
    """List all unique concepts from the database."""
    print_separator()
    print("🔬 ALL CONCEPTS IN DATABASE")
    print_separator()
    
    concepts = query_all_concepts()
    
    if not concepts:
        print("❌ No concepts found in the database.")
        return
    
    print(f"Found {len(concepts)} unique concepts:\n")
    for i, concept in enumerate(concepts, 1):
        print(f"  {i:3d}. {concept}")
    
    print(f"\nTotal: {len(concepts)} concepts")


def list_topics_with_concepts():
    """List all topics with their associated concepts."""
    print_separator()
    print("📖 TOPICS WITH THEIR CONCEPTS")
    print_separator()
    
    topics_data = get_main_topics_from_sqlite()
    
    if not topics_data:
        print("❌ No topics found in the database.")
        return
    
    print(f"Found {len(topics_data)} topics with concepts:\n")
    
    for i, topic_info in enumerate(topics_data, 1):
        topic_name = topic_info["topic_name"]
        concepts = topic_info["sub_concepts"]
        
        print(f"  {i:3d}. {topic_name}")
        if concepts:
            for concept in concepts:
                print(f"       • {concept}")
        else:
            print("       (no concepts)")
        print()
    
    print(f"Total: {len(topics_data)} topics")


def list_topics_with_counts():
    """List all topics with their chunk counts and doc_type breakdown."""
    print_separator()
    print("📊 TOPICS WITH CHUNK COUNTS (INCLUDING DOC_TYPE)")
    print_separator()
    
    db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total counts per topic
    cursor.execute("""
        SELECT topic_name, COUNT(*) as count
        FROM document_chunks
        GROUP BY topic_name
        ORDER BY count DESC
    """)
    
    topics_with_counts = cursor.fetchall()
    
    if not topics_with_counts:
        print("❌ No topics found in the database.")
        conn.close()
        return
    
    # Get doc_type breakdown per topic
    cursor.execute("""
        SELECT topic_name, doc_type, COUNT(*) as count
        FROM document_chunks
        GROUP BY topic_name, doc_type
        ORDER BY topic_name, doc_type
    """)
    
    doc_type_breakdown = {}
    for topic_name, doc_type, count in cursor.fetchall():
        if topic_name not in doc_type_breakdown:
            doc_type_breakdown[topic_name] = {}
        doc_type_breakdown[topic_name][doc_type or 'NULL'] = count
    
    conn.close()
    
    print(f"Found {len(topics_with_counts)} topics:\n")
    print(f"{'Topic Name':<45} {'Total':>8} {'Doc Types':<30}")
    print("-" * 85)
    
    for topic_name, total_count in topics_with_counts:
        doc_types = doc_type_breakdown.get(topic_name, {})
        doc_type_str = ", ".join([f"{dt}({cnt})" for dt, cnt in sorted(doc_types.items())])
        if not doc_type_str:
            doc_type_str = "N/A"
        print(f"{topic_name:<45} {total_count:>8} {doc_type_str:<30}")
    
    total_chunks = sum(count for _, count in topics_with_counts)
    print("-" * 85)
    print(f"{'TOTAL':<45} {total_chunks:>8}")


def list_topics_with_doc_types():
    """List all topics with their doc_type breakdown."""
    print_separator()
    print("📋 TOPICS WITH DOC_TYPE BREAKDOWN")
    print_separator()
    
    db_path = settings.SQLITE_DB_PATH
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query topics with doc_type counts
    cursor.execute("""
        SELECT topic_name, doc_type, COUNT(*) as count
        FROM document_chunks
        GROUP BY topic_name, doc_type
        ORDER BY topic_name, doc_type
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("❌ No topics found in the database.")
        return
    
    # Group by topic_name
    topics_dict = {}
    for topic_name, doc_type, count in results:
        if topic_name not in topics_dict:
            topics_dict[topic_name] = {}
        topics_dict[topic_name][doc_type or 'NULL'] = count
    
    print(f"Found {len(topics_dict)} topics:\n")
    print(f"{'Topic Name':<50} {'Doc Type':<20} {'Count':>10}")
    print("-" * 82)
    
    for topic_name in sorted(topics_dict.keys()):
        doc_types = topics_dict[topic_name]
        first = True
        for doc_type, count in sorted(doc_types.items()):
            if first:
                print(f"{topic_name:<50} {doc_type:<20} {count:>10}")
                first = False
            else:
                print(f"{'':<50} {doc_type:<20} {count:>10}")
        print()
    
    # Summary
    total_by_doc_type = {}
    for topic_name, doc_types in topics_dict.items():
        for doc_type, count in doc_types.items():
            total_by_doc_type[doc_type] = total_by_doc_type.get(doc_type, 0) + count
    
    print("-" * 82)
    print(f"{'TOTAL BY DOC_TYPE':<50} {'':<20}")
    for doc_type in sorted(total_by_doc_type.keys()):
        print(f"{'':<50} {doc_type:<20} {total_by_doc_type[doc_type]:>10}")
    
    grand_total = sum(total_by_doc_type.values())
    print("-" * 82)
    print(f"{'GRAND TOTAL':<50} {'':<20} {grand_total:>10}")


def main():
    """Main function to run all tests."""
    print("\n" + "=" * 80)
    print("SQLite Database - Topics and Concepts Listing")
    print("=" * 80)
    print(f"\nDatabase path: {settings.SQLITE_DB_PATH}")
    print(f"Database exists: {os.path.exists(settings.SQLITE_DB_PATH)}")
    print()
    
    try:
        # List all topics
        list_all_topics()
        print()
        
        # List all concepts
        list_all_concepts()
        print()
        
        # List topics with their concepts
        list_topics_with_concepts()
        print()
        
        # List topics with counts
        list_topics_with_counts()
        print()
        
        # List topics with doc_type breakdown
        list_topics_with_doc_types()
        print()
        
        print_separator()
        print("✅ All queries completed successfully!")
        print_separator()
        
    except Exception as e:
        print_separator()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print_separator()
        sys.exit(1)


if __name__ == "__main__":
    main()

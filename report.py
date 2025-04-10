import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Any
import numpy as np

def generate_summary(comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report of the analyzed comments."""
    total_comments = len(comments)
    offensive_comments = [c for c in comments if c.get("is_offensive", False)]
    offensive_count = len(offensive_comments)
    offensive_percentage = (offensive_count / total_comments) * 100 if total_comments > 0 else 0
    offense_types = Counter([c.get("offense_type", "") for c in offensive_comments if c.get("offense_type")])
    sorted_comments = sorted(comments, key=lambda x: x.get("severity_score", 0), reverse=True)
    top_offensive = sorted_comments[:5]
    print("\n===== COMMENT ANALYSIS SUMMARY =====")
    print(f"Total comments analyzed: {total_comments}")
    print(f"Offensive comments detected: {offensive_count} ({offensive_percentage:.2f}%)")
    
    print("\nOffense Type Breakdown:")
    for offense_type, count in offense_types.items():
        percentage = (count / offensive_count) * 100 if offensive_count > 0 else 0
        print(f"- {offense_type}: {count} ({percentage:.2f}%)")
    
    print("\nTop 5 Most Offensive Comments (by severity):")
    for i, comment in enumerate(top_offensive):
        if comment.get("is_offensive", False):
            print(f"\n{i+1}. [Severity: {comment.get('severity_score', 0):.4f}] [Type: {comment.get('offense_type', '')}]")
            print(f"   Text: {comment.get('comment_text', '')[:100]}{'...' if len(comment.get('comment_text', '')) > 100 else ''}")
            print(f"   Explanation: {comment.get('explanation', 'No explanation provided')}")
    summary_data = {
        "total_comments": total_comments,
        "offensive_count": offensive_count,
        "offensive_percentage": offensive_percentage,
        "offense_types": dict(offense_types),
        "top_offensive": top_offensive
    }
    
    return summary_data

def generate_visual_report(comments: List[Dict[str, Any]], output_path: str) -> None:
    """Generate visual charts for the analyzed comments."""
    offensive_comments = [c for c in comments if c.get("is_offensive", False)]
    if not offensive_comments:
        print("No offensive comments to visualize.")
        return
    offense_types = Counter([c.get("offense_type", "unclassified") for c in offensive_comments])
    if "" in offense_types:
        offense_types["unclassified"] = offense_types.pop("")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    labels = list(offense_types.keys())
    sizes = list(offense_types.values())
    
    ax1.pie(sizes, labels=None, autopct='%1.1f%%', 
            startangle=90, shadow=True)
    ax1.set_title('Distribution of Offense Types')
    ax1.legend(labels, loc="best")
    x = np.arange(len(labels))
    bars = ax2.bar(x, sizes, width=0.6, color='crimson')
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_ylabel('Count')
    ax2.set_title('Frequency of Offense Types')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visual report saved to {output_path}")
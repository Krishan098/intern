import csv
import os
import torch
from typing import Dict, List, Any
from model import CommentAnalyzer
from report import generate_summary, generate_visual_report
from fpdf import FPDF  


INPUT_PATH = 'dataset.csv'
OUTPUT_PATH = 'dataset_analyzed.csv'
CHART_OUTPUT_PATH = 'dataset_report.png'
PDF_OUTPUT_PATH = 'dataset_report.pdf'  


GENERATE_CHART = True  


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def load_comments_csv(file_path: str) -> List[Dict[str, Any]]:
    comments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                comments.append(row)
        return comments
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

def save_comments_csv(comments: List[Dict[str, Any]], file_path: str) -> None:
    if not comments:
        print("No comments to save")
        return
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as csv_file:
            fieldnames = comments[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comments)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving to CSV file: {e}")

def display_comment_preview(comments: List[Dict[str, Any]], count: int = 5) -> None:
    print(f"\nPreview of {min(count, len(comments))} comments:")
    for i, comment in enumerate(comments[:count]):
        print(f"\n--- Comment {i+1} ---")
        print(f"ID: {comment.get('comment_id', 'N/A')}")
        print(f"User: {comment.get('username', 'N/A')}")
        text = comment.get('comment_text', 'N/A')
        preview = text[:100] + ('...' if len(text) > 100 else '')
        print(f"Text: {preview}")

def generate_pdf_report(summary: Dict[str, Any], chart_path: str = None) -> None:
    """
    Generate a PDF report with the summary and optional chart
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Comment Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary Statistics', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                pdf.cell(0, 8, f"{key}: {value}", 0, 1)
            elif isinstance(value, dict):
                pdf.cell(0, 8, f"{key}:", 0, 1)
                for sub_key, sub_value in value.items():
                    pdf.cell(10)  
                    pdf.cell(0, 8, f"{sub_key}: {sub_value}", 0, 1)
            else:
                pdf.cell(0, 8, f"{key}: {str(value)}", 0, 1)
        
        
        if chart_path and os.path.exists(chart_path):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Visual Analysis', 0, 1, 'L')
            img_width = 180  
            pdf.image(chart_path, x=10, y=None, w=img_width)
        

        pdf.output(PDF_OUTPUT_PATH)
        print(f"PDF report generated at {PDF_OUTPUT_PATH}")
    
    except Exception as e:
        print(f"Error generating PDF report: {e}")

def main():
    comments = load_comments_csv(INPUT_PATH)
    if not comments:
        print("No comments loaded. Exiting.")
        return
   
    print(f"\nLoaded {len(comments)} comments from {INPUT_PATH}")
   

    comments_to_analyze = comments[::10]
    display_comment_preview(comments_to_analyze, 5)
   
    analyzer = CommentAnalyzer(device=DEVICE)
   
    analyzed_comments = analyzer.analyze_comments(comments_to_analyze)
   
    save_comments_csv(analyzed_comments, OUTPUT_PATH)
   

    summary = generate_summary(analyzed_comments)
    

    if GENERATE_CHART:
        generate_visual_report(analyzed_comments, CHART_OUTPUT_PATH)
    

    generate_pdf_report(summary, CHART_OUTPUT_PATH if GENERATE_CHART else None)

if __name__ == "__main__":
    main()
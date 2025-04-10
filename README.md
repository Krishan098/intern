# Comment Analysis Project

A tool for analyzing comments for offensive content using deep learning and generating detailed reports with visualizations.

## Overview

This project provides a comprehensive solution for analyzing user comments to detect offensive content. It uses a RoBERTa-based model from Hugging Face to classify comments, categorize offense types, and generate detailed reports with visualizations.

## Features

- Analyzes comments for offensive content using a pre-trained RoBERTa model
- Categorizes offensive content into types (hate speech, profanity, harassment, threats, toxicity)
- Generates detailed summary reports with statistics and examples
- Creates visual charts showing distribution of offense types
- Exports results to CSV for further analysis
- Produces PDF reports with statistics and visualizations

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Matplotlib
- FPDF
- better_profanity
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/comment-analysis.git
   cd comment-analysis
   ```

2. Install required packages:
   ```
   pip install torch transformers matplotlib fpdf better_profanity numpy
   ```

3. Download the required model:
   ```
   # The model will be automatically downloaded when you first run the script
   ```

## Setup

1. Ensure you have a CSV file with comments to analyze. The CSV should have a column named `comment_text` containing the text to analyze.

2. Configure the paths in `main.py`:
   ```python
   INPUT_PATH = 'dataset.csv'              # Your input CSV file
   OUTPUT_PATH = 'dataset_analyzed.csv'    # Where to save analysis results
   CHART_OUTPUT_PATH = 'dataset_report.png' # Chart image path
   PDF_OUTPUT_PATH = 'dataset_report.pdf'   # PDF report path
   ```

3. Set the `GENERATE_CHART` flag in `main.py` to control whether charts are generated.

## Usage

Run the main script:

```
python main.py
```

The script will:
1. Load comments from the input CSV file
2. Analyze the comments using the RoBERTa model
3. Save analyzed comments to the output CSV file
4. Generate a summary report in the console
5. Create visual charts (if enabled)
6. Generate a PDF report with statistics and charts

## Sample Output

### Console Output

```
Using device: cuda
Loaded 1000 comments from dataset.csv

Preview of 5 comments:

--- Comment 1 ---
ID: 12345
User: user123
Text: This is a sample comment that will be analyzed by the system.

--- Comment 2 ---
ID: 12346
User: user456
Text: I completely disagree with your opinion, you're so wrong about this topic.

Loading model cardiffnlp/twitter-roberta-base-offensive on cuda...
Analyzing comment 1/100...
Analyzing comment 101/200...
...

===== COMMENT ANALYSIS SUMMARY =====
Total comments analyzed: 100
Offensive comments detected: 27 (27.00%)

Offense Type Breakdown:
- toxicity: 12 (44.44%)
- harassment: 8 (29.63%)
- hate_speech: 4 (14.81%)
- profanity: 2 (7.41%)
- threat: 1 (3.70%)

Top 5 Most Offensive Comments (by severity):
1. [Severity: 0.9872] [Type: hate_speech]
   Text: You are so stupid! Everyone from your country is worthless and should be banned from this platform...
   Explanation: Comment contains language associated with hate speech or discrimination. (Confidence: 0.99)

2. [Severity: 0.9653] [Type: profanity]
   Text: This is complete **** and you know it. Stop spreading these lies you *****...
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.97)

Visual report saved to dataset_report.png
Results saved to dataset_analyzed.csv
PDF report generated at dataset_report.pdf
```

### Visual Report
The script generates charts showing the distribution and frequency of different offense types among the detected offensive comments.

### PDF Report
The PDF report includes summary statistics and visualizations for easy sharing and archiving of results.

## Project Structure

- `main.py`: Main script to run the analysis
- `model.py`: Contains the CommentAnalyzer class that performs the offensive content detection
- `report.py`: Functions for generating summary reports and visualizations
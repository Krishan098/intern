# Comment Analysis Project

A tool for analyzing comments for offensive content using deep learning and generating detailed reports with visualizations.

## Overview

This project provides a comprehensive solution for analyzing user comments to detect offensive content. It uses a RoBERTa-based model from Hugging Face to classify comments, categorize offense types, and generate detailed reports with visualizations. The dataset consisted of 2,23,550 entries. This project checks only 22355 out of these entries throught the entire dataset due to hardware constraints. I checked every 10th comment to check for.

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
   pip install -r requirements.txt
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

Loaded 223549 comments from dataset.csv

Preview of 5 comments:

--- Comment 1 ---
ID: 0000997932d777bf
User: reevesmaxwell
Text: Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't ...

--- Comment 2 ---
ID: 0005300084f90edc
User: lmccoy
Text: "
Fair use rationale for Image:Wonju.jpg

Thanks for uploading Image:Wonju.jpg. I notice the image p...

--- Comment 3 ---
ID: 000b08c464718505
User: millersarah
Text: "

 Regarding your recent edits 

Once again, please read WP:FILMPLOT before editing any more film a...

--- Comment 4 ---
ID: 0011cc71398479c4
User: kimberly27
Text: How could I post before the block expires?  The funny thing is, you think I'm being uncivil!

--- Comment 5 ---
ID: 001735f961a23fc4
User: johnelliott
Text: "
 Sure, but the lead must briefly summarize Armenia's history. I simply added what I found necessar...
Loading model cardiffnlp/twitter-roberta-base-offensive on cuda...

===== COMMENT ANALYSIS SUMMARY =====
Total comments analyzed: 22355
Offensive comments detected: 3028 (13.55%)

Offense Type Breakdown:
- toxicity: 608 (20.08%)
- threat: 73 (2.41%)
- profanity: 2056 (67.90%)
- hate_speech: 140 (4.62%)
- harassment: 151 (4.99%)

Top 5 Most Offensive Comments (by severity):

1. [Severity: 0.9520] [Type: profanity]
   Text: Fuck you. I can do whatever the fuck I want, you piece of shit. Personally, I think your a stuck up ...
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.95)

2. [Severity: 0.9510] [Type: profanity]
   Text: bitch 

you are such a whiny ass attention whore bitch, go choke on a cock
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.95)

3. [Severity: 0.9510] [Type: profanity]
   Text: Fuck you, like I give a shit. Point to the word where I cast suspicion on her. Point to it. Point to...
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.95)

4. [Severity: 0.9510] [Type: profanity]
   Text: Go fuckin' hang yourself! Fuckin' scum of the earth! Fuck you asshole!
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.95)

5. [Severity: 0.9510] [Type: profanity]
   Text: Who the fuck do you think you are? How dare you fucking block me! You have NO IDEA how your messing ...
   Explanation: Comment contains profane language detected by profanity filter. (Confidence: 0.95)
Visual report saved to dataset_report.png
PDF report generated at dataset_report.pdf
```

### Visual Report
The script generates charts showing the distribution and frequency of different offense types among the detected offensive comments.
![dataset_report](https://github.com/user-attachments/assets/d0de174f-609a-4619-b10d-fb036c2ae4bc)


### PDF Report
The PDF report includes summary statistics and visualizations for easy sharing and archiving of results.

## Project Structure

- `main.py`: Main script to run the analysis
- `model.py`: Contains the CommentAnalyzer class that performs the offensive content detection
- `report.py`: Functions for generating summary reports and visualizations

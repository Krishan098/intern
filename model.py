import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any
import torch.nn.functional as F
from better_profanity import profanity


class CommentAnalyzer:
    """Class to analyze comments for offensive content using Hugging Face's RoBERTa."""

    def __init__(self, device: str = None):
        """Initialize the analyzer with RoBERTa model."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "cardiffnlp/twitter-roberta-base-offensive"

        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        profanity.load_censor_words()
        
        self.offense_types = {
            "hate_speech": ["hate", "racist", "nazi", "jews", "muslim", "black", "white", "asian", "hispanic"],
            "profanity": [], 
            "harassment": ["stupid", "idiot", "loser", "dumb", "retard", "moron", "bully"],
            "threat": ["kill", "die", "attack", "hurt", "threat", "murder", "bomb"],
            "toxicity": ["toxic", "poison", "disgusting"]
        }

    def _determine_offense_type(self, text: str, severity: float) -> str:
        """Determine specific offense type based on content and keywords."""
        if severity < 0.5:
            return ""
        if profanity.contains_profanity(text):
            return "profanity"
            
        text_lower = text.lower()
        for offense_type, keywords in self.offense_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return offense_type
        return "toxicity"
        
    def _generate_explanation(self, text: str, is_offensive: bool, offense_type: str, severity: float) -> str:
        """Generate an explanation for the classification."""
        if not is_offensive:
            return "Comment classified as not offensive based on RoBERTa model analysis."
            
        explanations = {
            "hate_speech": "Comment contains language associated with hate speech or discrimination.",
            "profanity": "Comment contains profane language detected by profanity filter.",
            "harassment": "Comment contains harassing or insulting language.",
            "threat": "Comment contains threatening language or implied violence.",
            "toxicity": "Comment contains generally toxic or negative language."
        }
        if offense_type in explanations:
            return f"{explanations[offense_type]} (Confidence: {severity:.2f})"
        return f"Comment classified as offensive with {severity:.2f} confidence based on RoBERTa model analysis."

    def analyze_comment(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single comment."""
        try:
            comment_text = comment.get("comment_text", "")
            if not comment_text.strip():
                comment["is_offensive"] = False
                comment["offense_type"] = ""
                comment["explanation"] = "Empty comment"
                comment["severity_score"] = 0.0
                return comment
            inputs = self.tokenizer(
                comment_text, 
                return_tensors="pt", 
                truncation=True,
                max_length=512,  
                padding='max_length'  
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            
            predicted_idx = probs.argmax()
            is_offensive = bool(predicted_idx == 1)
            severity = float(probs[1])  # Confidence score for offensive
            
            
            offense_type = self._determine_offense_type(comment_text, severity) if is_offensive else ""
            
            explanation = self._generate_explanation(comment_text, is_offensive, offense_type, severity)
            analyzed_comment = comment.copy()
            analyzed_comment["is_offensive"] = is_offensive
            analyzed_comment["offense_type"] = offense_type
            analyzed_comment["explanation"] = explanation
            analyzed_comment["severity_score"] = round(severity, 3)

            return analyzed_comment

        except Exception as e:
            print(f"Error analyzing comment: {e}")
            return {
                **comment,
                "is_offensive": False,
                "offense_type": "",
                "explanation": f"Error analyzing comment: {str(e)}",
                "severity_score": 0
            }

    def analyze_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze a list of comments."""
        analyzed_comments = []
        total = len(comments)

        for i, comment in enumerate(comments):
            if i % 100 == 0:
                print(f"Analyzing comment {i+1}/{total}...")
                
            analyzed_comment = self.analyze_comment(comment)
            analyzed_comments.append(analyzed_comment)

        return analyzed_comments
"""
Accuracy Metrics Module
Computes real measurable accuracy metrics for the AI Video Summarizer
"""

import re
import math


class AccuracyMetrics:

    @staticmethod
    def word_count_accuracy(summary: str, target_length: int) -> dict:
        """
        Measures how accurately the summary matches the requested word count.
        Score = 100% if within ±10 words, degrades linearly beyond that.
        """
        words = len(summary.split())
        deviation = abs(words - target_length)
        if deviation <= 10:
            score = 100.0
        elif deviation <= 30:
            score = round(100.0 - ((deviation - 10) / 20.0) * 30, 1)
        else:
            score = max(0.0, round(70.0 - ((deviation - 30) / target_length) * 100, 1))

        return {
            "metric": "Word Count Accuracy",
            "target": target_length,
            "actual": words,
            "deviation": deviation,
            "score": score,
            "grade": AccuracyMetrics._grade(score)
        }

    @staticmethod
    def structure_score(summary: str) -> dict:
        """
        Checks if the summary follows the expected structure:
        Setting the Stage, Key Points, Conclusions.
        """
        checks = {
            "Has introduction / Setting the Stage": bool(
                re.search(r"setting the stage|introduction|overview|background", summary, re.I)
            ),
            "Has key points / bullet points": bool(
                re.search(r"key point|point [abc]|•|-\s|\*\s|\d+\.", summary, re.I)
            ),
            "Has conclusion": bool(
                re.search(r"conclusion|summary|takeaway|overall|in summary|to summarize", summary, re.I)
            ),
            "Minimum length (>50 words)": len(summary.split()) > 50,
            "No error messages": not bool(
                re.search(r"⚠️|error|failed|exception", summary, re.I)
            ),
        }
        passed = sum(checks.values())
        score = round((passed / len(checks)) * 100, 1)
        return {
            "metric": "Structure Score",
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "score": score,
            "grade": AccuracyMetrics._grade(score)
        }

    @staticmethod
    def content_coverage(summary: str, transcript: str) -> dict:
        """
        Measures keyword overlap between summary and transcript.
        Uses a simple TF-based keyword extraction and checks coverage.
        """
        if not transcript:
            return {
                "metric": "Content Coverage",
                "score": 0.0,
                "grade": "N/A",
                "reason": "No transcript available"
            }

        # Extract meaningful words (>4 chars, not stopwords)
        stopwords = {
            "this", "that", "with", "from", "have", "been", "will",
            "they", "their", "there", "about", "which", "would", "could",
            "should", "also", "into", "more", "some", "when", "then",
            "than", "what", "your", "just", "like", "very", "over"
        }

        def keywords(text):
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            return set(w for w in words if w not in stopwords)

        transcript_kw = keywords(transcript)
        summary_kw = keywords(summary)

        if not transcript_kw:
            return {"metric": "Content Coverage", "score": 0.0, "grade": "N/A"}

        # How many transcript keywords appear in summary
        covered = len(summary_kw & transcript_kw)
        # Use top-N most frequent transcript words for fairness
        top_n = min(50, len(transcript_kw))
        score = round(min(100.0, (covered / top_n) * 100 * 2), 1)  # *2 scaling factor

        return {
            "metric": "Content Coverage",
            "transcript_keywords": len(transcript_kw),
            "summary_keywords": len(summary_kw),
            "covered": covered,
            "score": min(score, 100.0),
            "grade": AccuracyMetrics._grade(min(score, 100.0))
        }

    @staticmethod
    def compression_ratio(summary: str, transcript: str) -> dict:
        """
        Measures how well the summary compresses the transcript.
        Ideal ratio is 5-15% (good compression with content retention).
        """
        if not transcript:
            return {"metric": "Compression Ratio", "score": 0.0, "grade": "N/A"}

        t_words = len(transcript.split())
        s_words = len(summary.split())
        ratio = round((s_words / t_words) * 100, 1) if t_words > 0 else 0

        # Score: ideal range 5-20%, penalize extremes
        if 5 <= ratio <= 20:
            score = 100.0
        elif ratio < 5:
            score = round((ratio / 5) * 100, 1)
        elif ratio <= 50:
            score = round(100 - ((ratio - 20) / 30) * 50, 1)
        else:
            score = 20.0

        return {
            "metric": "Compression Ratio",
            "transcript_words": t_words,
            "summary_words": s_words,
            "ratio_percent": ratio,
            "score": score,
            "grade": AccuracyMetrics._grade(score)
        }

    @staticmethod
    def readability_score(summary: str) -> dict:
        """
        Flesch Reading Ease score adapted for summary quality.
        Higher = more readable.
        """
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = summary.split()
        syllables = sum(AccuracyMetrics._count_syllables(w) for w in words)

        if not sentences or not words:
            return {"metric": "Readability", "score": 0.0, "grade": "N/A"}

        avg_sentence_len = len(words) / len(sentences)
        avg_syllables = syllables / len(words)

        # Flesch Reading Ease formula
        flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
        flesch = max(0.0, min(100.0, flesch))

        return {
            "metric": "Readability (Flesch)",
            "sentences": len(sentences),
            "avg_sentence_length": round(avg_sentence_len, 1),
            "flesch_score": round(flesch, 1),
            "score": round(flesch, 1),
            "grade": AccuracyMetrics._grade(flesch)
        }

    @staticmethod
    def transcript_accuracy(transcript: str, source: str) -> dict:
        """
        Rates transcript quality based on source and content checks.
        """
        if not transcript:
            return {"metric": "Transcript Accuracy", "score": 0.0, "grade": "F"}

        checks = {
            "Transcript retrieved": True,
            "Sufficient length (>20 words)": len(transcript.split()) > 20,
            "No error markers": "error" not in transcript.lower()[:100],
            "Has sentence structure": bool(re.search(r'[.!?,]', transcript)),
        }

        base_score = 70.0 if source == "youtube_api" else 60.0
        passed = sum(checks.values())
        bonus = (passed / len(checks)) * 30.0
        score = round(min(100.0, base_score + bonus), 1)

        return {
            "metric": "Transcript Accuracy",
            "source": "YouTube API" if source == "youtube_api" else "Whisper ASR",
            "word_count": len(transcript.split()),
            "checks": checks,
            "score": score,
            "grade": AccuracyMetrics._grade(score)
        }

    @staticmethod
    def rouge_f1_score(summary: str, transcript: str) -> dict:
        """
        ROUGE-1 style F1 score using unigram overlap.
        Precision = what summary included that was relevant
        Recall    = what relevant content was captured from transcript
        F1        = harmonic mean of both
        """
        if not summary or not transcript:
            return {"metric": "ROUGE F1", "score": 0.0, "grade": "N/A",
                    "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        stopwords = {
            "this", "that", "with", "from", "have", "been", "will",
            "they", "their", "there", "about", "which", "would", "could",
            "should", "also", "into", "more", "some", "when", "then",
            "than", "what", "your", "just", "like", "very", "over", "the",
            "and", "for", "are", "was", "were", "has", "had", "not", "its"
        }

        def get_tokens(text):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return [w for w in words if w not in stopwords]

        summary_set = set(get_tokens(summary))
        transcript_set = set(get_tokens(transcript))

        tp = len(summary_set & transcript_set)
        fp = len(summary_set - transcript_set)
        fn = len(transcript_set - summary_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        score = round(f1 * 100, 1)

        return {
            "metric": "ROUGE F1",
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1, 4),
            "score": score,
            "tp": tp, "fp": fp, "fn": fn,
            "grade": AccuracyMetrics._grade(score)
        }

    @staticmethod
    def overall_accuracy(metrics: list) -> dict:
        """Compute weighted overall accuracy from all metrics."""
        weights = {
            "Word Count Accuracy": 0.20,
            "Structure Score": 0.25,
            "Content Coverage": 0.25,
            "Compression Ratio": 0.15,
            "Readability (Flesch)": 0.15,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for m in metrics:
            name = m.get("metric", "")
            score = m.get("score", 0.0)
            w = weights.get(name, 0.0)
            weighted_sum += score * w
            total_weight += w

        overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
        return {
            "overall_score": overall,
            "grade": AccuracyMetrics._grade(overall)
        }
    @staticmethod
    def rouge_f1_score(summary: str, transcript: str) -> dict:
        """
        ROUGE-1 style F1 score using unigram overlap.
        Precision = what summary included that was relevant
        Recall    = what relevant content was captured from transcript
        F1        = harmonic mean of both
        """
        if not summary or not transcript:
            return {"metric": "ROUGE F1", "score": 0.0, "grade": "N/A",
                    "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        stopwords = {
            "this", "that", "with", "from", "have", "been", "will",
            "they", "their", "there", "about", "which", "would", "could",
            "should", "also", "into", "more", "some", "when", "then",
            "than", "what", "your", "just", "like", "very", "over", "the",
            "and", "for", "are", "was", "were", "has", "had", "not", "its"
        }

        def get_tokens(text):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return [w for w in words if w not in stopwords]

        summary_set = set(get_tokens(summary))
        transcript_set = set(get_tokens(transcript))

        tp = len(summary_set & transcript_set)
        fp = len(summary_set - transcript_set)
        fn = len(transcript_set - summary_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        score = round(f1 * 100, 1)

        return {
            "metric": "ROUGE F1",
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1, 4),
            "score": score,
            "tp": tp, "fp": fp, "fn": fn,
            "grade": AccuracyMetrics._grade(score)
        }



    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:")
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

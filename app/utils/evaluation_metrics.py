import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk


class EvaluationMetrics:
     """
    A utility class for evaluating transcription and translation performance.
    Provides methods to compute Word Error Rate (WER) and BLEU score.
    """
     
     @staticmethod
     def calculate_wer(reference: str, hypothesis: str) -> float:
          """
          Calculate Word Error Rate (WER) between a reference and hypothesis string.
          """
          reference_words = reference.split()
          hypothesis_words = hypothesis.split()
          n = len(reference_words)

          dp = np.zeros((len(reference_words) + 1, len(hypothesis_words) + 1), dtype=int)
          for i in range(len(reference_words) + 1):
               dp[i][0] = i
          for j in range(len(hypothesis_words) + 1):
               dp[0][j] = j

          for i in range(1, len(reference_words) + 1):
               for j in range(1, len(hypothesis_words) + 1):
                    if reference_words[i - 1] == hypothesis_words[j - 1]:
                         dp[i][j] = dp[i - 1][j - 1]
                    else:
                         dp[i][j] = 1 + min(dp[i - 1][j],        # Del
                                            dp[i][j - 1],        # Ins
                                            dp[i - 1][j - 1])    # Sub
                         
          wer = dp[len(reference_words)][len(hypothesis_words)] / n
          return round(wer * 100, 2)
     

     @staticmethod
     def calculate_bleu(reference: str, hypothesis: str) -> float:
          """
        Calculate BLEU score between a reference and hypothesis string.
        Uses NLTK's sentence_bleu with smoothing for short sentences.
        """
          ref_tokens = [reference.split()]
          hyp_tokens = hypothesis.split()
          smoothie = SmoothingFunction().method4
          bleu_score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
          return round(bleu_score, 2)
     

     @staticmethod
     def calculate_meteor(reference: str, hypothesis: str) -> float:
          """
          Calculate METEOR score between reference and hypothesis.
          """
          ref_tokens = [reference.split()]
          hyp_tokens = hypothesis.split()
          score = meteor_score(ref_tokens, hyp_tokens)
          return round(score, 2)




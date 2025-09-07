import requests
from evaluation_metrics import EvaluationMetrics


# Example: Test data (ground truth transcription and translation)
test_cases = [
    # {
    #     "audio_file": "../../audio_samples/audio2_native.mp4",
    #     "target_language": "English",
    #     "ground_truth_transcription": "Technology is helping people around the world talk to each other, even when they speak different languages.",
    #     "ground_truth_translation": "Technologie helpt mensen over de hele wereld met elkaar te praten, zelfs als ze verschillende talen spreken."
    # }
]

# api_url = "http://localhost:8000/v1/transcribe-and-translate"
api_url = ""

for case in test_cases:
    with open(case["audio_file"], "rb") as audio:
        response = requests.post(api_url, files={"file": audio}, data={"target_language": case["target_language"]})
        result = response.json()
        print(result)

        # Extract system outputs
        system_transcription = result["data"]["transcription"]
        system_translation = result["data"]["translation"]

        # Calculate BLEU
        bleu = EvaluationMetrics.calculate_bleu(case["ground_truth_translation"], system_translation)

        # Calculate METEOR
        meteor_score = EvaluationMetrics.calculate_meteor(case["ground_truth_translation"], system_translation)

        print(f"File: {case['audio_file']}")
        print(f"METEOR: {meteor_score}")
        print(f"BLEU: {bleu}\n")

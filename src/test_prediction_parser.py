
from llm_verifier import extract_predicted_label

test_cases = [
    ("**DATASET_LABEL:** FALSE\n**VERDICT:** REFUTED", "FALSE"),
    ("DATASET_LABEL: TRUE\nVERDICT: SUPPORTED", "TRUE"),
    ("**VERDICT:** REFUTED", "FALSE"),
    ("VERDICT: SUPPORTED", "TRUE"),
    ("DATASET_LABEL: MANIPULATED", "MANIPULATED"),
    ("DATASET_LABEL: MOSTLY_FALSE", "MOSTLY_FALSE"),
    ("VERDICT: INSUFFICIENT EVIDENCE", "INSUFFICIENT_EVIDENCE"),
]

print("\n========== PREDICTION PARSER TEST ==========\n")

passed = 0

for response, expected in test_cases:

    actual = extract_predicted_label(response)

    status = "PASS" if actual == expected else "FAIL"

    print(
        f"{status} | Expected: {expected} | Actual: {actual}"
    )

    if actual == expected:
        passed += 1

print(f"\nPassed: {passed}/{len(test_cases)}")

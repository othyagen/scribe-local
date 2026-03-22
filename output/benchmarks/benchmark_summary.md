# Benchmark Summary

**Timestamp:** 2026-03-15T19:36:41.620993

## Per-Case Results

| Case | Symptoms | Negations | Medications | Durations | Qualifiers | Patterns | Red Flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chest_pain_consultation | 100% | 67% | 100% | 0% | 0% | 0% | 100% |
| cough_fever_telephone | 100% | 100% | 100% | 0% | 0% | 100% | 100% |
| abdominal_pain_consultation | 33% | 67% | 100% | 0% | 0% | 100% | 100% |

## Aggregate Scores

- **symptom_extraction_recall:** 0.7778
- **negation_accuracy_recall:** 0.7778
- **medication_accuracy_recall:** 1.0000
- **duration_accuracy_recall:** 0.0000
- **qualifier_accuracy_overall_accuracy:** 0.0000
- **pattern_detection_recall:** 0.6667
- **red_flag_detection_recall:** 1.0000
- **overall_score:** 0.6032

## Diagnosis Details

### chest_pain_consultation

**symptom_extraction:**
  - False positives: pain

**negation_accuracy:**
  - False negatives: heart disease
  - False positives: No I do not have any heart problems

**duration_accuracy:**
  - False negatives: three days

**pattern_detection:**
  - False negatives: angina_like

### cough_fever_telephone

**duration_accuracy:**
  - False negatives: five days

### abdominal_pain_consultation

**symptom_extraction:**
  - False negatives: abdominal pain, bloating
  - False positives: diarrhea, vomiting, pain

**negation_accuracy:**
  - False negatives: blood in stool
  - False positives: No blood in the stool either

**duration_accuracy:**
  - False negatives: two weeks

**pattern_detection:**
  - False positives: gastroenteritis_like

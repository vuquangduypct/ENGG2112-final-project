# Data engineering decisions and report checklist

## Scope

The user's revised target is five-grade APTOS classification, retaining diagnosis 0,1,2,3,4. Input is a fundus photograph; output is a predicted dataset severity grade. This is a student research prototype. Clinical utility or improvement in patient outcomes has not been established by this work.

Official source: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

## Explain the research decisions

Use disease-specific motivation, explain each method in your own words, and describe unavailable metadata honestly.

For the final report, the student should explain in their own words:

1. Why retain five severity levels, and what extra difficulty does this introduce compared with the binary proposal?
2. Where the images and labels came from, and what variation exists in cameras/collection conditions according to the dataset documentation.
3. Actual original and eligible sample counts, per-grade counts, dimensions, exclusions and duplicate findings from the local audit.
4. Why the chosen resolution and preprocessing are appropriate. Compare examples; a resized image looking tidy does not demonstrate preservation of small lesions.
5. Why a group-aware split is needed. Report actual split counts and its limitations, rather than claiming exact 70/15/15 or patient-level separation without evidence.
6. What changed from the proposal and original team allocation.

## Implemented baseline decisions

| Decision | Implementation | Reason / limitation |
|---|---|---|
| Labels | Retain original grades 0..4 | Matches revised target; no binary remapping |
| Originals | Read only, retained | Supports reproducibility and review |
| Integrity | PNG verification and full pixel decoding | Corrupt/truncated files should not silently enter training |
| Exact duplicates | SHA-256 of RGB pixels and dimensions | Captures identical decoded pixels even with different PNG encodings |
| Conflicting labels | Quarantine identical-pixel groups with multiple grades | Avoids inventing a corrected grade; expert/manual resolution remains outstanding |
| Similar images | dHash candidate list | Review heuristic; can miss related images and generate false matches |
| Splits | Group-aware, fixed seed 2112, approximate 70/15/15 | Provisional until related-image review and any available patient metadata are checked |
| Resize baseline | Aspect-preserving 224x224 RGB, black padding | Low-cost baseline; not established as optimal for fine retinal detail |
| Black crop | Optional, OFF by default | Review risk of removing dark but relevant peripheral content |
| Normalization | Deferred to chosen model | Avoids double normalization or guessing pretrained model requirements |
| Augmentation | Deferred to training pipeline | Validation and test remain deterministic |

## Remaining research and team decisions

- Find and read primary references on DR screening need, fundus imaging and existing image-based methods. Record exact source and claim; do not use unsupported prevalence or elderly-benefit claims.
- Confirm whether any authorised supplemental metadata contains age or patient IDs. If not, report that age-subgroup evaluation and patient-level split verification are not possible with the available CSV.
- Review `artifacts/audit/near_duplicate_candidates.csv`, and the exact duplicate / conflicting-label files. Use local image inspection and record decisions; visual similarity alone does not establish patient identity.
- Agree on resolution and normalization with the model teammate. Select experiments with validation data, then freeze the test split.
- Agree on five-class metrics and how ordinal severity errors will be discussed. Do not rely solely on overall accuracy.
- Preserve citations and AI prompts/outputs plus student modifications. The generated code requires student understanding and review.

## Course links

- Week 3 (ML pipeline): https://canvas.sydney.edu.au/courses/75715/modules/items/3182411
- Week 4 (preprocessing): https://canvas.sydney.edu.au/courses/75715/modules/items/3182412
- Week 5 (CNNs): https://canvas.sydney.edu.au/courses/75715/modules/items/3182413
- Week 9 worksheet (data strategy, related observations, evaluation): https://canvas.sydney.edu.au/courses/75715/files/52894771?wrap=1
- Proposal and feedback: https://canvas.sydney.edu.au/courses/75715/assignments/704307

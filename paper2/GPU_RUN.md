# Paper 2 GPU execution contract
Frozen dataset: UTD-MHAD, official paired RGB + 3D skeleton, 27 actions, 861 paired sequences.
Frozen outer split: subjects 1,3,5,7 = development/train pool (431); subjects 2,4,6,8 = final held-out test (430).
Model selection: leave-one-training-subject-out only. Final retrain uses all four training subjects. Final test is evaluated once.
Seeds: 42, 123, 2026.
Primary scientific claim: synchronized multi-level explanation-to-decision faithfulness, not raw accuracy novelty.
Required recognition outputs: top-1, macro-F1, per-class recall, confusion matrix.
Required XAI outputs: temporal, RGB-region, skeleton-joint and modality evidence; deletion AUC/confidence drop; matched random; bottom-vs-top; stability; insertion where meaningful.
Required statistics: paired bootstrap confidence intervals and effect sizes.
Required baselines: RGB-only, skeleton-only, ordinary fusion, full model.
Required ablations: no motion-conditioned interaction, no temporal evidence selection, modality-specific interventions, and any explanation-consistency training component if introduced.
No test-driven tuning. Fit normalization/calibration on development data only. Archive exact commit, environment, configs, seeds, checkpoints, predictions, metrics, figures and logs before deleting paid compute.
The next paid GPU run may start only after Paper2 Real Data Smoke and Paper2 GPU Ready Gate both pass.

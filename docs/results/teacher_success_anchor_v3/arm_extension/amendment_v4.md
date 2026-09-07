# Prospective v4 binding for the arm-start extension

The [amendment](amendment_v4.json) rebinds the already frozen arm-start extension
to the eventual `clear_simulator_winner` of the prospective v4 confirmation.
The legacy v3 model screen was not executed. The original extension protocol,
start-pool audit, README and manifest remain unchanged.

Eligibility now requires the v4 two-seed development bridge to pass every frozen
gate, followed by a non-null clear winner from its reserved confirmation. The
parent protocol SHA256 is
`29d921b476807e68e129472d4090f47920af52ce60b6dc999d00f7fef820fe36`.
The extension is not run if either condition fails. There is no fallback to a
legacy or gel-only result, a merely ranked leader, or extra seeds.

The same six authentic measured starts and two fresh seeds, 904601 and 904602,
remain frozen: the Sept4 anchor control plus Aug22 starts 5928, 6273, 6128, 5963
and 6461. This gives 12 trials, including two anchor controls. Their selection
rule, file hashes, order, measured joints, aperture and wrist bias are unchanged.
All use the shared Sept4 tactile baseline and fixed successful waffle/scene.
This deliberately measures start sensitivity under one shared sensor baseline;
it does not constitute unseen-session transfer validation.

Only the prospectively accepted v4 combined controller profile is inherited:
gel coverage v2, gripper assembly wrist contact, live veto with current delivery
feedback, and normal release FINISH. Its simulator holds with sensors and safety
through 60 s unless an actual stop occurs. The extension cannot rerank models
or supply missing evidence for the original winner declaration.

This is a planning amendment, not an executed robustness result. The new manifest
checks the original frozen files and hashes the amendment separately.

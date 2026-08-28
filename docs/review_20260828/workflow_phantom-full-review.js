export const meta = {
  name: 'phantom-full-review',
  description: 'Full scan of the PHANTOM repo: 8 Opus lenses + 8 Codex second opinions, adversarial verification, synthesis into a ranked 3-week plan',
  phases: [
    { title: 'Find', detail: '8 Opus lenses over data / model / training / inference parity / closed-loop root cause / paper claims / deploy safety / self-improvement readiness' },
    { title: 'Codex', detail: 'independent Codex (gpt-5.4) pass per lens' },
    { title: 'Verify', detail: '2 adversarial refuters per high/medium finding' },
    { title: 'Synthesize', detail: 'ranked plan + completeness critic' },
  ],
}

const REPO = '/Users/sannikov/GitHub/phantom'
const OUT = '/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/review'
const CONTEXT = '/private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/research/CONTEXT.md'

const COMMON = `You are reviewing the PHANTOM robot-learning repo at ${REPO} (HEAD 3539988). Read ${CONTEXT} first: it states the architecture, the data, and the rig failure mode we are trying to fix (policy stops 65-120 mm above the object, closes on air, lifts anyway; executor exonerated). Also read ${REPO}/docs/rig_session_v5.md and ${REPO}/docs/training_playbook.md. Deadline is ~3 weeks (ICRA 2027). The goal of this review is to find everything that is wrong, weak, or missing — bugs, silent mismatches, design flaws, unverified assumptions, and missing experiments — and to say concretely how to fix each. Read the actual code (cite file:line); do not speculate from names. Run small python checks (cd ${REPO} && .venv/bin/python ...) whenever a claim can be checked in seconds; pytest is available. Do NOT modify any files under ${REPO}. Write your full narrative report (as long as needed) to ${OUT}/<lens>.md (mkdir -p first), then return ONLY the structured findings: at most 10, ranked by severity, each with file, line, a one-sentence claim, the evidence you saw, the concrete failure scenario, and the minimal fix. Severity: high = plausibly changes rig outcome or invalidates a paper claim; medium = correctness/validity risk; low = hygiene.`

const LENSES = [
  { key: 'data-pipeline', prompt: `${COMMON}
LENS: DATA PIPELINE. phantom/data/{episode_store,windows,schema,derived}.py, tools/intake_recovery.py, tools/episode_qc.py, manifests, norm_stats, phantom/train/common.py (WindowDataset, manifest_split, close_index, grasp-frac anchoring, photo-aug), configs/start_poses.yaml. Questions: is every input stream built identically for training and for the recorded rig episodes (you can compare with the rig data schema); is normalisation of ur_state (joints, velocities, tcp_speed) sane — what are the stds, does a 30%-slower arm land far in normalised space; do failure demos / recovery demos leak into val; is the close_index rule right for every task; are windows anchored in a way that biases toward the slow parts of demos; label degeneracy; NaN handling; anything that would make the offline metric look good while the rig fails. Write ${OUT}/data-pipeline.md.` },
  { key: 'model-losses', prompt: `${COMMON}
LENS: MODEL + LOSSES. phantom/model/{rf,phantom_dit,sequence,acc,attention_bias}.py, phantom/model/ace/*, phantom/model/hht/*, phantom/config/model.py, configs/training. Questions: through which path can the ACTION tokens actually see the tactile/contact information (attention masks, sequence layout, attention_bias) — is it possible the action head is structurally blind or weakly coupled to the gate/contact tokens; is the conditioning dropout (cond_dropout 0.10) and guidance applied to the right conditions; are the loss weights such that the video term dominates; is the flow-matching target/timestep sampling sensible for a 16x7 chunk; does prev_cpk feedback create a self-confirming loop (policy imagines contact, feeds it back); ACC two-pass; anything that would produce systematic under-commit (shrunken deltas) — e.g. mean-regression from MSE on multimodal chunks, EMA, normalisation. Write ${OUT}/model-losses.md.` },
  { key: 'training-loop', prompt: `${COMMON}
LENS: TRAINING LOOP + EVALUATION. phantom/train/{train_teacher,common,builder}.py, configs/training.py, tools/terminal_eval.py, tools/provision_v5.sh. Questions: LR/EMA/cosine interplay for the 3k fine-tune and for a 20k from-scratch run; grasp-frac anchoring vs the failure demos; DataLoader determinism; DDP correctness (all-reduce, no_sync, val sharding, seeds); checkpoint selection = EMA vs raw; is terminal_eval a valid proxy — it samples from DEMO states, so it cannot see covariate shift: design the cheapest offline metric that CAN (e.g. rollout-state replay from recorded rig episodes at data/episodes/deploy — describe exactly how to build it from planner_trace.json + zarr); is anything in the v5 fine-tune recipe likely to have reduced closed-loop commitment. Write ${OUT}/training-loop.md.` },
  { key: 'inference-parity', prompt: `${COMMON}
LENS: TRAIN/DEPLOY PARITY. phantom/inference/policy.py, phantom/deploy/{planner,executor,governor,safety,runtime}.py, phantom/scripts/run_deploy.py vs phantom/data/windows.py + phantom/train/common.py. Build a feature-by-feature parity table: for every model input (video cond frame timing/resolution/colour order, obs_gel, obs_mech, obs_proprio = ur_state composition and normalisation, wrist F/T window, text embedding, prev_cpk semantics and its zeroing at train time vs the true previous package at deploy, noise persistence, guidance, NFE, EMA) state how training builds it and how deploy builds it and whether they match; check the executor rebase/blend math for systematic shortening of commanded displacement; check the timing of the conditioning frame relative to the action chunk start (latency 0.9-1.3 s: the chunk is executed from a state ~1 s newer than the observation — how does training model that?). Write ${OUT}/inference-parity.md.` },
  { key: 'closed-loop-root-cause', prompt: `${COMMON}
LENS: CLOSED-LOOP ROOT CAUSE. The measured facts are in ${CONTEXT} and ${REPO}/docs/rig_session_v5.md; the rig episodes with planner_trace.json exist on compute3 (not on this machine) — reason from code and from the demo statistics in ${REPO}/configs/start_poses.yaml and norm stats. Produce ranked hypotheses for why the policy decelerates and stops 65-120 mm high and then lifts after closing on air (candidates: velocity/qd conditioning self-consistency, proprio z normalisation, observation latency, prev_cpk feedback, cond dropout making obs optional, video-gen prior dominating, terminal windows under-represented, time-anchored behaviour via visual cues, gripper-state coupling), each with the code evidence for/against and the CHEAPEST decisive offline experiment (exact inputs to swap, which recorded data, expected signature). Write the experiment plan as a concrete script outline (functions, data paths, metrics) in ${OUT}/closed-loop-root-cause.md.` },
  { key: 'paper-claims', prompt: `${COMMON}
LENS: PAPER CLAIMS + EXPERIMENTAL DESIGN. Read ${REPO}/docs/STATUS.md, README*, docs/*.md, and any paper/plan text in the repo. List every claim the paper intends to make (tactile WAM teacher, sensor-free student distillation, anticipatory contact, HID) and for each: what evidence exists in the repo today, what experiment is missing (baselines vision_only / no_distill, student, ablations, real-robot success rates, statistical protocol), what a hostile ICRA reviewer would attack, and the minimum credible experiment set achievable in 3 weeks with ~15-20 rig-hours and rented H100s. Be blunt about whether the current architecture is a WAM in the accepted sense and whether the video-generation head is defensible without an ablation. Write ${OUT}/paper-claims.md.` },
  { key: 'deploy-safety', prompt: `${COMMON}
LENS: DEPLOY SAFETY + ROBUSTNESS after today's safety batch (commit 3539988: joint gate, z floor, STOP hitbox on the commanded target, speed cap, gripper_ctl). phantom/deploy/*, phantom/drivers/real/*, phantom/scripts/run_deploy.py, phantom/scripts/gripper_ctl.py, tests/test_rig_safety_0828.py. Questions: remaining holes (hitbox checks the commanded target only — what about measured pose/overshoot; lift speed after a close; gripper force on pad-on-pad closes; cable wrap = wrist-3 turns; protective-stop recovery paths; what happens when the hitbox stops mid-chunk; homing moveJ sweep risk; e-stop state detection; RTDE reconnect); correctness bugs in the new code; tests that are missing. Write ${OUT}/deploy-safety.md.` },
  { key: 'self-improvement', prompt: `${COMMON}
LENS: SELF-IMPROVEMENT / DAGGER READINESS. phantom/dagger/*, the AWR config in phantom/config/training.py, phantom/recording/recorder.py relabel, phantom/data_collect/*, tools/intake_recovery.py. Questions: what exists to (a) run policy rollouts with noise, (b) auto-label success from tactile/gripper/lift signals (which streams carry what), (c) fold rollouts into the training manifest with weights, (d) retrain and stage; what is missing or broken for a 2-3 iteration filtered-BC / advantage-weighted loop with ~50-150 rollouts; and the concrete design (files to add, weighting formula, success rule with thresholds derived from the demo statistics in the repo) that would work within a week. Write ${OUT}/self-improvement.md.` },
]

const FINDINGS = {
  type: 'object', required: ['findings'],
  properties: { findings: { type: 'array', maxItems: 10, items: {
    type: 'object', required: ['title', 'file', 'line', 'severity', 'claim', 'evidence', 'failure_scenario', 'fix'],
    properties: {
      title: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' },
      severity: { type: 'string', enum: ['high', 'medium', 'low'] },
      claim: { type: 'string' }, evidence: { type: 'string' }, failure_scenario: { type: 'string' }, fix: { type: 'string' },
    } } } },
}
const VERDICT = {
  type: 'object', required: ['refuted', 'reason', 'confidence'],
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' }, confidence: { type: 'number' } },
}

phase('Find')
log('8 Opus lenses + 8 Codex passes running')
const codexPrompt = (l) => `You are orchestrating an independent Codex review. Run exactly this from ${REPO} (it takes several minutes; wait for it):
mkdir -p ${OUT}/codex && codex exec --sandbox read-only -m gpt-5.4 -c model_reasoning_effort=high -o ${OUT}/codex/${l.key}.md "DO NOT propose a plan or ask for approval — REPORT FINDINGS ONLY. $(cat <<'EOF'
${l.prompt.replace(/`/g, "'").replace(/\$/g, '')}
EOF
)" < /dev/null
(Write the heredoc content via a temp file if quoting is awkward; the point is that Codex receives the lens prompt with the instruction to only report findings and to cite file:line.) If codex fails on model availability, retry once with -m gpt-5.4. Then read ${OUT}/codex/${l.key}.md and return its findings as the structured list (max 10, ranked), copying Codex's file/line/evidence faithfully; do not add your own findings.`

const perLens = await pipeline(LENSES,
  l => parallel([
    () => agent(l.prompt, { label: `opus:${l.key}`, phase: 'Find', schema: FINDINGS, effort: 'xhigh' }),
    () => agent(codexPrompt(l), { label: `codex:${l.key}`, phase: 'Codex', schema: FINDINGS, effort: 'low' }),
  ]).then(([o, c]) => ({ lens: l.key, opus: o ? o.findings : [], codex: c ? c.findings : [] })),
)

// barrier is legitimate here: dedup across lenses + cap before the expensive verify stage
const all = perLens.filter(Boolean).flatMap(r => [
  ...r.opus.map(f => ({ ...f, lens: r.lens, source: 'opus' })),
  ...r.codex.map(f => ({ ...f, lens: r.lens, source: 'codex' })),
])
const seen = new Map()
for (const f of all) {
  const k = `${f.file}:${Math.round(f.line / 15)}`
  if (!seen.has(k)) seen.set(k, f)
  else seen.get(k).corroborated = true
}
const uniq = [...seen.values()]
const order = { high: 0, medium: 1, low: 2 }
uniq.sort((a, b) => order[a.severity] - order[b.severity])
const toVerify = uniq.filter(f => f.severity !== 'low').slice(0, 48)
log(`${all.length} raw findings → ${uniq.length} unique; verifying ${toVerify.length} high/medium (${uniq.length - toVerify.length} low/overflow skipped, kept in report)`)

phase('Verify')
const verified = await pipeline(toVerify,
  f => parallel(['correctness', 'does-it-matter-on-the-rig'].map(lens => () =>
    agent(`Adversarially verify this review finding about ${REPO} through the "${lens}" lens. Read the cited code yourself (${f.file}:${f.line} and around it), run a quick python/pytest check when possible, and try to REFUTE it. Finding: ${f.title}. Claim: ${f.claim}. Evidence: ${f.evidence}. Failure scenario: ${f.failure_scenario}. Proposed fix: ${f.fix}. For the correctness lens: is the claim factually true of the code? For the rig lens: would fixing it plausibly change the closed-loop outcome or the validity of a paper claim (${CONTEXT} describes the failure mode) — if it is true but irrelevant, refute. Default to refuted=true if uncertain. Return refuted, reason (2-4 sentences with file:line), confidence 0-1.`,
      { label: `verify:${f.lens}:${lens}`, phase: 'Verify', schema: VERDICT, effort: 'high', model: 'opus' })))
    .then(vs => ({ ...f, votes: vs.filter(Boolean), survives: vs.filter(Boolean).filter(v => !v.refuted).length >= 2 })),
)
const confirmed = verified.filter(Boolean).filter(f => f.survives)
const rejected = verified.filter(Boolean).filter(f => !f.survives)
log(`${confirmed.length} confirmed, ${rejected.length} refuted`)

phase('Synthesize')
const digest = JSON.stringify({ confirmed, rejected: rejected.map(f => ({ title: f.title, file: f.file, reasons: f.votes.map(v => v.reason) })), low: uniq.filter(f => f.severity === 'low').map(f => ({ title: f.title, file: f.file, line: f.line, fix: f.fix })) })
const synthesis = await agent(`You are the lead engineer synthesizing a full review of ${REPO} into a 3-week plan (ICRA 2027, one rig, 2 operators, rented H100s). Read ${CONTEXT}. The per-lens narrative reports are in ${OUT}/*.md and ${OUT}/codex/*.md — read them all. Literature reports MAY exist in /private/tmp/claude-501/-Users-sannikov/a5af36c2-8450-43d1-9a34-fa0cdb820706/scratchpad/research/0*.md — read any that exist. Confirmed/refuted findings (JSON): ${digest.slice(0, 120000)}.
Write ${OUT}/REVIEW_SYNTHESIS.md with: (1) the 10 most important confirmed problems, each with file:line, why it matters for the rig or the paper, and the exact fix; (2) the ranked root-cause hypotheses for the closed-loop failure and the decisive offline experiments to run first (with data paths and expected signatures); (3) a day-by-day 3-week plan: fixes → offline experiments → rig sessions → training runs → paper experiments, with explicit GO/NO-GO gates and the rig-hour and GPU-dollar budget per step; (4) what to CUT if things slip; (5) the refuted findings in one line each so nobody re-raises them. Then return a 25-line executive summary.`,
  { label: 'synthesis', phase: 'Synthesize', effort: 'xhigh', model: 'opus' })
const critic = await agent(`Completeness critic. Read ${OUT}/REVIEW_SYNTHESIS.md, ${CONTEXT}, and skim ${OUT}/*.md. What is missing from the plan: a subsystem nobody reviewed, a claim in the plan not backed by a finding or a paper, an experiment whose result cannot actually change a decision, a rig-time or GPU budget that does not add up, a safety item, a paper-reviewer attack not addressed. Return at most 12 bullet lines, each starting with the missing item and ending with what to do about it. Append the same list to ${OUT}/REVIEW_SYNTHESIS.md under a heading "Completeness critic".`,
  { label: 'critic', phase: 'Synthesize', effort: 'high', model: 'opus' })
return { summary: synthesis, critic, confirmed: confirmed.length, rejected: rejected.length, raw: all.length }
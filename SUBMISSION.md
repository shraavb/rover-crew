# RoverCrew — Cerebras × Gemma 4 Hackathon submission

**Track 1: Multiverse Agents (Best Multi-Agent + Multimodal)** — primary
(Optionally Track 2: People's Choice — also post the video on X, tag @Cerebras @googlegemma.)

## Discord post (copy-paste)

- **Project Name:** RoverCrew
- **Team Members:** @<your-discord-handle> (Shraavasti Bhat)
- **Project Description:** RoverCrew is an embodied multi-agent system where four
  Gemma 4 (31B) agents — perception, planner, critic, and safety — run on
  Cerebras to pilot a Waveshare UGV Beast rover. Every control tick the crew
  sees a camera frame, proposes a move, critiques it for goal alignment, and
  applies a safety veto, then drives the rover's Cyberwave digital twin in real
  time. Cerebras' ultra-fast inference is what makes four sequential multimodal
  LLM calls per loop viable — the difference between a slideshow and a
  responsive robot.
- **Demo Video:** (attached, ≤60s)
- **GitHub Repository:** https://github.com/shraavb/rover-crew

## How it maps to Track 1 judging

- **Agent Collaboration:** 4 independent Gemma-4 agents in a pipeline
  (perception → planner → critic → safety); the safety agent can veto/override
  the planner+critic (seen live when a person blocks the path → `stop`).
- **Multimodal Intelligence:** perception is a Gemma-4 *vision* call on the
  camera frame; the other agents reason over its structured output.
- **Speed in Action:** the loop prints `Nms/4-agents` per step — four Gemma-4
  calls back-to-back on Cerebras. Run with `MAX_RPM` raised (hackathon rate
  increase) to show full speed.
- **Innovation:** physical-AI / embodied agent driving a real rover + its
  Cyberwave digital twin.

## Demo video script (~60s)

1. **0–8s** — startup banner on screen: `brain: Cerebras · gemma-4-31b`,
   `agents: perception → planner → critic → safety`, `body: Cyberwave UGV Beast`.
   One-liner: "Four Gemma-4 agents on Cerebras pilot a rover in real time."
2. **8–40s** — terminal + Cyberwave viewport side by side. Hold up a red cup:
   perception flips `target_visible`, the twin turns toward it and drives in —
   point at the `Nms/4-agents` number to show Cerebras speed.
3. **40–52s** — trigger the safety veto (step in front → `safety … -> DO stop`)
   to show real multi-agent coordination.
4. **52–60s** — tagline + GitHub URL.

## Pre-record privacy checklist (REQUIRED — judges call this out)

- [ ] **Rotate the Cyberwave API key first** (it was printed in an earlier
      terminal error). Regenerate in dashboard, update `.env`.
- [ ] Fresh terminal — no scrollback showing the old 500 error / tokens.
- [ ] Do **not** show `.env`, browser address bar with tokens, notifications,
      or email.
- [ ] Close unrelated browser tabs in the recording frame.

## Run command (for recording)

```bash
cd ~/rover-crew
# raise MAX_RPM if you have the hackathon rate increase, to show full speed
MAX_RPM=600 USE_SIM=1 SIM_FRAME=webcam ./.venv/bin/python main.py "red cup"
```

# Roro — Cerebras × Gemma 4 Hackathon submission

**Track 1: Multiverse Agents (Best Multi-Agent + Multimodal)** — primary
(Optionally Track 2: People's Choice — also post the video on X, tag @Cerebras @googlegemma.)

## Discord post (copy-paste)

- **Project Name:** Roro — always-on voice-commanded home rover
- **Team Members:** @<your-discord-handle> (Shraavasti Bhat)
- **Project Description:** Roro is a home rover you talk to. Say *"robot, move
  toward the orange case"* — local Whisper transcribes it and a Gemma 4 (31B)
  **intent-parser agent** on Cerebras turns it into a structured command (move
  toward / away from / around an object, turn, or avoid it). While driving, a
  Gemma-4 **perception agent** reads the onboard camera every step and a
  **safety agent** vetoes unsafe moves and steers around obstacles. Roro listens
  continuously, so you can re-command it mid-motion — *"robot, stop"* — and it
  switches at the next step. Cerebras' speed is what makes a live speech + vision
  control loop feel real-time. Runs on a physical Waveshare UGV Beast and in a
  self-contained MuJoCo simulation.
- **Demo Video:** (attached, ≤60s)
- **GitHub Repository:** https://github.com/shraavb/rover-crew

## How it maps to Track 1 judging

- **Agent Collaboration:** three Gemma-4 agents collaborate around a shared
  control loop — an **intent-parser** (speech → structured command), a
  **perception** agent (camera → target bearing/distance/obstacles each step),
  and a **safety** agent that vetoes unsafe actions and deterministically steers
  around obstacles (never blocks arrival). Deterministic per-intent code policies
  turn the agents' reports into reliable motion (hybrid control: the model
  perceives, code steers — chosen after the LLM's raw distance/bearing estimates
  proved too noisy to drive geometry directly).
- **Multimodal Intelligence:** *speech* (Whisper) + *vision* (Gemma-4 perception
  on the onboard camera) + *language* (intent parsing). Perception discriminates
  the target from same-shape distractors (e.g. the orange cup, not the red or
  yellow one) in the MuJoCo scene.
- **Speed in Action:** every step runs Gemma-4 vision + safety on Cerebras inside
  a stop-look-move loop, plus an always-on Whisper listener — fast enough that
  speech-driven, vision-in-the-loop control runs in near real time.
- **Innovation:** embodied physical-AI — an always-on, voice-commanded home robot
  that understands spatial intents AND accepts barge-in commands while moving
  (mid-motion preemption). Demoable with zero hardware via the MuJoCo twin.

## Demo video script (~60s)

1. **0–8s** — startup: `brain: Cerebras · gemma-4-31b`, `body: …`, and the
   always-on prompt `👂 say 'robot <command>'`. One-liner: "Talk to Roro — Gemma 4
   on Cerebras turns your words into robot actions, in real time."
2. **8–24s** — **voice**: say *"robot, move toward the orange cup."* Show the
   heard text + parsed intent, then the rover searching, **ignoring the red /
   yellow distractor cups**, and driving to the orange one (`REACHED 🎯`).
3. **24–40s** — **barge-in**: while it's driving, say *"robot, move away from the
   orange cup"* → it preempts at the next step, turns ~180°, and drives off. This
   is the always-on, re-command-while-moving moment.
4. **40–52s** — **safety agent**: an obstacle moment — `obstacle ahead -> steer`
   — the rover routes around a crate instead of stopping. Multi-agent in action.
5. **52–60s** — tagline + GitHub URL.

> Tip for the "Show Cerebras speed" requirement: keep the terminal visible so the
> rapid per-step `see=… -> DO …` lines stream by, and call out that each line is a
> live Gemma-4 vision call on Cerebras.

## Run commands (for recording)

```bash
cd ~/rover-crew
# MuJoCo sim (no hardware). Raise MAX_RPM to show full Cerebras speed; tune the
# turn/step constants so motion is watchable on camera.
export USE_MJC=1 MAX_RPM=600 TURN_PULSES_90=4 MOVE_TIME=0.4 SETTLE_TIME=0.2

# always-on voice: just talk — "robot <command>" (no key press):
./.venv/bin/python main.py voice

# or typed commands (no mic):
./.venv/bin/python main.py "move toward the orange cup"
./.venv/bin/python main.py "move away from the orange cup"
./.venv/bin/python main.py "go around the orange cup"
./.venv/bin/python main.py "turn left"
./.venv/bin/python main.py "avoid the orange cup"
```

On the **physical rover**: drop `USE_MJC=1` and the sim env vars (set `ROVER_HOST`,
run `pi_server.py` on the Pi); leave `TURN_PULSES_90` at its hardware default.
Override the wake word with `WAKE_WORD=<word>` if "robot" misfires in your room.

## Pre-record privacy checklist (REQUIRED — judges call this out)

- [ ] **Rotate the Cyberwave API key first** — it printed in an earlier terminal
      error. Regenerate in the dashboard, update `.env`.
- [ ] Fresh terminal — no scrollback showing old errors / tokens.
- [ ] Don't show `.env`, address bars with tokens, notifications, or email.
- [ ] The MuJoCo demo window shows no keys — clean to record as-is.
- [ ] If recording the **physical rover**, the camera shows your room — clear any
      personal items / screens with sensitive info from view.

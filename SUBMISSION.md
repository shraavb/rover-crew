# Roro — Cerebras × Gemma 4 Hackathon submission

**Track 1: Multiverse Agents (Best Multi-Agent + Multimodal)** — primary
(Optionally Track 2: People's Choice — also post the video on X, tag @Cerebras @googlegemma.)

## Discord post (copy-paste)

- **Project Name:** Roro — voice-commanded home rover
- **Team Members:** @<your-discord-handle> (Shraavasti Bhat)
- **Project Description:** Roro is a voice-commanded home rover: you speak a
  command, local Whisper transcribes it, and Gemma 4 (31B) on Cerebras parses it
  into a structured intent — move toward / away from / around an object, turn, or
  avoid it. A four-agent Gemma-4 crew (perception → planner → critic → safety)
  then drives the rover, seeing the scene through its onboard camera each step
  and picking the right object out of look-alikes (e.g. the orange cup, not the
  red or yellow one). Cerebras' ultra-fast inference is what makes a speech +
  4-vision-calls control loop run in near real time. Runs on a physical Waveshare
  UGV Beast and in a self-contained MuJoCo simulation.
- **Demo Video:** (attached, ≤60s)
- **GitHub Repository:** https://github.com/shraavb/rover-crew

## How it maps to Track 1 judging

- **Agent Collaboration:** a parser agent turns speech → intent, then a 4-agent
  pipeline (perception → planner → critic → safety) executes it; the safety agent
  deterministically overrides unsafe moves (steers around obstacles, never blocks
  arrival).
- **Multimodal Intelligence:** *speech* (Whisper) + *vision* (Gemma-4 perception
  on the onboard camera) + *language* (intent parsing & planning). Perception
  discriminates the target from same-shape distractor cups of other colours.
- **Speed in Action:** the loop prints `Nms/4-agents` per step — four Gemma-4
  calls back-to-back on Cerebras (~0.8–1.2 s). Raise `MAX_RPM` with the hackathon
  rate increase to show full speed.
- **Innovation:** embodied physical-AI — a spoken-language home robot that
  understands spatial intents and acts on them, demoable with zero hardware via
  the MuJoCo twin.

## Demo video script (~60s)

1. **0–8s** — startup banner: `brain: Cerebras · gemma-4-31b`, `agents:
   perception → planner → critic → safety`, `body: MuJoCo simulation`. One-liner:
   "Talk to Roro — Gemma 4 on Cerebras turns your words into robot actions."
2. **8–22s** — **voice**: say *"move toward the orange cup."* Show the
   transcription + parsed intent, then the rover searching, **ignoring the red /
   yellow / blue distractor cups**, and driving to the orange one (`REACHED`).
   Point at `Nms/4-agents` for Cerebras speed.
3. **22–38s** — say *"move away from the orange cup"* → it turns ~180° and drives
   off. Then *"avoid the orange cup"* → it steers clear while moving.
4. **38–52s** — obstacle moment: the **safety agent** steers around a crate
   (`obstacle ahead -> steer`) — multi-agent coordination in action.
5. **52–60s** — tagline + GitHub URL.

## Run commands (for recording)

```bash
cd ~/rover-crew
# MuJoCo sim (no hardware). MAX_RPM raised to show full Cerebras speed.
# TURN_PULSES_90=4 ≈ a quarter turn at the sim turn rate; MOVE_TIME makes motion
# watchable. A "RoverCrew - MuJoCo" window opens (chase view + onboard inset).
export USE_MJC=1 MAX_RPM=600 TURN_PULSES_90=4 MOVE_TIME=0.4 SETTLE_TIME=0.2

# voice (push-to-talk: speak, press Enter):
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

## Pre-record privacy checklist (REQUIRED — judges call this out)

- [ ] **Rotate the Cyberwave API key first** — it printed in an earlier terminal
      error. Regenerate in the dashboard, update `.env`.
- [ ] Fresh terminal — no scrollback showing old errors / tokens.
- [ ] Don't show `.env`, address bars with tokens, notifications, or email.
- [ ] The MuJoCo demo window shows no keys — clean to record as-is.

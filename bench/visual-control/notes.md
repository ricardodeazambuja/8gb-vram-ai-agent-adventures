# Visual control probe: what the models actually did

Notes from reading the answers, the reasoning and the step logs in `results/*.json`. The numbers
are in `results/summary.md`; this is the why behind them. Runs from 2026-10-28.

## LFM3-VL-2B, glance v1 (one picture, one decision)

These are the v1 scenes and prompts (see "A flaw of ours" at the end); the v2 numbers are in
`results/summary.md`.

**Thinking off (45/60, 0.4 s a decision).** It sees well and points well, and it ignores the rules.

- 🛢️ **Tanks 12/12**, including the three scenes where nothing is low: it makes no call, as asked.
- 🖱️ **Click 12/12**: every click lands inside the right button, and for the two "Export"/"Print"
  scenes where the button doesn't exist it clicks nothing.
- 🤏 **Pick 11/12**: one miss on "the topmost object".
- 🚦 **Signal 6/12**: it said `go` on every yellow light and on a green light with a person on the
  crosswalk. Only red lights got a `stop`.
- ⏲️ **Gauge 4/12**: it read the dial within a few units (93 read as 90, 12 as 15) and then called
  `report` every time, never `open_valve` or `close_valve`, whatever the value. The reading is
  there; the rule ("above 70, open") isn't applied.

**Thinking on (52/60, median 3 s).** The rules get applied, and a new failure appears.

- Signal and gauge go to **12/12**. The thinking is short (100–1,000 tokens), and the readings
  are good: tanks within 1–2 points (47/48/74 read as "≈48/48/75"), the gauge rounded to the
  nearest 5 (47 → 45, 62 → 60).
- When it declines, it explains: "There is no "Print" button in this dialog. The visible buttons
  are Export, Cancel, Delete, Save, and Share only", listing the buttons correctly.
- 🖱️🤏 **But click and pick drop to 8/12**, all eight misses the same way: it finds the target and
  then *writes* the coordinates instead of calling the tool ("The 'Help' button is located at
  approximately [546, 486]."). The coordinates are right; the call is missing. Forcing a tool
  call (`tool_choice: required`) turns these into correct calls (checked on two of them).

## LFM3-VL-2B, grid (steer a robot, one move per step)

**Stateless, thinking off: 3/8 open, 0/8 with walls.** Two failure modes:

- It **refuses** a lot: in the failed episodes most steps are "I'm sorry, but I can't complete
  this request. The information provided does not specify a valid move for the robot…" or "…does
  not specify a starting position for the robot". The picture has all of it; without thinking it
  doesn't take the position from the image and asks for it in text.
- With walls it **walks into the edge**: in walls episode 0 it moved down five times, reached the
  bottom row, and kept calling `down` (blocked) between refusals.

**Stateless, thinking on: 8/8 open, 3/8 with walls**, at about 45 s a step. Open grids take 7–10
steps against the shortest 5–9; the three mazes it solved took 15–16 steps against 7–10. It
almost never walks into a wall (6 blocked moves in 229 steps, against 50 without thinking), but
77 steps ended with no move and no text at all: the 6,000-token thinking budget ran out before a
call (this run didn't save the reasoning; the later ones do). A maze episode takes 10–20
minutes this way.

**Full history, thinking off: 5/8 open, 0/8 with walls.** Keeping the conversation helps on the
open grid (5/8 against 3/8 stateless, and the solved ones on the shortest path), and then the
pibench failure shows up in miniature. In four of the eight mazes it wrote its reasoning as plain
text instead of calling `move` ("I need to identify the robot's current position and determine a
valid move toward the green square. Let me analyze the grid. The grid is 7x7. The blue robot
appears at row 2 (0-indexed) column 6? Act…"), and then **copied that same paragraph, word for
word, at every following step**: 28 steps without a move in walls episode 0. With its own
previous answer in the context, it repeats it. That is the loop that cost it pibench, and it is
why a stateless design (the harness keeps the state, each step starts fresh) suits this model.

## Forcing a tool call (`tool_choice: required`)

llama-server can constrain the answer to a tool call. With a call forced the model can no longer
decline by not calling, so every request then also offers a `no_action(reason)` tool.

- **It isn't watertight.** The grammar allows free text *before* the call, so a model that starts
  writing can write until `max_tokens` and never get to it. One thinking-mode glance answer ran
  6,000 tokens of text ("The 'Help' button is located at approximately **[546, 487]**…"), and in
  the full-history grid run 203 of 354 replies were text again: refusals, "I need to investigate
  this further…", and the copied paragraph from above. Stateless, it held: 4 text replies in 280
  steps.
- **`no_action` gets misused.** Without thinking, it was called to stall ("I need to identify the
  bottommost object before picking it", three times); with thinking, once as "done" ("User
  requested the point of the bottommost object; no further action required").
- **Grid, stateless, no thinking: 6/8 open (from 3/8), still 0/8 with walls.** The refusals are
  gone and open grids are walked on the shortest path when the way is straight. But at a wall it
  presses the same direction again and again: in walls episode 0 it called `down` into a wall 23
  times in a row, although every prompt listed its last moves as "down (blocked)". Blocked moves
  went from 50 to 167. It doesn't use the feedback.
- **Grid, stateless, thinking on: 8/8 open, 3/8 with walls**, exactly as without forcing, at
  ~43 s a step (2 h 55 min for 16 episodes). Forcing changes nothing here, because the grammar
  only applies after the thinking ends: 90 of 243 steps still ended with no call, and this run
  saved the reasoning, so we can see why. In 84 of those 90 the reasoning stops mid-sentence
  ("…Then down to (3,4). Then down to (4,4). Then left? Actually need to get to column5
  eventually. At "): the 6,000-token budget ran out while it was re-deriving the whole maze
  cell by cell, which it does from scratch at every step because each step starts fresh. When it
  does call, it rarely hits a wall (8 blocked moves in 153). Careful but too slow, and a bigger
  budget would only make it slower.

## Qwen3-VL-8B for comparison (an instruct model: no thinking)

- **Glance (v1): 50/60** at 1.0 s a decision, against LFM3-VL's 45 (thinking off) and 52–58
  (thinking on). It applies the traffic rules (12/12) where LFM3-VL without thinking can't. It is
  **over-eager with the tanks**: in three of the four "nothing is low" scenes it refilled tanks at
  47–63% anyway (8/12). On the gauge it reads 12 as "10" and then calls `report(10)` instead of
  `close_valve`, the rule misapplied (7/12).
- **Grid: 8/8 open, 0–1/8 with walls**, never a step without a move (it always calls). It never
  hammers one blocked direction like LFM3-VL, but it wanders: open grids take 1.6–1.8× the
  shortest path, and in mazes it falls into four-move cycles ("right, up (blocked), left, down
  (blocked), right, …") until the step cap.

## A flaw of ours in glance v1

Both models "missed" the same pick scene (number 1, "the topmost object") at nearly the same point
(493, 292 and 495, 287). Looking at the scene, the orange triangle's centre is only ~7 px above the
red square's, and with the triangle's tip it's a matter of taste. And the click and pick prompts
said "Click the 'Help' button. Give the point in 0-1000 coordinates", and the thinking model took
the second sentence literally ("I should not click anything else; just give coordinates."). Glance
v2 requires the "leftmost/topmost…" object to lead by 60 px and names the tool in the prompt ("Use
the click tool to click…"). The v1 results are kept in `results/v1/`; the tables use v2.

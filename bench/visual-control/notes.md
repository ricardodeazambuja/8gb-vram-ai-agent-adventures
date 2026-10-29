# Visual control probe: what the models actually did

Notes from reading the answers, the reasoning and the step logs in `results/*.json`. The numbers
are in `results/summary.md`; this is the why behind them. Runs from 2026-10-28.

## LFM3-VL-2B, glance v1 (one picture, one decision)

These are the v1 scenes and prompts (see "A flaw of ours" below); v2 is further down.

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

## LFM3-VL-2B, glance v2 with 30 scenes of each kind (forced calls)

Twelve scenes per kind leave the percentages coarse, so the forced-call runs were repeated with 30
(`glance-lfm3vl-n30-*`).

- **Thinking on: 144/150, median 2.7 s.** Click, signal and gauge are perfect (30/30 each), pick
  29/30 (one pick at the wrong object), tanks 25/30. The tank misses show something the 12-scene
  set couldn't, because none of those scenes had two low tanks: with two or three tanks to refill,
  the reasoning plans every call ("call refill with tank "A", then refill with tank "B"") and the
  answer carries only the first one, in 4 of 6 such scenes. In a real loop it would get the tool
  result back and could do the next one; in one picture, one decision, it's a miss. The fifth
  miss refilled tank B at 70%, nowhere near low.
- **Thinking off: 117/150, median 0.4 s.** Click 30/30, pick 26/30, signal 27/30 (`go` on a green
  light with a person on the crosswalk, all three times that came up), tanks 24/30, gauge 10/30.
  Without thinking it **never makes more than one call**, so every scene with two or three low
  tanks is a miss (6 of 6). The gauge is the same story as in v1: 20 of 30 answered with `report`,
  the reading usually within 5 (93 → 90, 6 → 5), the rule ignored; a few readings were far off
  (3 → 35, 12 → 45). Two picks were `no_action` stalls ("I need to identify the bottommost
  object before picking it"), one was at (6, 6).

## Glance v2 (unambiguous scenes, prompts that name the tool)

- **LFM3-VL-2B with thinking: 60/60** (59/60 with a forced call: one yellow light answered with
  `stop`). The v1 click and pick misses are gone: told to "use the click tool", it uses it. So
  the v1 "writes coordinates instead of calling" failure was mostly our prompt.
- **LFM3-VL-2B without thinking: 45/60, 50/60 forced.** Unchanged from v1 where v1 was fair:
  the gauge gets `report` 8 times of 12, the yellow light gets `go`. One odd, repeatable effect:
  forcing a call fixes the traffic light (7/12 → 12/12 here, 6 → 12 in v1, 27/30 at n=30). The
  only differences in the request are `tool_choice` and the extra `no_action` tool; we haven't
  pinned down which one does it.
- **Qwen3-VL-8B: 51/60, 49/60 forced, 124/150 at n=30 forced.** Click, pick and signal near
  perfect; the misses are the same two habits as in v1. Tanks: it refills tanks that aren't low
  (12 of 30 scenes wrong, refilling at 47–91%, sometimes all three tanks). Gauge: 12 of 30 dials
  got `report` instead of a valve, most of them low values read as "10" and reported (9 times),
  although 10 is under 20; twice 97 was read as 90 and reported; one 59 got `open_valve`.
- **One pick both models miss:** scene 1, "the bottommost object". The yellow square (y 525 of
  600) is 75 px below the red circle, and both Qwen3-VL and LFM3-VL without thinking picked the
  red circle. A real miss this time, not an ambiguous scene; with thinking LFM3-VL gets it.

## Qwen3-VL-8B, grid with full history: 4/8 open, 2/8 with walls

Worse than stateless (8/8 open), and for a new reason. In all five episodes that ended with
steps without a move, it stopped **exactly one cell short of the goal** and declared victory: "The
blue robot is now on the same cell as the green square. The task is completed." Then, with a
fresh picture after every step showing the robot still next to the goal, it repeated the claim
6 to 18 times in a row ("The blue robot is already on the green square, so no further moves are
needed."). With its own earlier answer in the context it believes that over the image, the
same trap LFM3-VL fell into (there it copied a paragraph word for word, here it rephrases).
Stateless, it never stopped calling. In walls episode 0 it pressed `down` into a wall 14 times
in a row. Lesson for both models: don't keep the conversation, keep the state.

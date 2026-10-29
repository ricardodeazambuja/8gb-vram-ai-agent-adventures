# Visual control probe: what the models actually did

Notes from reading the answers, the reasoning and the step logs in `results/*.json`. The numbers
are in `results/summary.md`; this is the why behind them. Runs from 2026-10-28.

## LFM3-VL-2B, glance (one picture, one decision)

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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
logdir = sys.argv[1]
tag = sys.argv[2]
lo, hi = int(sys.argv[3]), int(sys.argv[4])
ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
ea.Reload()
ev = ea.Scalars(tag)
inside = [e for e in ev if lo <= e.step <= hi]
print("count", len(inside), "step_min", inside[0].step if inside else None, "step_max", inside[-1].step if inside else None)

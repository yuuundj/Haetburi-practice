import sys, itertools, time, pickle, numpy as np
sys.path.insert(0, "tools")
import ankle_search as A
t = time.time()
res = []
for lower in ("pitch", "roll"):
    lo_shaft, up_shaft = ("y", "x") if lower == "pitch" else ("x", "y")
    for (ln, lh), (un, uh) in itertools.product(A.dirs_perp(lo_shaft), A.dirs_perp(up_shaft)):
        for ls, us, dz in itertools.product(np.arange(-24, 24.1, 4.0), np.arange(-24, 24.1, 4.0),
                                            np.arange(0, 60.1, 5.0)):
            r = A.evaluate(lower, lh, uh, ls, us, dz)
            if r:
                r["names"] = (ln, un); res.append(r)
res.sort(key=lambda r: r["cost"])
print(len(res), "layouts in", round(time.time() - t), "s")
ranked = A.rank_with_bun(res, top=80)
print(f"re-ranked by the rounded foot (exponent {A.P.FOOT_EXPONENT}):")
for f, r in ranked[:10]:
    print(f"  {r['lower']:5s} low {r['names'][0]} up {r['names'][1]} shiftL {r['lo_shift']:4.0f} shiftU {r['up_shift']:4.0f} dz {r['dz']:3.0f}"
          f" | foot L {f['L']:5.1f} W {f['W']:5.1f} H {f['H']:5.1f} offset-out {f['cy']:4.0f} cost {f['cost']:6.1f}"
          f" | box L {r['L']:5.1f} W {r['W']:5.1f} H {r['H']:5.1f}")
pickle.dump(ranked[:10], open(sys.argv[1], "wb"))

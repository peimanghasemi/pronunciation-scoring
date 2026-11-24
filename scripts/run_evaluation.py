"""Compare phonetically weighted scoring with the plain phoneme error rate.

Agreement with the (simulated) clinician ratings is measured with Spearman
correlation and quadratic weighted kappa, with 95% intervals from a
bootstrap over speakers. Results go to results/evaluation.md.
"""
from pathlib import Path

import numpy as np

from pronscore import load_lexicon, score_attempt
from pronscore.evaluate import cluster_bootstrap, quadratic_kappa, spearman, to_rating
from pronscore.simulate import SEVERITY, simulate

rows = simulate(load_lexicon(), n_speakers=40, attempts=30, seed=1)
rating = np.array([r["rating"] for r in rows])
speaker = np.array([r["speaker"] for r in rows])
severity = np.array([r["severity"] for r in rows])
conf = np.array([r["confidence"] for r in rows])

weighted = np.array([score_attempt(r["word"], r["target"], r["recognised"]).score for r in rows])
plain = np.array([100 * max(0.0, 1 - score_attempt(r["word"], r["target"], r["recognised"]).per)
                  for r in rows])

def rho(s, y): return spearman(s, y)
def kappa(s, y): return quadratic_kappa(to_rating(s), y, 5)

lines = ["# Evaluation on simulated therapy attempts", "",
         f"{len(rows)} attempts from {len(np.unique(speaker))} speakers "
         f"({', '.join(SEVERITY)} severity), ratings 1 to 5. "
         "95% intervals from 1000 bootstrap samples over speakers.", "",
         "| scoring | Spearman | weighted kappa |", "|---|---|---|"]
for name, s in [("plain phoneme error rate", plain), ("phonetically weighted", weighted)]:
    r_ci = cluster_bootstrap(rho, speaker, s=s, y=rating)
    k_ci = cluster_bootstrap(kappa, speaker, s=s, y=rating)
    lines.append(f"| {name} | {rho(s, rating):.3f} ({r_ci[0]:.3f} to {r_ci[1]:.3f}) | "
                 f"{kappa(s, rating):.3f} ({k_ci[0]:.3f} to {k_ci[1]:.3f}) |")

diff = cluster_bootstrap(lambda w, p, y: rho(w, y) - rho(p, y), speaker, w=weighted, p=plain, y=rating)
lines += ["", f"Difference in Spearman (weighted minus plain): {rho(weighted, rating) - rho(plain, rating):.3f}, "
          f"95% interval {diff[0]:.3f} to {diff[1]:.3f}.", "",
          "## By severity", "",
          "| severity | attempts | mean rating | mean score | exact agreement | within one point |",
          "|---|---|---|---|---|---|"]
pred = to_rating(weighted)
for sev in SEVERITY:
    m = severity == sev
    lines.append(f"| {sev} | {m.sum()} | {rating[m].mean():.2f} | {weighted[m].mean():.1f} | "
                 f"{np.mean(pred[m] == rating[m]):.2f} | {np.mean(abs(pred[m] - rating[m]) <= 1):.2f} |")

low = conf < 0.5
lines += ["", "## Low recognition confidence", "",
          f"{low.sum()} attempts ({100 * low.mean():.1f}%) fall below the confidence threshold of 0.5. "
          f"Exact agreement is {np.mean(pred[~low] == rating[~low]):.2f} on the remaining attempts and "
          f"{np.mean(pred[low] == rating[low]):.2f} on the flagged ones.", ""]

Path("results").mkdir(exist_ok=True)
Path("results/evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

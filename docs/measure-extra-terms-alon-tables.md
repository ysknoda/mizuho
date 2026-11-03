# Generated tables: candidate terms on the Al-O-N zero-error classes

Output of `scripts/measure_extra_terms.py report` (2026-11-01) on runs/alon (record, classes,
MACE ledger, ranking.csv, excluded.csv) with the seven re-optimisation ledgers. Summary and
reading in [measure-extra-terms-alon.md](measure-extra-terms-alon.md).

Zero-error classes: 2565; relaxed by MACE: 2565; zero-error entries: 4735.

## At the recorded points

| term(s) | classes kept | removed: excl. anion-anion | removed: excl. not converged | removed: saddle | removed: > 150 meV | removed: <= 100 meV | removed: stage-1 picks | median E kept (meV) | % kept <= 100 meV |
|---|---|---|---|---|---|---|---|---|---|
| none | 2565 | 0/419 | 0/19 | 0/1857 | 0/1855 | 0/52 | 0/19 | 278 | 2.0 |
| bvs_0.10 | 145 | 406/419 | 19/19 | 1757/1857 | 1745/1855 | 48/52 | 17/19 | 284 | 2.8 |
| bvs_0.15 | 324 | 372/419 | 18/19 | 1616/1857 | 1615/1855 | 47/52 | 16/19 | 273 | 1.5 |
| bvs_0.20 | 860 | 277/419 | 15/19 | 1198/1857 | 1226/1855 | 39/52 | 13/19 | 272 | 1.5 |
| bvs_0.25 | 1128 | 236/419 | 14/19 | 1014/1857 | 1026/1855 | 36/52 | 10/19 | 274 | 1.4 |
| bvs_0.30 | 1692 | 161/419 | 8/19 | 634/1857 | 617/1855 | 18/52 | 7/19 | 274 | 2.0 |
| aa_2.0 | 754 | 381/419 | 18/19 | 1345/1857 | 1228/1855 | 40/52 | 14/19 | 284 | 1.6 |
| aa_2.2 | 366 | 409/419 | 19/19 | 1628/1857 | 1550/1855 | 44/52 | 18/19 | 285 | 2.2 |
| aa_2.3 | 223 | 415/419 | 19/19 | 1726/1857 | 1670/1855 | 46/52 | 18/19 | 274 | 2.7 |
| aa_2.4 | 129 | 417/419 | 19/19 | 1782/1857 | 1753/1855 | 47/52 | 18/19 | 270 | 3.9 |
| aa_2.5 | 76 | 419/419 | 19/19 | 1813/1857 | 1797/1855 | 49/52 | 18/19 | 261 | 3.9 |
| xalx | 482 | 317/419 | 13/19 | 1463/1857 | 1527/1855 | 44/52 | 18/19 | 257 | 1.7 |
| xalx_t4 | 476 | 317/419 | 13/19 | 1467/1857 | 1531/1855 | 44/52 | 18/19 | 257 | 1.7 |
| axa_150 | 598 | 323/419 | 12/19 | 1458/1857 | 1436/1855 | 34/52 | 12/19 | 269 | 3.0 |
| axa_160 | 910 | 276/419 | 12/19 | 1244/1857 | 1211/1855 | 25/52 | 11/19 | 275 | 3.0 |
| axa_170 | 1287 | 218/419 | 8/19 | 953/1857 | 931/1855 | 22/52 | 9/19 | 274 | 2.3 |
| bvs_0.20+aa_2.3 | 54 | 419/419 | 19/19 | 1830/1857 | 1806/1855 | 51/52 | 18/19 | 325 | 1.9 |
| bvs_0.20+xalx | 351 | 336/419 | 15/19 | 1558/1857 | 1616/1855 | 48/52 | 18/19 | 262 | 1.1 |
| bvs_0.20+axa_160 | 211 | 394/419 | 18/19 | 1713/1857 | 1703/1855 | 45/52 | 16/19 | 241 | 3.3 |
| aa_2.3+xalx | 29 | 419/419 | 19/19 | 1840/1857 | 1833/1855 | 50/52 | 18/19 | 227 | 6.9 |
| aa_2.3+axa_160 | 75 | 419/419 | 19/19 | 1819/1857 | 1796/1855 | 46/52 | 18/19 | 222 | 8.0 |
| xalx+axa_160 | 106 | 405/419 | 18/19 | 1782/1857 | 1785/1855 | 47/52 | 18/19 | 225 | 4.7 |
| bvs_0.20+aa_2.3+xalx | 17 | 419/419 | 19/19 | 1849/1857 | 1841/1855 | 51/52 | 18/19 | 310 | 5.9 |
| bvs_0.20+aa_2.3+axa_160 | 13 | 419/419 | 19/19 | 1852/1857 | 1845/1855 | 51/52 | 18/19 | 309 | 7.7 |
| bvs_0.20+xalx+axa_160 | 57 | 412/419 | 18/19 | 1815/1857 | 1817/1855 | 49/52 | 18/19 | 206 | 5.3 |
| aa_2.3+xalx+axa_160 | 11 | 419/419 | 19/19 | 1851/1857 | 1850/1855 | 50/52 | 18/19 | 143 | 18.2 |
| bvs_0.20+aa_2.3+xalx+axa_160 | 4 | 419/419 | 19/19 | 1855/1857 | 1854/1855 | 51/52 | 18/19 | 122 | 25.0 |

## After local re-optimisation from the recorded points

| term(s) | classes kept | removed: excl. anion-anion | removed: excl. not converged | removed: saddle | removed: > 150 meV | removed: <= 100 meV | removed: stage-1 picks | median E kept (meV) | % kept <= 100 meV |
|---|---|---|---|---|---|---|---|---|---|
| bvs_0.10 | 1522 | 202/419 | 12/19 | 779/1857 | 734/1855 | 20/52 | 6/19 | 271 | 2.1 |
| bvs_0.15 | 2368 | 41/419 | 3/19 | 155/1857 | 132/1855 | 4/52 | 2/19 | 276 | 2.0 |
| bvs_0.20 | 2459 | 22/419 | 2/19 | 88/1857 | 64/1855 | 3/52 | 2/19 | 279 | 2.0 |
| aa_2.3 | 1108 | 335/419 | 19/19 | 1129/1857 | 964/1855 | 33/52 | 12/19 | 274 | 1.7 |
| xalx | 666 | 309/419 | 13/19 | 1358/1857 | 1379/1855 | 37/52 | 18/19 | 256 | 2.3 |
| axa_160 | 1581 | 157/419 | 7/19 | 739/1857 | 719/1855 | 20/52 | 9/19 | 272 | 2.0 |
| bvs_0.20+aa_2.3+xalx+axa_160 | 114 | 415/419 | 19/19 | 1798/1857 | 1766/1855 | 47/52 | 18/19 | 213 | 4.4 |

## Fraction of relaxed structures (not excluded) each term passes, by MACE energy

| term | E <= 75 (n=20) | 75 < E <= 100 (n=32) | 100 < E <= 150 (n=220) | 150 < E <= 300 (n=902) | 300 < E (n=953) |
|---|---|---|---|---|---|
| bvs_0.10 | 0.60 | 0.28 | 0.49 | 0.41 | 0.24 |
| bvs_0.15 | 1.00 | 0.81 | 0.71 | 0.76 | 0.56 |
| bvs_0.20 | 1.00 | 0.97 | 0.89 | 0.91 | 0.76 |
| bvs_0.25 | 1.00 | 0.97 | 0.92 | 0.97 | 0.86 |
| bvs_0.30 | 1.00 | 0.97 | 0.97 | 0.99 | 0.93 |
| aa_2.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| aa_2.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| aa_2.3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| aa_2.4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| aa_2.5 | 1.00 | 0.94 | 0.98 | 0.92 | 0.84 |
| xalx | 0.85 | 0.78 | 0.63 | 0.38 | 0.23 |
| xalx_t4 | 0.85 | 0.78 | 0.60 | 0.38 | 0.22 |
| axa_150 | 0.95 | 0.72 | 0.71 | 0.47 | 0.34 |
| axa_160 | 1.00 | 0.88 | 0.78 | 0.64 | 0.48 |
| axa_170 | 1.00 | 0.97 | 0.84 | 0.78 | 0.65 |

## Anion CN of the relaxed structures (sites), by MACE energy

| E band (meV/atom) | O2 | O3 | O4 | O5 | N3 | N4 | N5 |
|---|---|---|---|---|---|---|---|
| -1000000000..75 | 0 | 89 | 28 | 0 | 8 | 33 | 0 |
| 75..100 | 7 | 151 | 16 | 0 | 5 | 63 | 0 |
| 100..150 | 49 | 850 | 264 | 10 | 94 | 321 | 43 |
| 150..300 | 216 | 3423 | 1327 | 65 | 466 | 1763 | 252 |
| 300..1000000000 | 950 | 2741 | 1319 | 18 | 638 | 2029 | 172 |

## Anion BVS by CN at the recorded points (all zero-error entries)

| anion CN | sites | BVS min | median | max |
|---|---|---|---|---|
| N3 | 3127 | 1.28 | 2.04 | 2.57 |
| N4 | 18942 | 1.66 | 2.66 | 3.41 |
| O2 | 9842 | 0.83 | 1.34 | 1.80 |
| O3 | 8963 | 1.31 | 2.00 | 2.54 |
| O4 | 19304 | 1.55 | 2.40 | 3.30 |

Classes with O at CN 2 in every entry: 1152; MACE median 281 meV/atom (others 276); <= 100 meV: 28 (others 24).

Classes with N at CN 3 in every entry: 473; MACE median 292 meV/atom (others 276); <= 100 meV: 8 (others 44).

## Stage-1 picks

| class | E (meV/atom) | why | bvs_0.20 | aa_2.3 | xalx | axa_160 | relaxed structure |
|---|---|---|---|---|---|---|---|
| Al3O3N__class_34_sg166 | 43.5 | composition #1 | kept | kept | kept | kept | pass |
| Al3O3N__class_354_sg24 | 47.1 | overall | kept | removed | removed | kept | pass |
| Al8O9N2__class_15_sg164 | 61.7 | composition #1 | kept | removed | removed | removed | pass |
| Al4O3N2__class_635_sg59 | 64.7 | composition #1 | recovered | removed | removed | removed | pass |
| Al3O3N__class_131_sg59 | 66.4 | overall | recovered | recovered | removed | kept | xalx |
| Al3O3N__class_16_sg129 | 66.6 | overall | removed | removed | removed | removed | pass |
| Al3O3N__class_294_sg18 | 66.7 | overall | recovered | removed | removed | recovered | pass |
| Al3O3N__class_113_sg51 | 67.3 | overall | recovered | removed | removed | removed | pass |
| Al3O3N__class_118_sg59 | 68.8 | overall | recovered | removed | removed | kept | pass |
| Al3O3N__class_681_sg37 | 71.3 | overall | kept | removed | removed | removed | pass |
| Al3O3N__class_163_sg63 | 71.3 | overall | kept | removed | removed | removed | xalx |
| Al5O6N__class_117_sg12 | 73.4 | composition #1 | recovered | removed | removed | removed | pass |
| Al6O3N4__class_198_sg23 | 101.2 | composition #1 | recovered | recovered | removed | kept | pass |
| Al7O6N3__class_29_sg157 | 120.9 | composition #1 | kept | removed | removed | kept | xalx |
| Al9O6N5__class_16_sg189 | 180.9 | composition #1 | removed | removed | removed | removed | pass |
| Al7O3N5__class_08_sg150 | 191.6 | composition #1 | recovered | recovered | removed | recovered | axa_160 |
| Al5O3N3__class_07_sg157 | 206.0 | composition #1 | recovered | recovered | removed | kept | axa_160 |
| Al8O3N6__class_02_sg187 | 243.9 | composition #1 | recovered | recovered | removed | kept | axa_160 |
| Al10O3N8__class_11_sg121 | 277.6 | composition #1 | recovered | recovered | removed | removed | xalx |

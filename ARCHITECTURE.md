# Architecture

How `outlet-audit` turns a folder of outlet photos into flagged images with reasons.
Code map is in [README.md](README.md#layout); method and numbers in [WRITEUP.md](WRITEUP.md).

The whole system is one batch CLI. No services, no database, no queue: files in, files out,
with a content-hash cache so a rerun only pays for new photos.

---

## 1. End-to-end: `outlet-audit run`

```mermaid
flowchart TB
    CFG["<b>config.yaml</b><br/>model ids · gate thresholds<br/>prompts · harness settings"]

    subgraph SCAN["1 · Scan &nbsp;<code>io.py</code>"]
        DATA[("dataset/outlet_ID/image_*.jpg<br/>159 outlets · 2,042 photos")]
        REC["scan_dataset → ImageRec<br/>sha256 of bytes · dHash · readable?"]
        DATA --> REC
    end

    subgraph EMBED["2 · Represent &nbsp;<code>embed.py</code> + <code>pipeline.py</code>"]
        DD["readable_reps<br/>union-find on sha256 + dHash ≤ 8<br/>→ one representative per copy group"]
        EM["DINOv2-base CLS token<br/>resize 294×224, L2-normalised<br/><i>clip / phash swappable</i>"]
        EC[("cache/MODEL/SHA.npy")]
        DD --> EM
        EM <--> EC
    end

    subgraph CAL["3 · Calibrate once, no labels &nbsp;<code>calibrate.py</code> <code>verify.py</code>"]
        C1["fit_consensus_lr<br/>within-folder vs cross-outlet"]
        C2["fit_geometry_lr<br/>300 SIFT pairs, same trick"]
    end

    subgraph SCORE["4 · Score every folder &nbsp;<code>pipeline.Scorer.score_folder</code>"]
        SC["cosine matrix → median consensus<br/>→ dual gate → geometry → subgroup<br/><i>detail in §2</i>"]
    end

    subgraph EXP["5 · Explain flagged only &nbsp;<code>explain.py</code>"]
        T["CLIP zero-shot<br/>scene + telecom brand tags"]
        O["EasyOCR bn+en<br/>signage text vs 5 peers"]
        R["Explainer.reason<br/>templated, every clause cites a number"]
        T --> R
        O --> R
        OC[("cache/ocr_bn_en/SHA.json")]
        O <--> OC
    end

    subgraph OUT["6 · Write &nbsp;<code>report.py</code>"]
        J["results.json / results.csv<br/>one record per outlet"]
        I["results_images.csv<br/>one row per image"]
        K["calibration.json"]
        H["report.html contact sheet"]
    end

    CFG -.-> SCAN & EMBED & CAL & SCORE & EXP
    REC --> DD
    EM --> CAL
    EM --> SCORE
    CAL --> SCORE
    SCORE --> EXP
    EXP --> OUT
```

Only flagged images are tagged and OCR'd — that is why a cold full run is ~1 h instead of ~20 h.

---

## 2. The decision for one image

`Scorer.score_folder` runs per outlet folder. Everything below is per image inside that folder.

```mermaid
flowchart TB
    START(["image in folder"]) --> RD{readable?}
    RD -- no --> UNR["kind = unreadable<br/>score 0.5, flagged"]
    RD -- yes --> DUP{"copy of another<br/>image here?"}
    DUP -- yes --> D2["kind = duplicate<br/>score 0.3, flagged"]
    DUP -- no --> XO{"same sha256 under<br/>a different outlet?"}
    XO -- yes --> X2["kind = cross_duplicate<br/>score 0.9, flagged"]
    XO -- no --> S

    subgraph S["scored path"]
        CM["s = median cosine to every peer<br/><i>median, not centroid: tolerates<br/>&lt;50% contamination, survives<br/>multi-modal folders</i>"]
        POST["p = LR posterior(s)<br/>calibrated in §3"]
        ZS["z = MAD modified z-score of s"]
        CM --> POST & ZS
        GATE{"candidate?<br/>z &lt; −2.5 <b>OR</b> p ≥ p_abs 0.5"}
        POST --> GATE
        ZS --> GATE
    end

    GATE -- no --> CLEAN["not flagged<br/>keeps its score for ranking"]
    GATE -- yes --> GEO["<b>SIFT + RANSAC</b> vs 3 nearest peers<br/>+ 2 most typical peers<br/><i>peers restricted to images at or above<br/>the support level — else a transplanted<br/>group verifies its own siblings</i>"]
    GEO --> GMUL["p ← posterior(s, extra_log_lr = geometry)"]
    GMUL --> SG{"≥3 mutually similar<br/>images here, and one of<br/>them agrees with the<br/>rest of the folder?"}
    SG -- "yes (anchored)" --> DOWN["p ← p × p_subgroup<br/><i>legit interior/close-up cluster</i>"]
    SG -- "no" --> KEEP["p unchanged<br/><i>coherent but foreign as a whole</i>"]
    DOWN --> FIN
    KEEP --> FIN
    FIN{"p ≥ p_thresh 0.30?"} -- yes --> FLAG["<b>flagged</b> → gets a reason"]
    FIN -- no --> CLEAN
```

Two folder-level escapes, because per-image rules cannot always work:

| case | rule | result |
|---|---|---|
| folder has exactly 2 usable images | no relative gate is possible from one shared similarity | both flagged as `pair` if they fail the absolute floor |
| folder median consensus < `gate.folder_review_floor` 0.45 | folder holds several places; per-image flags unreliable | `review_folder: true`, reviewer looks at the whole folder |

---

## 3. Calibration without labels

The dataset has no ground truth, so both thresholds are learned from the data's own structure:
images inside one folder are *presumed legit*, images from a *different outlet* are
*guaranteed foreign*. Same machinery (`calibrate.fit_lr`) serves both stages.

```mermaid
flowchart LR
    subgraph POS["legit population"]
        P1["consensus of each image<br/>vs its own folder,<br/>z-trimmed to drop the<br/>folder's own outliers"]
    end
    subgraph NEG["foreign population"]
        N1["same statistic vs a<br/>random <b>other</b> outlet"]
    end
    P1 --> KDE["gaussian_kde on each side"]
    N1 --> KDE
    KDE --> LR["log LR = log p_foreign − log p_legit<br/>clipped to ±12"]
    LR --> BAYES["posterior = σ( logit(prior 0.05) + logLR + geomLR )"]
    BAYES --> FLOOR["solve_floor(p_thresh) → absolute similarity floor<br/>solve_floor(prior) → support level for peers/subgroups"]
```

Geometry gets the identical treatment: legit = image vs its own folder's peer set,
foreign = the same image vs another outlet's peer set, statistic = `log1p(max inliers)`,
floored at RANSAC's 4-match minimum so the sparse 0-inlier bin cannot flip sign between runs.

Both log-LRs add, naive-Bayes style — appearance and geometry are treated as independent evidence.

---

## 4. `outlet-audit evaluate` — the validation harness

No labels exist, so labels are manufactured: transplant photos between outlets and check
whether the pipeline finds them.

```mermaid
flowchart LR
    F["real folders"] --> INJ["inject()<br/><b>easy</b>: random donor outlet<br/><b>hard</b>: nearest outlet by mean embedding"]
    INJ --> SCR["same Scorer, z_grid × 3 seeds × 2 modes"]
    SCR --> M["pr_table + average_precision<br/>respecting the relative gate"]
    M --> E["eval_EMBEDDER.json"]
    M -.->|"--write-threshold"| W["set_config_values<br/>writes best-F0.5 z_thresh + p_thresh<br/>back into config.yaml"]
```

Baselines (`--embedder clip|phash`, `--no-geometry`) run through the exact same harness, so the
comparison isolates one variable at a time. The harness scores every original image as legit,
which makes its precision a lower bound — real foreign originals that the pipeline correctly
catches count as errors there.

---

## 5. Data and caching

Everything expensive is keyed by **sha256 of the file bytes**, namespaced by model and
preprocessing. Adding new visits to a folder re-embeds only the new files; folder statistics are
recomputed from cache in seconds.

| cache | holds | written by |
|---|---|---|
| `cache/facebook__dinov2-base_294x224_cls/SHA.npy` | 768-d embedding | `embed.embed_records` |
| `cache/openai__clip-vit-base-patch32_squash/SHA.npy` | CLIP embedding for tags | same |
| `cache/sift_640_2000/SHA.npz` | SIFT keypoints + descriptors | `verify.GeomVerifier.features` |
| `cache/ocr_bn_en/SHA.json` | OCR text + confidences | `explain.OCR.read` |

Module boundaries, one responsibility each:

```mermaid
flowchart TB
    cli["cli.py<br/>run · evaluate"]
    cfgm["config.py"]
    iom["io.py<br/>scan · hash · dedup"]
    embm["embed.py<br/>DINOv2 · CLIP · pHash"]
    scom["scoring.py<br/>cosine · consensus · z · subgroup"]
    calm["calibrate.py<br/>KDE LR → posterior"]
    verm["verify.py<br/>SIFT · RANSAC"]
    pipm["pipeline.py<br/>Scorer: orchestrates a folder"]
    expm["explain.py<br/>tags · OCR · reasons"]
    repm["report.py<br/>JSON · CSV · HTML"]
    evm["evaluate.py<br/>injection harness"]

    cli --> cfgm & iom & embm & pipm & expm & repm & evm
    pipm --> scom & calm & verm & iom
    verm --> calm
    evm --> pipm
    expm --> embm
```

`pipeline.Scorer` is the only place that knows the full decision; `cli.py` and `evaluate.py` both
drive it, which is why harness numbers describe the code that actually ships.

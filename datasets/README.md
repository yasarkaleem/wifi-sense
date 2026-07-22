# datasets

Public WiFi CSI datasets, converted into wifi-sense's canonical CSI frame
schema ([`../docs/csi-frame-schema.md`](../docs/csi-frame-schema.md)) so
`services/replay` can stream them as if they were live, and
`services/pipeline`'s training scripts can train on them.

This directory's contents (other than this README, `download.py`,
`pyproject.toml`, and `tests/`) are gitignored — converted `.npz` files are
regenerated locally, not committed.

## Converting a dataset

```bash
pip install -e ".[dev]"     # numpy (+ pytest for datasets/tests/)

python download.py ut-har --source /path/to/Dataset/Data --out ut-har
python download.py --list   # see what's supported
```

This only *converts* already-downloaded raw data — it does not
auto-download the source archives (see below for why, and for manual
download steps). Output is one `.npz` file per recording session, e.g.
`ut-har/bed_1.npz`, `ut-har/walk_3.npz`, ... — each one streamable
directly:

```bash
cd ../services/replay
python -m replay --dataset ut-har --file ../../datasets/ut-har/bed_1.npz --target localhost:5566
```

## Supported datasets

### UT-HAR — implemented

Human activity recognition: 7 activities (lie down, fall, walk, pick up an
object, run, sit down, stand up), 1 person, 1 room, Intel 5300 NIC (Linux
802.11n CSI Tool), amplitude + phase, 30 subcarriers × 3 antennas.

**Download** (manual — see "Why no automatic download" below):

1. Download the archive (~4GB) from the dataset repository's Google Drive
   link: https://github.com/ermongroup/Wifi_Activity_Recognition
   (see its README's "How to run" step 0).
2. Extract it. `download.py` expects the extracted `Dataset/Data/`
   directory — the one containing `input_*.csv` / `annotation_*.csv` file
   pairs (raw per-packet CSI capture — 1 timestamp column, 90 amplitude
   columns, 90 phase columns — and per-frame activity annotations, one
   pair of files per recording session).
3. `python download.py ut-har --source /path/to/Dataset/Data --out ut-har`

**Placeholder fields**: the raw export has no RSSI or WiFi-channel column,
so converted frames use placeholders (`rssi=0`, `channel=1`) to satisfy
the canonical schema's required fields — these are *not* measured values.
See `download.py`'s `UT_HAR_PLACEHOLDER_*` constants.

**License**: the source repository (code *and* data — there's no separate
data license) is [GPL-3.0](https://github.com/ermongroup/Wifi_Activity_Recognition/blob/master/LICENSE).
Check the repository directly for current terms before redistributing
converted data.

**Citation** — cite the paper if you publish results using this dataset:

```bibtex
@article{yousefi2017survey,
  author  = {Yousefi, Siamak and Narui, Hirokazu and Dayal, Sankalp and Ermon, Stefano and Valaee, Shahrokh},
  title   = {A Survey on Behavior Recognition Using {WiFi} Channel State Information},
  journal = {IEEE Communications Magazine},
  year    = {2017},
  volume  = {55},
  number  = {10},
  pages   = {98--104},
  doi     = {10.1109/MCOM.2017.1700082}
}
```

### Widar 3.0 — stub only

Gesture recognition: 258K gesture instances across 75 "domains"
(room × user × torso location × orientation × receiver), complex-valued
CSI plus derived Doppler Frequency Shift (DFS) and Body-coordinate
Velocity Profile (BVP) features, stored as `.mat` files rather than a flat
per-packet CSV.

`download.py widar3` raises `NotImplementedError` — see its docstring for
exactly what a real implementation would need (a `scipy.io.loadmat`-based
parser, complex amplitude/phase extraction, and domain-tuple-aware session
grouping, all materially different from UT-HAR's simple CSV pair format).

**Download**: dataset access must be requested at
http://tns.thss.tsinghua.edu.cn/widar3.0/ (not a direct public link).

**License**: not publicly stated in the materials found while building
this stub — check the request-access page for current terms before using
it.

**Citation**:

```bibtex
@inproceedings{zheng2019widar3,
  author    = {Zheng, Yue and Zhang, Yi and Qian, Kun and Zhang, Guidong and Liu, Yunhao and Wu, Chenshu and Yang, Zheng},
  title     = {Zero-Effort Cross-Domain Gesture Recognition with {Wi-Fi}},
  booktitle = {Proceedings of the 17th Annual International Conference on Mobile Systems, Applications, and Services (MobiSys)},
  year      = {2019},
  doi       = {10.1145/3307334.3326081}
}
```

## Why no automatic download

Both source archives are large (UT-HAR ~4GB) and hosted in ways that
resist reliable scripted downloading — Google Drive's interactive
"can't scan this file for viruses" confirmation page for large files, and
Widar 3.0's access-request gate. Rather than depend on a fragile
`gdown`-style workaround we can't fully guarantee works (link rot, quota
limits, and confirmation-token changes are common failure modes for this
exact pattern), `download.py` focuses on the part that's fully ours to get
right — conversion — and documents the one-time manual download step
above. If you script this yourself, please still respect the dataset's
license and cite the paper.

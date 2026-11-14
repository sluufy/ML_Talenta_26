# A Machine Learning-Based Decision Support System for Nine-Box Talent Placement (Boxes 8-9)

Sohib Pahlupi — Magister Sistem Informasi, Universitas Nusa Mandiri

Kode, data, hasil, dan naskah penelitian yang menguji apakah machine learning dapat menentukan pegawai
yang layak berada di kotak talenta 8-9 (layak promosi) tanpa melanggar aturan manajemen talenta ASN
(PermenPANRB 3/2020 jo. 20/2025; PermenPANRB 19/2025). Sumbu kinerja tetap dari penilaian kinerja (SKP);
ML (CatBoost buta-usia) mengalibrasi sumbu potensi (Tahap 1) dan memprioritaskan kandidat di dalam kotak 8-9
(Tahap 2), dilengkapi audit kebocoran, SHAP, audit keadilan, dan rujukan konformal Mondrian ke komite talenta.

Penomoran kotak mengikuti Lampiran PermenPANRB 20/2025: kotak 9 = kinerja di atas ekspektasi & potensial tinggi,
kotak 8 = sesuai ekspektasi & potensial tinggi, kotak 7 = di atas ekspektasi & potensial menengah
(kotak 7-9 = kelompok rencana suksesi instansi; kotak 9 = kelompok rencana suksesi nasional).

## Struktur folder

```
code/          skrip eksperimen + reproduce.ipynb
data/raw/      dataset (lihat bagian Data)
results/       tables/ (CSV hasil) dan figures/ (PNG 300 dpi)
manuscripts/   Manuskrip_Q1_EN/ID (format jurnal) dan Paper_UTS_AdvML_EN/ID (template UTS), .docx + .pdf
submission/    berkas pengiriman jurnal: naskah anonim, title page, cover letter, graphical abstract
references/    reference_checklist.csv (verifikasi Scopus), scopus_export.csv
requirements.txt
LICENSE        MIT (kode); data mengikuti lisensi sumbernya
CITATION.cff   cara mengutip repositori ini
```

## Data

| File | Sumber | Keterangan |
|---|---|---|
| `data/raw/promotion_train.csv` | Kaggle, *HR Analytics: Employee Promotion Data* (https://www.kaggle.com/datasets/arashnic/hr-ana), file `train.csv` | 54.808 pegawai, target `is_promoted` (8,52%) |
| `data/raw/openml_43897_ibm-employee-performance.csv` | OpenML ID 43897 (IBM HR Analytics, ODbL) | 1.470 pegawai (fiktif), target `PerformanceRating` |

Dataset Kaggle tidak diunggah ke repositori publik (lihat `.gitignore`); unduh ulang dari sumbernya.

## Menjalankan ulang

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd code
python fetch_data.py            # unduh data IBM (OpenML)
python ibm_leakage.py           # Eksperimen 1: audit kebocoran IBM
python run_benchmark.py         # Eksperimen 2: 7 model x 3 tugas x 10 ulangan + uji Friedman/Wilcoxon-Holm
python run_trust.py             # Eksperimen 3: kalibrasi, SHAP, keadilan, konformal, 9-box, analisis error
python run_conformal_repeats.py # Eksperimen 4: cakupan konformal pada 10 ulangan
python plot_conformal_repeats.py # gambar cakupan konformal (fig_conformal_coverage_*.png)
python run_fairness_repeats.py  # Eksperimen 5: audit & mitigasi keadilan pada 10 ulangan
python run_talent_box.py        # Eksperimen 6 (utama): penempatan kotak 9-box & prioritas kotak 8-9
python run_box_extras.py        # Eksperimen 7: model buta-usia & rujukan konformal di kotak 8-9
python run_robustness.py         # Eksperimen 8: uji ketahanan (definisi alternatif) & interval kepercayaan bootstrap
python make_pipeline_figure.py  # Gambar 1
python plot_figures.py          # gambar hasil versi EN dan ID
python plot_talent_box.py       # gambar kotak sembilan & kurva prioritas (EN dan ID)
python make_graphical_abstract.py
```

Atau buka `code/reproduce.ipynb` (set `RUN_EXPERIMENTS = True` untuk menjalankan ulang semuanya, ±30 menit di CPU 16 inti).
Seluruh pembagian data memakai *seed* tetap sehingga hasil dapat direproduksi.

## Pengujian

```
python -m unittest discover -s tests -v   # 8 uji unit, < 5 detik, tanpa data asli
python tests/smoke_pipeline.py             # uji end-to-end pada data sintetis (±1 menit)
```

Uji unit (`tests/test_core.py`) memeriksa jaminan metodologis yang dipakai naskah:
- penomoran kotak di semua skrip sama dengan Lampiran PermenPANRB 20/2025 (kotak 7-9 = suksesi instansi);
- partisi *tuning* 20% tidak pernah dipakai untuk evaluasi; pembagian 60/10/10/20 saling lepas, terstratifikasi, dan dapat diulang;
- gender dan `is_promoted` tidak pernah menjadi fitur; sumbu potensi tidak memakai nilai kinerja;
- prediksi konformal Mondrian menjamin cakupan di setiap kelas, sedangkan cakupan marginal gagal pada kelas langka;
- perhitungan ECE.

`tests/smoke_pipeline.py` membangkitkan data sintetis (`code/make_smoke_data.py`) dan menjalankan eksperimen utama
(`run_talent_box.py`, `run_box_extras.py`) ke direktori sementara. Data sintetis tidak pernah dipakai untuk hasil yang dilaporkan.

## Gambar yang dipakai di naskah

| Gambar | File |
|---|---|
| Pipeline | `fig_pipeline_en.png`, `fig_pipeline_id.png` |
| Konvergensi | `conv_tale_en.png`, `conv_tale_id.png` |
| Confusion matrix | `cm_tale_en.png`, `cm_tale_id.png` |
| Kalibrasi | `fig_calibration_talent.png` |
| SHAP | `fig_shap_beeswarm_talent.png` |
| Konformal | `conf_tale_en.png`, `conf_tale_id.png` |
| 9-box | `ninebox_en.png`, `ninebox_id.png` |
| Kotak sembilan (konvensional vs ML) | `box_en.png`, `box_id.png` |
| Kurva prioritas kotak 8-9 | `prio_en.png`, `prio_id.png` |
| Graphical abstract | `graphical_abstract.png` |

## Lingkungan

Windows 11, AMD Ryzen AI Max+ 395 (16 inti/32 thread), RAM 59,6 GB, tanpa GPU.
Python 3.12.10, scikit-learn 1.9.1, CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, SHAP 0.54.1, Fairlearn 0.15.0.

# Analisis Tabel Input-Output di Indonesia dengan memasukkan sektor hijau

**Penulis:** Muhammad Fawdy Renardi Wahyu  
**Deskripsi:**  
Repositori ini berisi skrip Python untuk melakukan analisis Tabel Input-Output (IO) di Indonesia, yang mencakup pembersihan data, analisis linkage, multipliers, dan simulasi skenario pertumbuhan sektor. Skrip ini juga mendukung ekspansi untuk sektor hijau (Green Industries) dan integrasi data satelit seperti tenaga kerja, nilai tambah, dan kompensasi tenaga kerja sesuai dokumen "HOW TO MEASURE AND MODEL SOCIAL AND EMPLOYMENT OUTCOMES OF CLIMATE AND SUSTAINABLE DEVELOPMENT POLICIES" yang diterbitkan oleh International Labour Organization (ILO) dan ditulis oleh Green Jobs Assessment Institutions Network (GAIN) tahun 2017.

## Fitur Utama
- **Cleaning Matrix:** Membersihkan data IO dari tabel mentah untuk menghasilkan matriks teknologi, permintaan akhir, nilai tambah, dan variabel relevan lainnya.
- **IO Analysis:** Menghitung backward dan forward linkage, multiplier output, serta efek pertama dan dampak lanjutan dari perubahan permintaan akhir atau nilai tambah.
- **Update IO Table:** Memperbarui tabel IO berdasarkan tingkat pertumbuhan PDB dan sektor individu menggunakan algoritma Iterative Proportional Fitting (IPF).
- **Green Industries Expansion:** Menambahkan sektor hijau ke tabel IO, menghitung multipliers tenaga kerja, nilai tambah, dan pendapatan, serta melakukan simulasi pertumbuhan untuk sektor hijau.

## Struktur Repositori
- **`io_analysis_module.py`**: Berisi modul dengan fungsi-fungsi untuk pembersihan data IO, analisis, pembaruan tabel, dan ekspansi sektor hijau.
- **`run_io_analysis.py`**: Skrip utama untuk menjalankan modul dengan data IO mentah dan menghasilkan hasil analisis.
- **Contoh Data**: Template data IO mentah dengan nama `Tabel Input Ouput 2016.xlsx`.
- **Dokumentasi**: Penjelasan singkat tentang cara menggunakan skrip dan contoh hasil analisis.

## Teknologi yang Digunakan
- **Python**: Versi 3.8 atau lebih baru.
- **Pustaka Python**:
  - `pandas` untuk manipulasi data.
  - `numpy` untuk komputasi numerik.
  - `ipfn` untuk algoritma Iterative Proportional Fitting (IPF).

## Cara Menggunakan
1. Clone repositori ini: `git clone https://github.com/fawdywahyu18/green_input_output.git`
2. jalankan di terminal : `python run_io_analysis.py`



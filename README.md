### Background Remover

Alat penghapus latar belakang gambar berbasis AI (lokal/offline) menggunakan `rembg` dan `U²‑Net`/`BiRefNet`. Pilih sesuai kebutuhanmu.

---

## 📁 Perbandingan Singkat

| **Aspek**            | **remove.py** (Versi Dasar)                           | **remover.py** (Versi Pro)                                  |
|----------------------|-------------------------------------------------------|-------------------------------------------------------------|
| **UI Framework**     | Tkinter standar (tampilan klasik)                     | CustomTkinter (dark mode) + fallback Tkinter                |
| **Model AI**         | 4 model (`u2net`, `u2netp`, `u2net_human_seg`, `isnet`) | 10 model termasuk **BiRefNet**, ISNet, U²‑Net, Silueta      |
| **Proses**           | Manual: pilih file → klik "Mulai Jalankan"            | Otomatis: pilih file → langsung proses                      |
| **Peningkatan Tepi** | Tidak ada                                             | ✅ Edge refinement (rambut/kaca) + guided filter            |
| **Enhance Otomatis** | Tidak ada                                             | ✅ Sharpness, kontras, denoise (opsional)                   |
| **Logging Sistem**   | Tidak ada                                             | ✅ Log real-time di konsol & panel UI                       |
| **Nama Output**      | `nama_file_nobg.png`                                  | `nama_file_remover.png`                                     |
| **Cache Model**      | Default `rembg` (~/.u2net)                            | Folder `model/` di samping script                           |

---

## Fitur Utama `remover.py` (Versi Pro)

- **Koleksi Model Canggih**  
  BiRefNet (General, Portrait, High-Res, Massive, DIS) – hasil lebih presisi untuk rambut, objek transparan, dan gambar blur.
- **Proses Sekali Klik**  
  Begitu model AI siap (indikator hijau "Ready"), pilih gambar dan proses langsung berjalan.
- **Perbaikan Tepi Otomatis**  
  Menggunakan alpha matting dan guided filter untuk hasil potongan yang halus.
- **Tampilan Modern**  
  Dark mode dengan progress bar, scrollable file list, dan system log.
- **Penanganan Error Andal**  
  Multithreading, fallback UI, dan log lengkap mencegah crash.

---

## Cara Menjalankan

1. **Install dependensi** (wajib untuk `remover.py`):
   ```bash
   pip install rembg pillow opencv-python scipy scikit-image customtkinter
   ````
2. **Jalankan script**:

  ```bash
  python remover.py      # versi pro (lebih berat)
  # atau
  python remove.py       # versi dasar (lebih ringan)
  ```
3. **Pilih model AI** (dropdown) dan tunggu hingga status "Ready".
4. **Klik "Select Images"**, pilih satu/lebih gambar.
5. Hasil otomatis tersimpan di folder `Desktop` (atau folder pilihan).

## Catatan:
- Model diunduh otomatis saat pertama kali dipilih (ukuran 40–176 MB).
- Proses berjalan **sepenuhnya offline** – tidak mengirim data ke server.
- `remover.py` akan otomatis menggunakan tampilan Tkinter biasa jika `customtkinter` tidak terpasang.
- *Masih dalam tahap pengujian untuk `remover.py`!*

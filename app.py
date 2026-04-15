"""
Background Remover Pro - Penghapus Latar Belakang Gambar Otomatis
Dibangun dengan rembg (U²-Net) dan Tkinter GUI.
"""

import os
import threading
from tkinter import Tk, Button, Label, Frame, filedialog, messagebox, StringVar, ttk
from tkinter.ttk import Progressbar, Combobox
from PIL import Image, ImageTk
import io

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


class BackgroundRemoverApp:
    """Aplikasi GUI penghapus latar belakang modern."""

    def __init__(self, root):
        self.root = root
        self.root.title("Background Remover Pro - AI Powered")
        self.root.geometry("650x500")
        self.root.resizable(False, False)

        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        self.processing = False
        self.selected_files = []
        self.output_dir = StringVar(value=os.path.expanduser("~/Desktop"))
        self.session = None
        self.model_name = StringVar(value="u2net")
        self._create_widgets()

        if not REMBG_AVAILABLE:
            self._show_dependency_warning()

        threading.Thread(target=self._load_model, daemon=True).start()

    def _create_widgets(self):
        """Membangun elemen antarmuka."""
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        title = Label(main_frame, text="Background Remover Pro",
                      font=("Segoe UI", 18, "bold"), fg="#2c3e50")
        title.pack(pady=(0, 10))

        desc = Label(main_frame,
                     text="Hapus latar belakang gambar dengan AI. Mendukung Format: JPG, PNG, WEBP, BMP, TIFF.",
                     font=("Segoe UI", 10), fg="#7f8c8d")
        desc.pack(pady=(0, 20))

        model_frame = Frame(main_frame)
        model_frame.pack(fill="x", pady=5)
        Label(model_frame, text="Model:", font=("Segoe UI", 10)).pack(side="left", padx=(0, 10))
        model_combo = Combobox(model_frame, textvariable=self.model_name,
                               values=["u2net", "u2netp", "u2net_human_seg", "isnet-general-use"],
                               state="readonly", width=20)
        model_combo.pack(side="left")
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        file_frame = Frame(main_frame)
        file_frame.pack(fill="x", pady=10)
        Button(file_frame, text="📂 Pilih Gambar", command=self.select_files,
               font=("Segoe UI", 11), bg="#3498db", fg="white",
               activebackground="#2980b9", padx=15, pady=5).pack(side="left")
        self.file_count_label = Label(file_frame, text="0 file dipilih",
                                      font=("Segoe UI", 10), fg="#2c3e50")
        self.file_count_label.pack(side="left", padx=20)

        out_frame = Frame(main_frame)
        out_frame.pack(fill="x", pady=5)
        Label(out_frame, text="Simpan ke:", font=("Segoe UI", 10)).pack(side="left", padx=(0, 10))
        out_entry = ttk.Entry(out_frame, textvariable=self.output_dir, width=40)
        out_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        Button(out_frame, text="Jelajahi", command=self.choose_output_dir,
               font=("Segoe UI", 9), bg="#95a5a6", fg="white").pack(side="left")

        self.progress = Progressbar(main_frame, orient="horizontal", length=500, mode="determinate")
        self.progress.pack(pady=15)
        self.status_label = Label(main_frame, text="Siap. Model sedang dimuat...",
                                  font=("Segoe UI", 9), fg="#7f8c8d")
        self.status_label.pack(pady=5)
        self.process_btn = Button(main_frame, text="Sedang menjalankan...", command=self.start_processing,
                                  font=("Segoe UI", 12, "bold"), bg="#2ecc71", fg="white",
                                  activebackground="#27ae60", padx=20, pady=8, state="disabled")
        self.process_btn.pack(pady=15)

        footer = Label(main_frame, text="Dibangun dengan rembg + U²‑Net | © 2025",
                       font=("Segoe UI", 8), fg="#bdc3c7")
        footer.pack(side="bottom", pady=(10, 0))

    def _show_dependency_warning(self):
        """Tampilkan peringatan jika rembg belum terinstall."""
        messagebox.showwarning(
            "Dependensi Hilang",
            "Library 'rembg' tidak ditemukan.\n\n"
            "Silakan install terlebih dahulu dengan perintah:\n"
            "pip install rembg[gpu]    (untuk GPU)\n"
            "atau\n"
            "pip install rembg         (untuk CPU)\n\n"
            "Setelah itu jalankan ulang aplikasi."
        )
        self.status_label.config(text="❌ rembg belum terpasang! Install terlebih dahulu, lalu restart ulang.")
        self.process_btn.config(state="disabled")

    def _load_model(self):
        """Muat model AI di background thread."""
        if not REMBG_AVAILABLE:
            return
        try:
            self.status_label.config(text="Memuat model AI...")
            self.session = new_session(self.model_name.get())
            self.status_label.config(text="✅ Model AI siap digunakan.")
            self.process_btn.config(state="normal")
        except Exception as e:
            self.status_label.config(text=f"❌ Gagal memuat model: {str(e)[:50]}")
            messagebox.showerror("Error Model", f"Gagal memuat model:\n{e}")

    def _on_model_change(self, event=None):
        """Ganti model AI saat dipilih."""
        if not self.processing:
            self.process_btn.config(state="disabled")
            threading.Thread(target=self._load_model, daemon=True).start()

    def select_files(self):
        """Buka dialog file dengan kemampuan scroll (multi-select)."""
        filetypes = [
            ("Semua Gambar", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("WEBP", "*.webp"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff *.tif"),
        ]
        files = filedialog.askopenfilenames(
            title="Pilih Gambar untuk Dihapus Latarnya",
            filetypes=filetypes
        )
        if files:
            self.selected_files = list(files)
            self.file_count_label.config(text=f"{len(self.selected_files)} file dipilih")
            # Tampilkan nama file pertama sebagai contoh
            if len(files) == 1:
                self.status_label.config(text=f"📄 {os.path.basename(files[0])}")
            else:
                self.status_label.config(text=f"📁 {len(files)} file siap diproses.")

    def choose_output_dir(self):
        """Pilih folder output."""
        folder = filedialog.askdirectory(title="Pilih Folder Penyimpanan")
        if folder:
            self.output_dir.set(folder)

    def start_processing(self):
        """Mulai proses penghapusan latar di thread terpisah."""
        if not self.selected_files:
            messagebox.showinfo("Info", "Pilih minimal satu gambar terlebih dahulu!")
            return
        if not os.path.isdir(self.output_dir.get()):
            messagebox.showerror("Error", "Folder penyimpanan tidak valid!")
            return
        if self.processing:
            return

        self.processing = True
        self.process_btn.config(state="disabled", text="Sedang Memproses...")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.selected_files)

        threading.Thread(target=self._process_images, daemon=True).start()

    def _process_images(self):
        """Proses semua gambar (dipanggil dari thread)."""
        total = len(self.selected_files)
        success = 0
        errors = []

        for idx, filepath in enumerate(self.selected_files):
            try:
                self._update_progress(idx + 1, f"Memproses {os.path.basename(filepath)}...")

                with open(filepath, "rb") as f:
                    input_data = f.read()

                output_data = remove(input_data, session=self.session)

                base, ext = os.path.splitext(os.path.basename(filepath))
                out_filename = f"{base}_nobg.png"
                out_path = os.path.join(self.output_dir.get(), out_filename)

                with open(out_path, "wb") as f:
                    f.write(output_data)

                success += 1

            except Exception as e:
                errors.append(f"{os.path.basename(filepath)}: {str(e)}")

        self.root.after(0, self._processing_done, success, total, errors)

    def _update_progress(self, value, status_text):
        """Perbarui progress bar dan status dari thread."""
        self.root.after(0, lambda: self.progress.config(value=value))
        self.root.after(0, lambda: self.status_label.config(text=status_text))

    def _processing_done(self, success, total, errors):
        """Dipanggil setelah semua gambar selesai diproses."""
        self.processing = False
        self.process_btn.config(state="normal", text="Mulai Jalankan")
        self.progress["value"] = 0

        if errors:
            error_msg = "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... dan {len(errors)-5} error lainnya."
            messagebox.showerror("Beberapa File Gagal",
                                 f"Berhasil: {success}/{total}\n\nError:\n{error_msg}")
        else:
            messagebox.showinfo("Sukses!",
                                f"Semua {total} gambar berhasil diproses.\n\n"
                                f"Hasil disimpan di:\n{self.output_dir.get()}")

        self.status_label.config(text=f"✅ Selesai. {success} dari {total} berhasil.")
        self.file_count_label.config(text="0 file dipilih")
        self.selected_files = []


def main():
    root = Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

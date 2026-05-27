"""
Streamlit Frontend — Chatbot Pencarian File

UI chatbot yang terhubung ke FastAPI backend.
Fitur:
  - Chat interface untuk pencarian file
  - Sidebar untuk konfigurasi & indexing
  - Tampilan file yang ditemukan dengan detail
"""

import streamlit as st
import requests
import json
import time
import os
import re
from datetime import datetime, timezone, timedelta

# =====================================================================
# KONFIGURASI
# =====================================================================
API_BASE_URL = "http://localhost:8001/api"

st.set_page_config(
    page_title="Chatbot Pencarian File",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_file_icon(file_type: str) -> str:
    icons = {
        ".pdf": "📕", ".docx": "📘", ".xlsx": "📗",
        ".csv": "📊", ".pptx": "📙", ".txt": "📄",
    }
    return icons.get(file_type, "📁")


def api_request(
    endpoint: str,
    method: str = "GET",
    data: dict = None,
    timeout_seconds: int | None = None,
    show_error: bool = True,
) -> dict | None:
    """Helper untuk memanggil API backend."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        timeout_value = timeout_seconds or (300 if method == "POST" else 90)
        if method == "POST":
            resp = requests.post(url, json=data, timeout=timeout_value)
        else:
            resp = requests.get(url, timeout=timeout_value)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        if show_error:
            st.warning("⏳ Backend masih sibuk memproses. Coba tunggu sebentar lalu refresh status.")
        return None
    except requests.ConnectionError:
        if show_error:
            st.error("❌ Tidak dapat terhubung ke backend. Pastikan server berjalan di port 8001.")
        return None
    except requests.HTTPError as e:
        if show_error:
            st.error(f"❌ Error dari server: {e}")
        return None
    except Exception as e:
        if show_error:
            st.error(f"❌ Error: {e}")
        return None


def parse_list_input(raw: str) -> list[str]:
    """Parse input list dari newline/koma/titik-koma menjadi list bersih."""
    if not raw or not raw.strip():
        return []

    parts = re.split(r"[\n,;]+", raw)
    cleaned = [p.strip() for p in parts if p and p.strip()]

    # Deduplicate sambil menjaga urutan
    seen = set()
    result = []
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def format_index_root_label(root: str) -> str:
    """Ubah root source menjadi label yang lebih ramah di UI."""
    if not root:
        return "Unknown"

    value = root.strip().lower()
    if value in {"gdrive", "gdrive:"}:
        return "Google Drive (OAuth)"
    return root


def format_timestamp_wib(value: str | None) -> str | None:
    """Format timestamp ISO ke tampilan WIB yang mudah dibaca."""
    if not value:
        return None

    try:
        raw = value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)

        # Jika timestamp tanpa timezone, anggap UTC dari backend.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        wib = timezone(timedelta(hours=7))
        local_dt = dt.astimezone(wib)
        return local_dt.strftime("%d-%m-%Y %H:%M WIB")
    except Exception:
        return value


def render_scrollable_list(title: str, items: list[str], empty_text: str, height_px: int = 240) -> None:
    """Render daftar di dalam box scrollable agar semua item tetap bisa dilihat."""
    if title:
        st.markdown(f"**{title}**")
    if not items:
        st.caption(empty_text)
        return

    rows = "".join(f"<li style='margin-bottom: 0.25rem;'>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div style="
            max-height: {height_px}px;
            overflow-y: auto;
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 0.5rem;
            padding: 0.75rem 0.9rem;
            background: transparent;
            color: inherit;
            box-shadow: none;
        ">
            <ol style="margin: 0; padding-left: 1.1rem; color: inherit;">
                {rows}
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# SIDEBAR — Konfigurasi & Indexing
# =====================================================================
with st.sidebar:
    st.title("⚙️ Konfigurasi")

    if "watch_config_loaded" not in st.session_state:
        watch_config = api_request("index/watch-config", show_error=False) or {}
        st.session_state["watch_local_dirs"] = "\n".join(watch_config.get("local_directories", []))
        st.session_state["watch_exclude_dirs"] = ", ".join(watch_config.get("exclude_directories", []))
        gdrive_ids = watch_config.get("gdrive_folder_ids", [])
        st.session_state["watch_gdrive_folder_id"] = gdrive_ids[0] if gdrive_ids else ""
        st.session_state["watch_gdrive_monitor_all"] = bool(watch_config.get("gdrive_monitor_all", False))
        st.session_state["watch_config_updated_at"] = watch_config.get("updated_at")
        st.session_state["watch_local_last_started"] = watch_config.get("last_local_index_started_at")
        st.session_state["watch_gdrive_last_started"] = watch_config.get("last_gdrive_index_started_at")
        st.session_state["watch_local_last_check_at"] = watch_config.get("last_local_check_at")
        st.session_state["watch_local_last_check_has_changes"] = watch_config.get("last_local_check_has_changes")
        st.session_state["watch_local_last_check_stats"] = watch_config.get("last_local_check_stats", {})
        st.session_state["watch_gdrive_last_check_at"] = watch_config.get("last_gdrive_check_at")
        st.session_state["watch_gdrive_last_check_has_changes"] = watch_config.get("last_gdrive_check_has_changes")
        st.session_state["watch_gdrive_last_check_stats"] = watch_config.get("last_gdrive_check_stats", {})
        st.session_state["watch_config_loaded"] = True

    if st.session_state.get("watch_config_updated_at"):
        watch_updated = format_timestamp_wib(st.session_state.get("watch_config_updated_at"))
        st.caption(f"Watch config updated: {watch_updated}")

    local_check_at = format_timestamp_wib(st.session_state.get("watch_local_last_check_at"))
    if local_check_at:
        local_changed = st.session_state.get("watch_local_last_check_has_changes")
        local_stats = st.session_state.get("watch_local_last_check_stats", {}) or {}
        local_delta = int(local_stats.get("new", 0)) + int(local_stats.get("updated", 0)) + int(local_stats.get("deleted", 0))
        local_status = "Ada perubahan" if local_changed else "Tidak ada perubahan"
        st.caption(f"Cek terakhir local: {local_check_at} ({local_status}, delta={local_delta})")
    else:
        st.caption("Cek terakhir local: belum ada data")

    gdrive_check_at = format_timestamp_wib(st.session_state.get("watch_gdrive_last_check_at"))
    if gdrive_check_at:
        gdrive_changed = st.session_state.get("watch_gdrive_last_check_has_changes")
        gdrive_stats = st.session_state.get("watch_gdrive_last_check_stats", {}) or {}
        gdrive_delta = int(gdrive_stats.get("new", 0)) + int(gdrive_stats.get("updated", 0)) + int(gdrive_stats.get("deleted", 0))
        gdrive_status = "Ada perubahan" if gdrive_changed else "Tidak ada perubahan"
        st.caption(f"Cek terakhir GDrive: {gdrive_check_at} ({gdrive_status}, delta={gdrive_delta})")
    else:
        st.caption("Cek terakhir GDrive: belum ada data")

    # --- Statistik ---
    st.subheader("📊 Statistik Database")
    if st.button("🔄 Refresh Stats"):
        st.session_state["refresh_stats"] = True

    stats = api_request("stats")
    if stats:
        col1, col2 = st.columns(2)
        col1.metric("Total File", stats["total_files"])
        col2.metric("Total Chunk", stats["total_chunks"])

        src1, src2 = st.columns(2)
        src1.metric("Local Files", stats.get("local_files", 0))
        src2.metric("GDrive Files", stats.get("gdrive_files", 0))

        local_dirs = stats.get("indexed_local_directories", [])
        local_dir_details = stats.get("indexed_local_directory_details", []) or []
        gdrive_roots = stats.get("indexed_gdrive_roots", [])
        gdrive_files_preview = stats.get("indexed_gdrive_files", [])
        gdrive_dir_details = stats.get("indexed_gdrive_directory_details", []) or []

        if local_dirs or gdrive_roots:
            with st.expander("📁 Direktori ter-index"):
                if local_dir_details:
                    st.markdown("**Local (ringkasan per direktori)**")
                    local_rows = []
                    for item in local_dir_details:
                        last_indexed = format_timestamp_wib(item.get("last_indexed"))
                        local_rows.append(
                            {
                                "Direktori": str(item.get("directory_path") or "-"),
                                "Total File": int(item.get("total_files") or 0),
                                "Embedded": int(item.get("embedded_files") or 0),
                                "Metadata Only": int(item.get("metadata_only_files") or 0),
                                "Last Indexed": last_indexed or "-",
                            }
                        )
                    st.dataframe(local_rows, use_container_width=True, hide_index=True)
                else:
                    render_scrollable_list(
                        "Local",
                        [str(d) for d in local_dirs],
                        "Belum ada index local",
                        height_px=260,
                    )

                st.write("")

                if gdrive_dir_details:
                    st.markdown("**Google Drive (ringkasan per direktori)**")
                    gdrive_rows = []
                    for item in gdrive_dir_details:
                        last_indexed = format_timestamp_wib(item.get("last_indexed"))
                        gdrive_rows.append(
                            {
                                "Direktori": str(item.get("directory_path") or "-"),
                                "Total File": int(item.get("total_files") or 0),
                                "Embedded": int(item.get("embedded_files") or 0),
                                "Metadata Only": int(item.get("metadata_only_files") or 0),
                                "Last Indexed": last_indexed or "-",
                            }
                        )
                    st.dataframe(gdrive_rows, use_container_width=True, hide_index=True)
                else:
                    render_scrollable_list(
                        "Google Drive",
                        [format_index_root_label(d) for d in gdrive_roots],
                        "Belum ada index Google Drive",
                        height_px=180,
                    )

                if gdrive_files_preview:
                    st.caption(f"File yang sudah ter-index ({len(gdrive_files_preview)} file):")
                    render_scrollable_list(
                        "",
                        [str(file_name) for file_name in gdrive_files_preview],
                        "",
                        height_px=220,
                    )

    st.divider()

    # --- Indexing ---
    if "indexing_active" not in st.session_state:
        st.session_state["indexing_active"] = False
    if "index_status_snapshot" not in st.session_state:
        st.session_state["index_status_snapshot"] = {}

    # Auto-resume monitoring jika backend sedang indexing (meski UI baru di-refresh).
    status_snapshot = api_request("index/status", timeout_seconds=30, show_error=False)
    if status_snapshot:
        st.session_state["index_status_snapshot"] = status_snapshot
        if bool(status_snapshot.get("running")):
            st.session_state["indexing_active"] = True

    st.subheader("📂 Indexing File")

    crawl_dirs = st.text_area(
        "Direktori yang akan di-crawl",
        key="watch_local_dirs",
        placeholder="D:/clone smt\nD:/daftar kuliah",
        help="Bisa dipisah baris baru, koma, atau titik-koma",
    )

    exclude_dirs = st.text_input(
        "🚫 Folder yang di-skip (pisah koma)",
        key="watch_exclude_dirs",
        placeholder="SOFTWARE, node_modules, .git",
        help="Nama folder yang akan dilewati saat crawling (tidak rekursif ke dalamnya)",
    )

    st.caption("Konfigurasi monitoring akan tersimpan otomatis saat menjalankan indexing.")

    col_idx1, col_idx2 = st.columns(2)

    with col_idx1:
        if st.button("🔄 Full Index", use_container_width=True):
            if crawl_dirs.strip():
                dirs = parse_list_input(crawl_dirs)
                excl = parse_list_input(exclude_dirs)

                valid_dirs = [d for d in dirs if os.path.exists(d)]
                invalid_dirs = [d for d in dirs if not os.path.exists(d)]
                if invalid_dirs:
                    st.warning(f"Path tidak ditemukan (diabaikan): {', '.join(invalid_dirs)}")
                if not valid_dirs:
                    st.error("Semua path direktori tidak valid.")
                    st.stop()

                result = api_request("index", method="POST", data={"directories": valid_dirs, "exclude_dirs": excl})
                if result:
                    if result.get("status") == "already_running":
                        st.warning("⏳ Indexing sedang berjalan. Tunggu hingga selesai.")
                        st.session_state["indexing_active"] = True
                        st.rerun()
                    else:
                        saved = api_request(
                            "index/watch-config",
                            method="POST",
                            data={"local_directories": valid_dirs, "exclude_directories": excl},
                            show_error=False,
                        )
                        if saved:
                            st.session_state["watch_config_updated_at"] = saved.get("updated_at")
                        st.session_state["indexing_active"] = True
                        st.rerun()
            else:
                st.warning("Masukkan minimal satu direktori.")

    with col_idx2:
        if st.button("⚡ Incremental", use_container_width=True):
            if crawl_dirs.strip():
                dirs = parse_list_input(crawl_dirs)
                excl = parse_list_input(exclude_dirs)

                valid_dirs = [d for d in dirs if os.path.exists(d)]
                invalid_dirs = [d for d in dirs if not os.path.exists(d)]
                if invalid_dirs:
                    st.warning(f"Path tidak ditemukan (diabaikan): {', '.join(invalid_dirs)}")
                if not valid_dirs:
                    st.error("Semua path direktori tidak valid.")
                    st.stop()

                result = api_request(
                    "index/incremental", method="POST", data={"directories": valid_dirs, "exclude_dirs": excl}
                )
                if result:
                    if result.get("status") == "already_running":
                        st.warning("⏳ Indexing sedang berjalan. Tunggu hingga selesai.")
                        st.session_state["indexing_active"] = True
                        st.rerun()
                    else:
                        saved = api_request(
                            "index/watch-config",
                            method="POST",
                            data={"local_directories": valid_dirs, "exclude_directories": excl},
                            show_error=False,
                        )
                        if saved:
                            st.session_state["watch_config_updated_at"] = saved.get("updated_at")
                        st.session_state["indexing_active"] = True
                        st.rerun()
            else:
                st.warning("Masukkan minimal satu direktori.")

    # --- Progress Polling ---
    if st.session_state.get("indexing_active"):
        progress_placeholder = st.empty()
        bar_placeholder = st.empty()

        while True:
            prog = api_request("index/status", timeout_seconds=120, show_error=False)
            if prog is None:
                last_prog = st.session_state.get("index_status_snapshot", {}) or {}
                last_phase = last_prog.get("phase", "unknown")
                last_current = last_prog.get("current_file") or "-"
                progress_placeholder.warning(
                    f"⏳ Status belum terbaca (backend sibuk), mencoba lagi... "
                    f"(last: phase={last_phase}, file={last_current})"
                )
                time.sleep(2)
                continue

            st.session_state["index_status_snapshot"] = prog

            phase = prog.get("phase", "idle")
            total = prog.get("total_files", 0)
            done = prog.get("processed_files", 0)
            current = prog.get("current_file", "")
            running = prog.get("running", False)

            if phase == "error":
                st.error(f"❌ Indexing error: {prog.get('error', 'Unknown')}")
                st.session_state["indexing_active"] = False
                break

            if phase == "done" or (not running and phase != "starting"):
                stats = prog.get("stats", {})
                bar_placeholder.progress(1.0, text="Selesai!")
                progress_placeholder.success(f"✅ Indexing selesai! {stats}")
                st.session_state["indexing_active"] = False
                break

            # Update progress bar
            pct = (done / total) if total > 0 else 0
            bar_placeholder.progress(min(pct, 1.0), text=f"{done}/{total} file")
            progress_placeholder.info(
                f"⏳ Phase: **{phase}** | File: {current} ({done}/{total})"
            )

            time.sleep(2)

    st.divider()

    # --- Google Drive OAuth + Indexing ---
    st.subheader("☁️ Google Drive")

    gstatus = api_request("auth/google/status", timeout_seconds=90, show_error=False)
    if gstatus is None:
        last_connected = st.session_state.get("gdrive_connected_last", False)
        gstatus = {"connected": last_connected}
        st.info("Status Google Drive belum dapat diambil (backend sibuk). Menampilkan status terakhir.")
    else:
        st.session_state["gdrive_connected_last"] = bool(gstatus.get("connected"))
    if gstatus.get("connected"):
        st.success("Google Drive terhubung")
    else:
        st.warning("Google Drive belum terhubung")

    if st.button("🔐 Connect Google Drive", use_container_width=True):
        login_data = api_request("auth/google/login")
        if login_data and login_data.get("auth_url"):
            st.info("Klik link berikut untuk login dengan email Google yang diinginkan:")
            st.markdown(f"[Login Google Drive OAuth]({login_data['auth_url']})")

    gdrive_folder_id = st.text_input(
        "Google Drive Folder ID (opsional)",
        key="watch_gdrive_folder_id",
        placeholder="Kosongkan untuk semua file yang bisa diakses akun",
        help="Isi jika ingin batasi indexing hanya dalam satu folder tertentu",
    )

    gdrive_monitor_all = st.checkbox(
        "Pantau semua file Google Drive yang bisa diakses",
        key="watch_gdrive_monitor_all",
    )

    st.caption("Watch Google Drive akan tersimpan otomatis saat menjalankan indexing Google Drive.")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if st.button("☁️ Full Index GDrive", use_container_width=True):
            if not gstatus.get("connected"):
                st.error("Hubungkan Google Drive dulu lewat tombol Connect.")
            else:
                current_folder_id = None if gdrive_monitor_all else (gdrive_folder_id or None)
                payload = {"folder_id": current_folder_id, "mode": "full"}
                result = api_request("index/google-drive", method="POST", data=payload)
                if result:
                    if result.get("status") == "already_running":
                        st.warning("⏳ Indexing sedang berjalan. Tunggu hingga selesai.")
                        st.session_state["indexing_active"] = True
                        st.rerun()
                    else:
                        saved = api_request(
                            "index/watch-config",
                            method="POST",
                            data={
                                "gdrive_monitor_all": bool(gdrive_monitor_all),
                                "gdrive_folder_ids": [] if gdrive_monitor_all else ([gdrive_folder_id] if gdrive_folder_id else []),
                            },
                            show_error=False,
                        )
                        if saved:
                            st.session_state["watch_config_updated_at"] = saved.get("updated_at")
                        st.session_state["indexing_active"] = True
                        st.rerun()

    with col_g2:
        if st.button("☁️ Incremental GDrive", use_container_width=True):
            if not gstatus.get("connected"):
                st.error("Hubungkan Google Drive dulu lewat tombol Connect.")
            else:
                current_folder_id = None if gdrive_monitor_all else (gdrive_folder_id or None)
                payload = {"folder_id": current_folder_id, "mode": "incremental"}
                result = api_request("index/google-drive", method="POST", data=payload)
                if result:
                    if result.get("status") == "already_running":
                        st.warning("⏳ Indexing sedang berjalan. Tunggu hingga selesai.")
                        st.session_state["indexing_active"] = True
                        st.rerun()
                    else:
                        saved = api_request(
                            "index/watch-config",
                            method="POST",
                            data={
                                "gdrive_monitor_all": bool(gdrive_monitor_all),
                                "gdrive_folder_ids": [] if gdrive_monitor_all else ([gdrive_folder_id] if gdrive_folder_id else []),
                            },
                            show_error=False,
                        )
                        if saved:
                            st.session_state["watch_config_updated_at"] = saved.get("updated_at")
                        st.session_state["indexing_active"] = True
                        st.rerun()

    st.divider()

    # --- Info ---
    st.subheader("ℹ️ Tentang Sistem")
    st.markdown("""
    **Chatbot Pencarian File**
    - **Embedding**: SBERT (distiluse-base-multilingual-cased-v2)
    - **Generator**: Ollama LLM (Qwen 2.5 / Mistral)
    - **Database**: PostgreSQL + pgvector
    - **Format**: PDF, DOCX, XLSX, CSV, PPTX, TXT
    """)


# =====================================================================
# MAIN — Chat Interface
# =====================================================================
st.title("🔍 Chatbot Pencarian File")
st.caption("Cari file dan tanyakan isi dokumen Anda menggunakan AI")

# Tombol history & hapus chat
_hcol1, _hcol2, _ = st.columns([1.5, 1.5, 7])
with _hcol1:
    if st.button("📜 Muat History", use_container_width=True):
        history_data = api_request("history?limit=20")
        if history_data:
            loaded = []
            for h in history_data:
                loaded.append({"role": "user", "content": h["user_message"]})
                files = []
                rf = h.get("retrieved_files")
                if rf:
                    try:
                        files = json.loads(rf) if isinstance(rf, str) else rf
                    except Exception:
                        files = []
                loaded.append({
                    "role": "assistant",
                    "content": h["bot_response"] or "",
                    "files": files,
                    "response_time_ms": h.get("response_time_ms", 0),
                })
            st.session_state.messages = loaded
            st.rerun()
        else:
            st.info("Belum ada history atau backend tidak dapat diakses.")
with _hcol2:
    if st.button("🗑️ Hapus Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Inisialisasi chat history di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan semua pesan sebelumnya
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Tampilkan file yang ditemukan (jika ada)
        if msg["role"] == "assistant" and "files" in msg:
            if msg["files"]:
                with st.expander(f"📎 {len(msg['files'])} file ditemukan", expanded=False):
                    for f in msg["files"]:
                        icon = get_file_icon(f.get("file_type", ""))
                        size = format_file_size(f.get("file_size") or 0)
                        sim = f.get("max_similarity") or 0.0
                        chunk_sims = sorted(
                            [c.get("similarity", 0.0) for c in f.get("relevant_chunks", [])],
                            reverse=True,
                        )
                        top3 = chunk_sims[:3]
                        avg_top3 = (sum(top3) / len(top3)) if top3 else 0.0
                        max_sim = top3[0] if top3 else 0.0
                        meta_score = f.get("metadata_score")
                        meta_exact = f.get("metadata_exact_hits")
                        meta_partial = f.get("metadata_partial_hits")
                        meta_fuzzy = f.get("metadata_fuzzy_hits")
                        meta_name_ratio = f.get("metadata_name_ratio")
                        meta_path_ratio = f.get("metadata_path_ratio")
                        meta_text_ratio = f.get("metadata_text_ratio")
                        meta_overlap = f.get("metadata_overlap")
                        meta_compact = f.get("metadata_compact_match")
                        meta_phrase = f.get("metadata_phrase_match")
                        st.markdown(
                            f"{icon} **{f.get('file_name', '-')}** "
                            f"(`{sim:.4f}` relevance score)\n"
                            f"📁 `{f.get('file_path', '-')}`  •  {size}\n"
                            f"avg_top3: `{avg_top3:.4f}`  •  max_sim: `{max_sim:.4f}`"
                        )
                        if meta_score is not None:
                            st.caption(
                                f"metadata_score: {meta_score:.0f}  |  exact: {meta_exact}  |  "
                                f"partial: {meta_partial}  |  fuzzy: {meta_fuzzy}  |  "
                                f"name_ratio: {meta_name_ratio:.2f}  |  path_ratio: {meta_path_ratio:.2f}  |  "
                                f"text_ratio: {meta_text_ratio:.2f}  |  overlap: {meta_overlap:.0f}  |  "
                                f"compact: {int(meta_compact or 0)}  |  phrase: {int(meta_phrase or 0)}"
                            )
                        if f.get("source") and f.get("source") != "local":
                            st.caption(f"Source: {f.get('source')}")
                        if f.get("web_view_link"):
                            st.markdown(f"[Buka file cloud]({f['web_view_link']})")
                        # Tampilkan chunk relevan untuk message history juga
                        chunks_hist = f.get("relevant_chunks", [])
                        if chunks_hist:
                            st.caption(f"📝 {len(chunks_hist)} bagian relevan:")
                            for c in chunks_hist:
                                raw_sim = c.get("raw_similarity")
                                boosted_sim = c.get("boosted_similarity")
                                sim_label = f"{c['similarity']:.4f}"
                                if raw_sim is not None or boosted_sim is not None:
                                    raw_val = raw_sim if raw_sim is not None else c["similarity"]
                                    boosted_val = boosted_sim if boosted_sim is not None else c["similarity"]
                                    sim_label = f"raw: {raw_val:.4f} | boosted: {boosted_val:.4f}"
                                prov = c.get("provenance")
                                prov_label = f" [{prov}]" if prov else ""
                                st.code(
                                    f"[Chunk {c['chunk_index']}] (sim: {sim_label}){prov_label}\n"
                                    f"{c['chunk_text'][:300]}...",
                                    language=None,
                                )

# Chat input
if prompt := st.chat_input("Tanyakan sesuatu... (cth: 'cari file laporan bulanan')"):
    # Tambah pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kirim ke backend
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari dan menganalisis dokumen..."):
            result = api_request("chat", method="POST", data={"message": prompt})

        if result:
            answer = result["answer"]
            files = result.get("retrieved_files", [])
            response_time = result.get("response_time_ms", 0)
            st.session_state["last_query"] = prompt

            # Tampilkan jawaban
            st.markdown(answer)

            # Tampilkan response time
            st.caption(f"⏱️ Response time: {response_time} ms")

            # Tampilkan file yang ditemukan
            if files:
                with st.expander(
                    f"📎 {len(files)} file ditemukan", expanded=True
                ):
                    for f in files:
                        icon = get_file_icon(f["file_type"])
                        size = format_file_size(f["file_size"])
                        sim = f["max_similarity"]
                        chunk_sims = sorted(
                            [c.get("similarity", 0.0) for c in f.get("relevant_chunks", [])],
                            reverse=True,
                        )
                        top3 = chunk_sims[:3]
                        avg_top3 = (sum(top3) / len(top3)) if top3 else 0.0
                        max_sim = top3[0] if top3 else 0.0
                        meta_score = f.get("metadata_score")
                        meta_exact = f.get("metadata_exact_hits")
                        meta_partial = f.get("metadata_partial_hits")
                        meta_fuzzy = f.get("metadata_fuzzy_hits")
                        meta_name_ratio = f.get("metadata_name_ratio")
                        meta_path_ratio = f.get("metadata_path_ratio")
                        meta_text_ratio = f.get("metadata_text_ratio")
                        meta_overlap = f.get("metadata_overlap")
                        meta_compact = f.get("metadata_compact_match")
                        meta_phrase = f.get("metadata_phrase_match")

                        st.markdown(
                            f"{icon} **{f['file_name']}** "
                            f"(relevance score: `{sim:.4f}`)\n"
                            f"📁 `{f['file_path']}`  •  {size}  •  "
                            f"Modified: {f['last_modified']}\n"
                            f"avg_top3: `{avg_top3:.4f}`  •  max_sim: `{max_sim:.4f}`"
                        )
                        if meta_score is not None:
                            st.caption(
                                f"metadata_score: {meta_score:.0f}  |  exact: {meta_exact}  |  "
                                f"partial: {meta_partial}  |  fuzzy: {meta_fuzzy}  |  "
                                f"name_ratio: {meta_name_ratio:.2f}  |  path_ratio: {meta_path_ratio:.2f}  |  "
                                f"text_ratio: {meta_text_ratio:.2f}  |  overlap: {meta_overlap:.0f}  |  "
                                f"compact: {int(meta_compact or 0)}  |  phrase: {int(meta_phrase or 0)}"
                            )
                        if f.get("source") and f.get("source") != "local":
                            st.caption(f"Source: {f.get('source')}")
                        if f.get("web_view_link"):
                            st.markdown(f"[Buka file cloud]({f['web_view_link']})")

                        # Tampilkan chunk relevan (tanpa nested expander)
                        chunks = f.get("relevant_chunks", [])
                        if chunks:
                            st.caption(f"📝 {len(chunks)} bagian relevan:")
                            for c in chunks:
                                raw_sim = c.get("raw_similarity")
                                boosted_sim = c.get("boosted_similarity")
                                sim_label = f"{c['similarity']:.4f}"
                                if raw_sim is not None or boosted_sim is not None:
                                    raw_val = raw_sim if raw_sim is not None else c["similarity"]
                                    boosted_val = boosted_sim if boosted_sim is not None else c["similarity"]
                                    sim_label = f"raw: {raw_val:.4f} | boosted: {boosted_val:.4f}"
                                prov = c.get("provenance")
                                prov_label = f" [{prov}]" if prov else ""
                                st.code(
                                    f"[Chunk {c['chunk_index']}] (sim: {sim_label}){prov_label}\n"
                                    f"{c['chunk_text'][:300]}...",
                                    language=None,
                                )
                        st.divider()


            # Simpan ke session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "files": files,
                "response_time_ms": response_time,
            })
        else:
            error_msg = "Maaf, terjadi kesalahan. Pastikan backend berjalan."
            st.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })


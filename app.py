import gc
import hashlib
import math
from pathlib import Path

import pandas as pd
import streamlit as st

import omr_core as core

LEGEND = ("🟩 correct · 🟥 wrong · 🔵 blue (right answer) · 🟨 faint mark (not counted)  · 🟧 double-shaded")

st.set_page_config(page_title="ASG", page_icon="📝")
ss = st.session_state
for _k, _v in {"results": [], "next_id": 1, "uploader_ver": 0, "manual_answers": {},
               "manual_ver": 0, "flash": []}.items():
    ss.setdefault(_k, _v)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { border-right: 1px solid rgba(127,127,127,0.25); }
    [data-testid="stMetric"] {
        background: rgba(127,127,127,0.08);
        border: 1px solid rgba(127,127,127,0.2);
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# small helper functions
def valid_letter(x):
    return isinstance(x, str) and x.strip().upper() in tuple(core.OPTIONS)


def qlist(nums, limit=8):
    shown = ", ".join(f"Q{n}" for n in nums[:limit])
    return shown + (f" (+{len(nums) - limit} more)" if len(nums) > limit else "")


def name_of(rec):
    return str(ss.get(f"name_{rec['id']}", "") or "").strip()


def good_results():
    return [r for r in ss.results if not r["error"]]


# sidebar: ASG setup
def scheme_input(n_q):
    """Returns {question: letter} for questions 1..n_q, or None while the scheme is incomplete."""
    mode = st.radio("Provide the answers by", ["Uploading a CSV", "Typing / pasting"],
                    horizontal=True, label_visibility="collapsed")
    key = None
    if mode == "Uploading a CSV":
        up = st.file_uploader("Upload Marking scheme", type=["csv"], key="scheme_csv")
        if up is None:
            st.info("Upload the marking scheme to continue.")
            return None
        try:
            all_key = core.parse_key_csv(up.getvalue())
        except ValueError as e:
            st.error(f"Could not read that CSV: {e}")
            return None
        missing = core.missing_questions(all_key, n_q)
        if missing:
            st.error(f"The CSV has no answer for {qlist(missing)}. It needs answers for questions 1-{n_q}.")
            return None
        if len(all_key) > n_q:
            st.info(f"The CSV lists {len(all_key)} answers - only questions 1-{n_q} will be used.")
        key = {q: all_key[q] for q in range(1, n_q + 1)}
    else:
        pasted = st.text_area("Paste the answers in order", height=90,
                              placeholder="e.g.  B C C A D …   or   BCCAD…")
        if st.button("Fill the table from this text"):
            letters = core.parse_pasted_answers(pasted)
            ss.manual_answers = {i + 1: a for i, a in enumerate(letters[:core.TOTAL_QUESTIONS])}
            ss.manual_ver += 1                      # new editor key so the table shows the pasted answers
            if len(letters) != n_q:
                st.warning(f"Found {len(letters)} answers but the exam has {n_q} questions - please check the table.")
        df = pd.DataFrame({
            "Question": list(range(1, n_q + 1)),
            "Answer": pd.Series([ss.manual_answers.get(q) for q in range(1, n_q + 1)], dtype="object"),
        })
        edited = st.data_editor(
            df, hide_index=True, disabled=["Question"],
            height=min(35 * (n_q + 1) + 3, 320), key=f"manual_editor_{ss.manual_ver}",
            column_config={"Answer": st.column_config.SelectboxColumn("Answer", options=list(core.OPTIONS))},
        )
        for q, a in zip(edited["Question"], edited["Answer"]):
            if valid_letter(a):
                ss.manual_answers[int(q)] = a.strip().upper()
            else:
                ss.manual_answers.pop(int(q), None)
        missing = core.missing_questions(ss.manual_answers, n_q)
        if missing:
            st.info(f"Still need an answer for {qlist(missing)}.")
            return None
        key = {q: ss.manual_answers[q] for q in range(1, n_q + 1)}

    st.success(f"Marking scheme ready - {n_q} answers")
    with st.expander("Preview answers"):
        st.dataframe(pd.DataFrame({"Question": list(key), "Answer": list(key.values())}),
                     hide_index=True)
    return key


def sidebar():
    with st.sidebar:
        st.title("ASG Setup")
        # st.subheader("Exam setup")
        label = st.text_input("Exam and Class", placeholder="e.g Physics SA1 Diamond",
                              help="Used to name the output folder and downloaded files.")
        n_q = int(st.number_input(
            "How many questions did you set in the exam?", min_value=1, max_value=core.TOTAL_QUESTIONS,
            value=40, step=1,
            help="The answer booklet normally has 60 questions. However only the number set above by you would be marked."))
        st.subheader("Marking Guide")
        key = scheme_input(n_q)
        with st.expander("Advanced options"):
            batch_size = st.slider("Scripts per batch", 5, 50, 30, 5,
                                   help="Big uploads are marked in chunks of this size, with progress shown "
                                        "after each chunk, so the app stays responsive.")
            min_drop = st.slider("Darkness needed to count as a mark", 10, 50, core.MIN_DROP, 1,
                                 help="Lower it if light pencil marks are being missed; raise it if smudges or "
                                      "shadows are being counted. Applies to scripts marked from now on.")
            ocr_ok = core.ocr_available()
            use_ocr = st.checkbox("Try to read names automatically (experimental, slow)", value=False,
                                  disabled=not ocr_ok,
                                  help="Handwriting recognition is imperfect - always check the names.")
            # if not ocr_ok:
            #     st.caption("Needs: `pip install transformers torch pillow`")
        if ss.results and st.button("Clear results and start a new class"):
            st.divider()
            for r in ss.results:
                ss.pop(f"name_{r['id']}", None)
            ss.results = []
            st.rerun()
    return {"label": label, "n_q": n_q, "key": key, "batch_size": batch_size,
            "min_drop": min_drop, "use_ocr": use_ocr}


# marking
def mark_one(file_name, data, cfg):
    rec = {"id": ss.next_id, "file": file_name, "hash": hashlib.sha1(data).hexdigest(), "error": None}
    ss.next_id += 1
    try:
        res = core.grade_script(data, cfg["key"], cfg["n_q"], cfg["min_drop"])
        rec.update(score=res.score, total=res.total, answered=res.answered, blank=res.blank,
                   faint=res.faint, multi=res.multi, confidence=res.confidence,
                   overlay=core.encode_jpg(res.overlay), crop=core.encode_png(res.name_crop))
        name = ""
        if cfg["use_ocr"]:
            try:
                name = core.ocr_name(res.name_crop)
            except Exception:
                name = ""
        ss[f"name_{rec['id']}"] = name
    except core.OMRError as e:
        rec["error"] = str(e)
    except Exception as e:                                          # never let one bad file stop the batch
        rec["error"] = f"Unexpected problem reading this file ({type(e).__name__})."
    return rec


def mark_files(files, cfg):
    seen = {r["hash"] for r in ss.results}
    n_batches = math.ceil(len(files) / cfg["batch_size"])
    bar = st.progress(0.0, text="Starting…")
    done = failed = dupes = 0
    for b in range(n_batches):
        for f in files[b * cfg["batch_size"]:(b + 1) * cfg["batch_size"]]:
            bar.progress(done / len(files), text=f"Batch {b + 1} of {n_batches} · marking {f.name}")
            data = f.getvalue()
            if hashlib.sha1(data).hexdigest() in seen:
                dupes += 1
            else:
                rec = mark_one(f.name, data, cfg)
                seen.add(rec["hash"])
                ss.results.append(rec)
                failed += bool(rec["error"])
            done += 1
        gc.collect()                                                # release the batch's images before the next one
    bar.progress(1.0, text="Done")
    msg = f"Marked {done - dupes - failed} script(s) in {n_batches} batch(es)."
    if failed:
        msg += f" {failed} could not be read (see below)."
    if dupes:
        msg += f" Skipped {dupes} duplicate file(s) that were already marked."
    ss.flash.append(msg)


# results UI
def remove(rec_id):
    ss.results = [r for r in ss.results if r["id"] != rec_id]
    ss.pop(f"name_{rec_id}", None)


def render_card(rec):
    with st.container(border=True):
        if rec["error"]:
            st.error(f"**{rec['file']}** could not be marked.\n\n{rec['error']}\n\n"
                     "Retake the photo (whole page in frame, good light) and upload it again.")
        else:
            st.image(rec["crop"])
            st.text_input("Student name", key=f"name_{rec['id']}", placeholder="Type the name shown above")
            pct = round(100 * rec["score"] / rec["total"]) if rec["total"] else 0
            st.markdown(f"### {rec['score']} / {rec['total']}  ·  {pct}%")
            notes = []
            if rec["blank"]:
                notes.append(f"{len(rec['blank'])} unanswered")
            if rec["multi"]:
                notes.append(f"double-shaded: {qlist(rec['multi'])}")
            if rec["faint"]:
                notes.append(f"faint mark (not counted): {qlist(rec['faint'])}")
            if rec["confidence"] < core.GOOD_MATCH / core.TOTAL_BUBBLES:
                notes.append("low-confidence scan - check the overlay")
            st.caption(f"{rec['file']}" + ("  \n" + "  \n".join(notes) if notes else ""))
        st.button("Remove", key=f"rm_{rec['id']}", on_click=remove, args=(rec["id"],))


def needs_attention(rec):
    return bool(rec["error"] or rec["multi"] or rec["faint"]
                or rec["confidence"] < core.GOOD_MATCH / core.TOTAL_BUBBLES)


def results_section(cfg):
    good = good_results()
    st.header("Check names and scores")
    st.caption("Type each student's name from the handwriting shown. Scores come from the marking scheme "
               "that was active when the script was marked.")
    if good:
        c = st.columns(2)
        c[0].metric("Scripts marked", len(good))
        c[1].metric("Cross check", sum(needs_attention(r) for r in ss.results))
    cols = st.columns(3)
    for i, rec in enumerate(ss.results):
        with cols[i % 3]:
            render_card(rec)

    if not good:
        return
    st.header("Check a marked script")
    by_id = {r["id"]: r for r in good}
    pick = st.selectbox("Script", list(by_id), format_func=lambda i: (
        f"{name_of(by_id[i]) or by_id[i]['file']}  —  {by_id[i]['score']}/{by_id[i]['total']}"))
    st.image(by_id[pick]["overlay"], caption=LEGEND)

    # export
    st.header("Export Student Results")
    rows = [{"name": name_of(r), "score": r["score"], "total": r["total"], "file": r["file"]} for r in good]
    stems = core.assign_filenames([(name_of(r), Path(r["file"]).stem) for r in good])
    csv_bytes = core.build_scores_csv(rows)
    items = [(stem, r["overlay"]) for stem, r in zip(stems, good)]
    slug = core.safe_filename(cfg["label"], "exam").replace(" ", "_")
    unnamed = sum(1 for r in rows if not r["name"])
    if unnamed:
        st.warning(f"{unnamed} script(s) have no name yet. They appear with a blank name in the CSV and are "
                   "named after the photo file in the overlays.")
    failed = len(ss.results) - len(good)
    if failed:
        st.warning(f"{failed} script(s) could not be read and are not included in the downloads.")
    c1, c2 = st.columns(2)
    c1.download_button("Download scores (CSV)", csv_bytes, file_name=f"{slug}_scores.csv", mime="text/csv")
    c2.download_button("Download overlays (ZIP, named by student)", core.build_overlay_zip(items),
                       file_name=f"{slug}_overlays.zip", mime="application/zip")


# main page
def main():
    cfg = sidebar()
    st.title("Automatic Script Grader (ASG)")
    for msg in ss.flash:
        st.success(msg)
    ss.flash = []

    ready = cfg["key"] is not None
    if ready:
        st.markdown(f"**{cfg['label']}** {cfg['n_q']} questions")
    else:
        st.info("Please setup ASG first in the sidebar before you can upload scripts.")

    st.header("Upload scripts")
    files = st.file_uploader("You can upload as many photos as set in the batch size. JPG/PNG format required",
                             type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                             key=f"uploader_{ss.uploader_ver}", disabled=not ready)
    if files:
        n_b = math.ceil(len(files) / cfg["batch_size"])
        st.caption(f"{len(files)} file(s) selected → will be marked in {n_b} batch(es) of up to {cfg['batch_size']}.")
        if len(files) > 60:
            st.warning("That is a lot of photos for one upload - the browser may get slow. "
                       "Consider uploading around 30-50 at a time; results from earlier uploads are kept.")
    if st.button("Mark scripts", type="primary", disabled=not (ready and files)):
        mark_files(files, cfg)
        ss.uploader_ver += 1                    # empties the uploader, ready for the next lot
        st.rerun()

    if ss.results:
        results_section(cfg)

    # ---- dev footer
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 24px 0; opacity: 0.85;">
            <p style="font-size:1.05rem; margin-bottom:6px;">
                Made with ❤️ by <b>Samuel Onwuka</b> for VISA Educators.
            </p>
            <p style="font-size:0.95rem; margin:0;">
                Got a complaint, have a suggestion or just feel like buying me Shawarma 🌯?
                <br>
                Drop me a line at
                <a href="mailto:onwukasamm@gmail.com?subject=OMR%ASG">onwukasamm@gmail.com</a>
                and I'll send a reply almost immediately 🤞🏽.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

main()

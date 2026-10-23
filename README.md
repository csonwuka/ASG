# OMR Automatic Script Grader

A Streamlit app that marks photographed or scanned OMR answer booklets. Give it a marking scheme and a batch of student scripts, and it returns each student's score, a CSV of results, and a marked-up overlay image per script so every mark can be checked by eye.

Built for the 60-question answer booklet used at Voyage International School Abuja (3 blocks of 20 questions, options A to E).

## Features

- **Flexible exam setup.** The booklet always has 60 rows, but the teacher chooses how many questions the exam has (1 to 60). Only those are marked; unanswered questions score 0.
- **Marking scheme your way.** Upload a CSV, or type or paste the answers into a table.
- **Batch marking.** Upload many photos at once; they are marked in chunks with a progress bar.
- **Works with real-world photos.** Handles angled, shadowed, curved or sideways photos, and ignores a teacher's red-pen ticks.
- **Honest about uncertainty.** Double-shaded rows, faint marks and low-confidence scans are flagged rather than silently guessed.
- **Downloads straight to your device.** A scores CSV and a ZIP of overlay images named by student downloadable by Educators

## Quick start

Requires Python 3.9 or newer.

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## How to use it

1. **Exam setup (sidebar).** Enter a class or exam name and the number of questions students should answer.
2. **Marking scheme.** Either upload a CSV or choose *Typing / pasting*, paste answers like `B C C A D …` and correct anything in the table.
3. **Upload scripts.** Drop in photos or scans (JPG or PNG) and press **Mark scripts**.
4. **Check names and scores.** Each script shows a crop of the handwritten name. Type the name in the box beside it. Scores, blanks and warnings appear on each card.
5. **Check a marked script.** Pick any script to view its overlay.
6. **Download.** Get the scores CSV and the overlays ZIP.

Results from earlier uploads are kept in the session, so you can mark a class in several rounds and download everything at the end.

### Marking scheme CSV format

```csv
question,answer
1,B
2,C
3,C
```

A header is optional, and a single column of answers in order also works. The file needs answers for every question from 1 to N. Extra rows are ignored, so a 60-row scheme works for a 40-question exam.

### Outputs

**Scores CSV** has the columns `Name, Score, Out of, Percentage, Script file`.

**Overlay images** are named after each student (for example `Ada Obi.jpg`; duplicates become `Ada Obi (2).jpg`). Colour key:

| Mark | Meaning |
|------|---------|
| 🟩 Green box | Correct answer shaded |
| 🟥 Red box | Wrong answer shaded |
| 🔵 Blue ring | The correct answer, shown when the student was wrong or left it blank |
| 🟧 Orange box | Double-shaded (not counted) |
| 🟨 Yellow box | Faint mark (not counted, flagged for you to check) |

## Contact

Questions, bug reports, or want to say thanks? Email me

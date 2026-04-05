#!/usr/bin/env python3
"""
Preprocessing script for Illinois Courts Case Law.

Label logic (metadata-first, then opinion content):
  1. Property        — foreclosure / ejectment / quiet title / partition (meta + opinion)
                        + real estate disputes + landlord/tenant
  2. Probate         — estate / decedent / executor / testate / intestate
  3. Corporate       — shareholder / directors / corporation / partnership (opinion)
  4. Court of Claims — court of claims
  5. Criminal        — People v. + criminal keyword in opinion
  6. Civil           — tort / contract / insurance / employment / bankruptcy /
                        municipal / administrative / constitutional / civil rights
  7. Other           — everything else (truly miscellaneous civil)

Training text: opinion body starts AFTER "delivered the opinion of the court".
"""

import json, re, os, pickle, random, time
from collections import Counter

try:
    import orjson
    def _load_json(line):
        return orjson.loads(line)
except ImportError:
    def _load_json(line):
        return json.loads(line)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ============ CONFIG ============
INPUT_FILE  = "data-new-labeled/text.data.jsonl"
OUTPUT_DIR  = "data/processed"
RANDOM_SEED = 42
MAX_TEXT_LEN = 15000
MIN_TEXT_LEN = 50

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
#  TEXT EXTRACTION
# ============================================================

_OPINION_MARKER_RE = re.compile(
    r'delivered\s+the\s+opinion\s+of\s+the\s+court',
    re.IGNORECASE
)

# Boilerplate header that appears at the START of some older opinions
# (no "delivered the opinion" marker — strip it directly)
# Broad boilerplate header patterns that can appear at the START of opinions
# (no "delivered the opinion" marker).
# Pattern 1: "Opinion by Justice X. Not to be published..."
# Pattern 2: "Opinion by Not to be published... Justice X." (reversed)
# Pattern 3: "Justice X. Opinion by Not to be published..."
# Also handles: Presiding Justice, JUDGE, MR, typo variations, quotes, commas
_BOILER_RE = re.compile(
    r'^'
    r'(?:'
    # Case 1: Opinion by [TITLE] [NAME]. Not to be published...
    r'Opinion\s+by\s+(?:Presiding\s+)?(?:Justice|JUSTICE|Judge|JUDGE|Mr|MR)\s+[\w]+'
    r'[^\n]*?\.?\s*'
    r'|'
    # Case 2: Opinion by Not to be published... [continuation]
    r'Opinion\s+by\s+Not\s+to\s+be\s+published'
    r'|'
    # Case 3: Justice NAME. Opinion by Not to be published...
    r'Justice\s+[\w]+\.\s+Opinion\s+by\s+Not\s+to\s+be\s+published'
    r')'
    # Capture the rest of the line up to a sentence-ending pattern
    r'[^\n]*?\.?\s*(?=\s*[A-Z][a-z]+\s+[A-Z]|\s*$)',
    re.IGNORECASE
)

def extract_opinion_body(opinion_text):
    """Skip header boilerplate; return text starting after 'delivered the opinion of the court'.

    For cases without the marker (older records), strip the
    "Opinion by Justice X. Not to be published in full." header if present,
    otherwise return the text as-is.
    """
    if not opinion_text:
        return ""
    m = _OPINION_MARKER_RE.search(opinion_text)
    if m:
        body = opinion_text[m.end():]
        body = re.sub(r'^[\s:]+', '', body, count=1)
        return body

    # Try broad regex first
    body = _BOILER_RE.sub('', opinion_text, count=1)
    if body != opinion_text and body.strip():
        return body

    # Fallback: strip line-by-line from the start if it starts with boilerplate
    lines = opinion_text.split('\n')
    skip = 0
    for i, line in enumerate(lines):
        ll = line.strip().lower()
        if (ll.startswith('opinion by') or ll.startswith('opinion of') or
                ll.startswith('opinon by') or ll.startswith('opinion- by') or
                ll.startswith('justice ') or ll.startswith('presidinq justice') or
                ll.startswith('mr. presiding justice') or ll.startswith('mr presiding justice') or
                'not to be published' in ll):
            skip = i + 1
        else:
            break
    if skip > 0:
        body = '\n'.join(lines[skip:])
        if body.strip():
            return body

    return opinion_text


# ============================================================
#  TEXT CLEANING + MASKING
# ============================================================

# Keep legal citations intact (Ill., ILCS, N.E., U.S., etc.) by avoiding
# aggressive punctuation normalization.

# Judge header masking (applied near top of document)
_JUDGE_HEADER_PATTERNS = [
    re.compile(r'(?im)(?:^|\n)\s*(?:Mr\.|Mrs\.|Miss\.|Justice|Chief\s+Justice|Presiding\s+Justice)\s+[A-Z][A-Za-z\u2019\']+(?:[\-\s][A-Z]?[A-Za-z\u2019\']+)*'),
    re.compile(r'(?im)(?:^|\n)\s*JUSTICE\s+[A-Z][A-Z0-9\u2019\']+(?:[\-\.\s][A-Z0-9\.]+)*'),
    re.compile(r'(?im)(?:^|\n)\s*PER\s+CURIAM\s*:?'),
]

_HEADER_SUFFIX_PATTERNS = [
    re.compile(r'(?im)(?:^|\n)\s*delivered the (?:judgment|opinion|order)(?: of the court)?[.,:\s]*'),
    re.compile(r'(?im)(?:^|\n)\s*concur(?:red|s)?[.,:\s]*'),
    re.compile(r'(?im)(?:^|\n)\s*dissent(?:ed|s)?[.,:\s]*'),
]

_ATTORNEY_FOOTER_PATTERNS = [
    re.compile(r'(?is)\n\s*Attorneys?\s+for\s+.*$'),
    re.compile(r'(?is)\n\s*Counsel\s+for\s+.*$'),
    re.compile(r'(?is)\n\s*Judge\s+below,.*$'),
]

_VS_SPLIT_RE = re.compile(r'\s+v\.?\s+|\s+vs\.?\s+', re.IGNORECASE)


def extract_parties(case_name):
    if not case_name:
        return None, None, None
    name = str(case_name).strip()

    m = re.match(r'In re[s]?\s+(.+)$', name, re.IGNORECASE)
    if m:
        in_re = m.group(1).strip()
        in_re = re.sub(r',\s*(Petitioner|Respondent|Claimant|Appellee|Appellant|Plaintiff|Defendant|Estate).*$','', in_re, flags=re.IGNORECASE).strip()
        return None, None, in_re

    parts = _VS_SPLIT_RE.split(name, maxsplit=1)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        strip_role = lambda s: re.sub(r',\s*(Petitioner|Respondent|Claimant|Appellee|Appellant|Plaintiff|Defendant|Estate).*$','', s, flags=re.IGNORECASE).strip()
        return strip_role(a), strip_role(b), None

    return name, None, None


def mask_parties(text, case_name):
    party_a, party_b, in_re = extract_parties(case_name)
    out = text
    if in_re and len(in_re) >= 3:
        out = re.sub(r'\b' + re.escape(in_re) + r'\b', '[PARTY_IN_RE]', out, flags=re.IGNORECASE)
    if party_a and len(party_a) >= 3:
        out = re.sub(r'\b' + re.escape(party_a) + r'\b', '[PARTY_A]', out, flags=re.IGNORECASE)
    if party_b and len(party_b) >= 3:
        out = re.sub(r'\b' + re.escape(party_b) + r'\b', '[PARTY_B]', out, flags=re.IGNORECASE)
    return out


def mask_judge_header(text, header_max=500):
    head = text[:header_max]
    tail = text[header_max:]
    for pat in _JUDGE_HEADER_PATTERNS:
        head = pat.sub('[JUDGE]', head)
    for pat in _HEADER_SUFFIX_PATTERNS:
        head = pat.sub('\n', head)
    head = re.sub(r'\n{3,}', '\n\n', head).rstrip()
    return head + tail


def remove_attorney_footer(text):
    out = text
    for pat in _ATTORNEY_FOOTER_PATTERNS:
        out = pat.sub('', out)
    return out.rstrip()


def clean_text(text, case_name=''):
    """Clean + mask while preserving legal citation tokens."""
    if not text:
        return ""

    text = mask_parties(text, case_name)
    text = mask_judge_header(text)
    text = remove_attorney_footer(text)

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
#  CLASSIFICATION
# ============================================================

def _any(text, keywords):
    return any(k in text for k in keywords)


def _wb(text, keywords):
    """Word-boundary match."""
    import re
    for kw in keywords:
        if re.search(r'(?<![a-zA-Z])' + re.escape(kw) + r'(?![a-zA-Z])', text):
            return True
    return False


def get_label(name, court_name, court_abbr, head_matter, opinion_lc):
    """
    7-class taxonomy for Illinois Courts case law:

    1. Property        — ejectment, foreclosure, landlord/tenant, liens, easements
    2. Probate         — estates, decedents, executors, testate/intestate
    3. Corporate       — corporations, partnerships, shareholders, directors
    4. Court of Claims  — Illinois Court of Claims
    5. Criminal        — criminal cases (People v. or criminal keywords)
    6. Civil           — torts, contracts, insurance, employment, bankruptcy,
                          municipal, administrative, constitutional, civil rights
    7. Other           — truly miscellaneous (mostly early-1800s generic disputes)
    """
    nm_lc    = name.lower()
    hm_lc    = head_matter.lower()
    court_lc = court_name.lower()
    all_meta = nm_lc + ' ' + hm_lc
    op       = opinion_lc[:3000]

    # ── Court of Claims ──────────────────────────────────────
    if 'court of claims' in court_lc or court_abbr == 'Ill. Ct. Cl.':
        return 'Court of Claims'

    # ── Property ──────────────────────────────────────────────
    if _any(nm_lc, ('foreclosure', 'ejectment', 'quiet title',
                    'partition', 'replevin', 'eviction')):
        return 'Property'
    if _any(op, (
        'action of ejectment', 'action of replevin', 'quiet title',
        'partition', 'partition of land', 'partition of property',
        'vendor and vendee', 'vendee and vendor',
        'easement', 'condemnation', 'eminent domain', 'adverse possession',
        'quitclaim deed', 'deed of trust', 'trust deed',
        'forcible entry', 'forcible detainer',
        'right of way', 'subdivision', 'mechanic lien',
        'real estate',
        'dower',
    )):
        return 'Property'
    if _wb(op, (
        'landlord', 'landlords', 'lease', 'leases', 'leasehold',
        'sublease', 'sublet', 'security deposit',
    )):
        return 'Property'

    # ── Criminal ─────────────────────────────────────────────
    # "people" phải đứng SAU "the " (name format: "The People v. Defendant")
    # NOT "people" ở giữa (ex rel. = civil quyền công)
    is_people = 'people' in nm_lc and (
        nm_lc.startswith('people') or
        nm_lc.startswith('the people') or
        nm_lc.startswith('the city of')   # "The City of X v. Defendant" = city ordinance criminal
    )
    # "ex rel." = civil, không phải criminal
    is_ex_rel = 'ex rel.' in nm_lc or 'ex relatione' in nm_lc

    if is_people and not is_ex_rel and _any(op, (
        'murder', 'robbery', 'burglary', 'assault', 'larceny',
        'guilty plea', 'rape', 'escape', 'sentenced', 'kidnap',
        'armed robbery', 'aggravated battery', 'sexual assault',
        'homicide', 'forgery', 'arson', 'drug', 'conspiracy',
        'perjury', 'riot', 'kidnapping', 'bribery', 'embezzlement',
        'penitentiary', 'beyond reasonable doubt', 'indicted',
        'probation revocation',
    )):
        return 'Criminal'
    # Non-People criminal: chỉ khi có conviction language rõ ràng
    if _any(op, (
        'sentenced to', 'imprisoned', 'incarcerated',
        'convicted of', 'found guilty of',
        'first degree murder', 'second degree murder',
        'armed robbery',
    )):
        return 'Criminal'

    # ── Probate ───────────────────────────────────────────────
    PROBATE_KWS = ('estate of', 'in re estate', 'decedent', 'testator',
                   'executor', 'testate', 'probate')
    if _any(nm_lc, PROBATE_KWS):
        return 'Probate'
    if ' in re ' in nm_lc and _any(hm_lc, ('estate', 'probate', 'decedent',
                                              'testator', 'testate', 'intestate')):
        return 'Probate'
    if _any(op, ('estate of', 'executor', 'administrator', 'decedent',
                 'testator', 'testate', 'intestate')):
        return 'Probate'

    # ── Corporate ────────────────────────────────────────────
    # Chỉ keyword mạnh, KHÔNG generic keywords như "defendants"
    if _any(op, (
        'shareholder', 'shareholders', 'directors', 'officers',
        'corporation', 'corporations',
        'bylaws', 'merger', 'derivative action', 'receiver',
        'dissolution of corporation', 'appraisal rights',
        'partnership',
        'quo warranto',
    )):
        return 'Corporate'

    # ── Civil ────────────────────────────────────────────────
    if _any(op, (
        # Tort (including negligence, recover damages)
        'personal injury', 'personal injuries', 'premises liability',
        'medical malpractice', 'legal malpractice', 'wrongful death',
        'assault and battery', 'negligence', 'negligent',
        'action of trespass', 'action of slander', 'action in case',
        'action of detinue', 'action of seduction', 'action of',
        'recover damages', 'recovery of damages',
        # Contract
        'promissory note', 'breach of contract', 'indorser', 'endorser',
        'indorsee', 'holder in due course', 'covenant', 'guaranty', 'surety',
        'action of debt', 'action of assumpsit', 'action of covenant',
        'action of account', 'note payable', 'sealed note',
        'security bond', 'collateral', 'forfeited recognizance',
        'security given', 'forfeiture',
        # Insurance
        'insurance', 'insurer', 'insured', 'coverage', 'indemnity',
        'underinsured motorist', 'bad faith insurance',
        # Employment
        'industrial commission', 'workers compensation',
        "workers' compensation", 'workmen compensation',
        "workmen's compensation", 'employment commission',
        # Bankruptcy
        'bankruptcy', 'chapter',
        # Municipal / Administrative
        'municipal', 'ordinance', 'zoning board', 'city council',
        'municipality', 'police power', 'inverse condemnation', 'public works',
        'mandamus', 'certiorari', 'administrative agency', 'agency review',
        'licensing board', 'disbarment', 'professional license', 'tax commission',
        # Equity / Chancery
        'bill in chancery', 'suit in chancery', 'equity',
        'scire facias', 'interpleader', 'garnishment', 'attachment', 'decree',
        # Family
        'divorce', 'custody', 'adoption', 'child support', 'paternity',
        'maintenance', 'visitation', 'dissolution of marriage',
        'separate maintenance',
    )):
        return 'Civil'

    # ── Other ────────────────────────────────────────────────
    return 'Other'


# ============================================================
#  PROCESSING
# ============================================================

def _count_lines(path):
    return int(
        __import__('subprocess').check_output(
            ['wc', '-l'], stdin=open(path, 'rb')
        ).split()[0]
    )


def load_and_process_data(total_lines):
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_lines, unit='lines', ncols=80,
                    bar_format='{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}]')
    except Exception:
        pbar = None

    stats     = Counter()
    conf_bins = Counter()
    records   = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if pbar:
                pbar.set_postfix_str(f"saved={len(records):,}", refresh=True)
                pbar.update(1)

            try:
                sample = _load_json(line)
            except Exception:
                stats['json_error'] += 1
                continue

            case_id    = sample.get('id', '') or ''
            name       = sample.get('name', '') or ''

            cb = (sample.get('casebody', {}) or {}).get('data', {}) or {}
            hm_raw = cb.get('head_matter', '') or ''
            opinions = cb.get('opinions', [])

            opinion_raw = ''
            if isinstance(opinions, list) and opinions:
                for op_item in opinions:
                    txt = (op_item or {}).get('text', '')
                    if txt and str(txt).strip():
                        opinion_raw = str(txt)
                        break

            if not opinion_raw:
                stats['no_opinion'] += 1
                continue

            court_name = (sample.get('court', {}) or {}).get('name', '') or ''
            court_abbr = (sample.get('court', {}) or {}).get('name_abbreviation', '') or ''

            # Extract opinion body (skip header)
            opinion_body = extract_opinion_body(opinion_raw)

            # Clean + masking for training
            clean_body = clean_text(opinion_body, case_name=name)

            if len(clean_body) > MAX_TEXT_LEN:
                clean_body = clean_body[:MAX_TEXT_LEN]
            if len(clean_body) < MIN_TEXT_LEN:
                stats['too_short'] += 1
                continue
            if not clean_body.strip():
                stats['empty_after_clean'] += 1
                continue

            # Classification
            op_lc = opinion_raw.lower()  # raw for keyword matching
            label = get_label(name, court_name, court_abbr, hm_raw, op_lc)
            stats[label] += 1

            # Confidence: metadata = high, well-sourced opinion = medium, Other = low
            high_conf = {'Property', 'Probate', 'Corporate', 'Court of Claims', 'Criminal'}
            medium_conf = {'Civil'}
            if label in high_conf:
                conf = 0.8
            elif label in medium_conf:
                conf = 0.5
            else:
                conf = 0.3

            ck = 'high' if conf >= 0.6 else ('medium' if conf >= 0.4 else 'low')
            conf_bins[ck] += 1

            records.append({
                'id':         case_id,
                'text':       clean_body,
                'label':      label,
                'name':       name,
                'confidence': conf,
            })
            stats['total_processed'] += 1

    if pbar:
        pbar.close()

    print(f"\n  Total saved: {len(records):,}")
    print(f"\n  Statistics:")
    for k, v in sorted(stats.items()):
        if k != 'total_processed':
            print(f"    {k}: {v:,}")

    print(f"\n  Class distribution:")
    label_order = ['Civil', 'Corporate', 'Court of Claims', 'Criminal', 'Other', 'Probate', 'Property']
    total_cls = sum(stats.get(l, 0) for l in label_order)
    for l in label_order:
        c = stats.get(l, 0)
        print(f"    {l:16s}: {c:6d}  ({100 * c / max(total_cls, 1):.1f}%)")

    print(f"\n  Label confidence:")
    total_conf = sum(conf_bins.values())
    for k in ['high', 'medium', 'low']:
        v = conf_bins.get(k, 0)
        print(f"    {k:6s}: {v:6d}  ({100 * v / max(total_conf, 1):.1f}%)")

    return records


# ============================================================
#  TRAIN / VAL / TEST SPLITS
# ============================================================

def create_splits(records):
    print("\nCreating train/val/test splits...")
    df = pd.DataFrame(records)
    print(f"  Total samples: {len(df):,}")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label'])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['label'])

    print(f"  Train: {len(train_df):,}  ({100*len(train_df)/len(df):.0f}%)")
    print(f"  Val:   {len(val_df):,}  ({100*len(val_df)/len(df):.0f}%)")
    print(f"  Test:  {len(test_df):,}  ({100*len(test_df)/len(df):.0f}%)")
    return train_df, val_df, test_df


# ============================================================
#  SAVE OUTPUTS
# ============================================================

def save_outputs(train_df, val_df, test_df):
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    export_cols = ['id', 'text', 'label']
    train_df[export_cols].to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df[export_cols].to_csv(f"{OUTPUT_DIR}/val.csv",   index=False)
    test_df[export_cols].to_csv(f"{OUTPUT_DIR}/test.csv",  index=False)

    le = LabelEncoder()
    le.fit(train_df['label'])
    with open(f"{OUTPUT_DIR}/label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)

    class_counts   = train_df['label'].value_counts()
    total          = len(train_df)
    n_classes      = len(le.classes_)
    class_weights  = {
        list(le.classes_).index(cls): total / (n_classes * cnt)
        for cls, cnt in class_counts.items()
    }
    with open(f"{OUTPUT_DIR}/class_weights.json", 'w') as f:
        json.dump(class_weights, f, indent=2)

    conf_stats = {
        split: {
            'mean':          float(df['confidence'].mean()) if 'confidence' in df.columns else None,
            'low_conf_pct':  float((df['confidence'] < 0.4).mean() * 100) if 'confidence' in df.columns else None,
            'high_conf_pct': float((df['confidence'] >= 0.6).mean() * 100) if 'confidence' in df.columns else None,
        }
        for split, df in [('train', train_df), ('val', val_df), ('test', test_df)]
    }

    split_info = {
        'train_size':   len(train_df),
        'val_size':     len(val_df),
        'test_size':    len(test_df),
        'num_classes':  len(le.classes_),
        'classes':      list(le.classes_),
        'random_seed':  RANDOM_SEED,
        'max_text_len': MAX_TEXT_LEN,
        'min_text_len': MIN_TEXT_LEN,
        'confidence_stats': conf_stats,
    }
    with open(f"{OUTPUT_DIR}/split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    print("  Saved:")
    print(f"    - train.csv           ({len(train_df):,})")
    print(f"    - val.csv             ({len(val_df):,})")
    print(f"    - test.csv            ({len(test_df):,})")
    print(f"    - label_encoder.pkl")
    print(f"    - class_weights.json")
    print(f"    - split_info.json")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Preprocessing — Illinois Courts Case Law")
    print("  Text: opinion body after 'delivered the opinion of the court'")
    print("  Label: metadata-first, then opinion content")
    print("=" * 60)

    t0 = time.time()
    total_lines = _count_lines(INPUT_FILE)
    t1 = time.time()
    print(f"\nInput file:  {INPUT_FILE}")
    print(f"Total lines: {total_lines:,}  (counted in {t1-t0:.1f}s)")

    records = load_and_process_data(total_lines)
    t2 = time.time()
    print(f"\nProcessing done in {t2-t1:.1f}s")

    train_df, val_df, test_df = create_splits(records)
    save_outputs(train_df, val_df, test_df)

    t3 = time.time()
    print(f"\nTotal time: {t3-t0:.1f}s")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

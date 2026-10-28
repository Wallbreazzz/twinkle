"""Validate an exported listening review against its suite; never infer quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(suite_path, review_path):
    suite_path, review_path = Path(suite_path), Path(review_path)
    suite = json.loads(suite_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if suite.get('format_version') != 1 or review.get('format_version') != 1:
        raise ValueError('Unsupported report version')
    if review.get('suite_sha256') != digest(suite_path):
        raise ValueError('Review belongs to a different or modified suite')
    expected = {}
    for case in suite['cases']:
        for variant, item in case['outputs'].items():
            key = (case['case_id'], variant)
            relative = f'{key[0]}/{variant}.wav'
            path = (suite_path.parent / relative).resolve()
            if not path.is_relative_to(suite_path.parent.resolve()):
                raise ValueError('Audio path escapes suite directory')
            if key in expected or digest(path) != item['sha256']:
                raise ValueError('Duplicate entry or changed audio')
            expected[key] = (relative, item['sha256'])
    if not expected or suite['audio_count'] != len(expected):
        raise ValueError('Invalid suite audio count')
    seen, counts, scores = set(), {}, {}
    for row in review['ratings']:
        key = (row['case_id'], row['variant'])
        if key not in expected or key in seen:
            raise ValueError('Unknown or duplicate rating')
        if (row['audio'], row['audio_sha256']) != expected[key]:
            raise ValueError('Rating audio does not match suite')
        seen.add(key)
        if row['content'] not in ('pending', 'pass', 'fail', 'uncertain'):
            raise ValueError('Invalid content rating')
        if any(not isinstance(row.get(field), str) for field in ('heard_text', 'notes')):
            raise ValueError('Review text fields must be strings')
        counts.setdefault(key[1], Counter())[row['content']] += 1
        for field in ('speaker_similarity', 'naturalness'):
            value = row[field]
            if value is not None and (type(value) is not int or not 1 <= value <= 5):
                raise ValueError('Scores must be null or integers from 1 to 5')
            scores.setdefault(key[1], {}).setdefault(field, []).extend([] if value is None else [value])
    if seen != set(expected):
        raise ValueError('Missing ratings; retain pending rows for unreviewed audio')
    rows = review['ratings']
    complete = all(r['content'] != 'pending' and all(r[f] is not None for f in
                   ('speaker_similarity', 'naturalness')) for r in rows)
    untouched = all(r['content'] == 'pending' and r['speaker_similarity'] is None and
                    r['naturalness'] is None and not r['heard_text'] and not r['notes'] for r in rows)
    status = 'complete' if complete else 'pending' if untouched else 'partial'
    if review.get('status') not in ('pending', 'partial', 'complete'):
        raise ValueError('Invalid review status')
    if review['status'] == 'complete' and not complete:
        raise ValueError('Review claims completion but contains pending fields')
    if not isinstance(review.get('reviewer'), str) or (not untouched and not review['reviewer'].strip()):
        raise ValueError('A named reviewer is required for filled ratings')
    return dict(format_version=1, suite_sha256=digest(suite_path), review_sha256=digest(review_path),
                reviewer=review['reviewer'], status=status, audio_count=len(expected),
                content_counts=counts, score_samples=scores, quality_improvement=None,
                interpretation='Descriptive human ratings only; no automatic quality or similarity verdict')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.suite, args.review)
    # Exclusive creation preserves original reviews and previous summaries.
    with Path(args.output).open('x', encoding='utf-8') as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

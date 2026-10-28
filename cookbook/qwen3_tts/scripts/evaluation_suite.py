"""Fixed text suites and explicit, unscored human review artifacts."""
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata


def text_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def read_suite(path, train_jsonl):
    training = [json.loads(line) for line in Path(train_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not training or any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in training):
        raise ValueError('Training manifest must contain nonempty text rows')
    train_texts = {text_key(r['text']) for r in training}
    cases, ids, texts = [], set(), set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ('id', 'text', 'language', 'category', 'checks'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Suite field {field} must be a nonempty string')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', row['id']) or row['id'] in ids:
            raise ValueError('Suite IDs must be unique safe path components')
        if row['language'] not in ('English', 'Chinese'):
            raise ValueError('This suite supports English or Chinese')
        key = text_key(row['text'])
        if key in train_texts or key in texts:
            raise ValueError('Evaluation text overlaps training or another case')
        ids.add(row['id'])
        texts.add(key)
        cases.append(row)
    if not cases:
        raise ValueError('Evaluation suite is empty')
    return cases


def write_review(out, cases, reports, provenance):
    """Do not turn structural audio validity into a content/quality score."""
    out = Path(out)
    if [r['case_id'] for r in reports] != [c['id'] for c in cases]:
        raise ValueError('Missing or out-of-order case results')
    pending = []
    sections = []
    for case, report in zip(cases, reports):
        rows = []
        if set(report['outputs']) != {'base', 'adapter'}:
            raise ValueError('Unexpected or missing audio variants')
        for variant in ('base', 'adapter'):
            if variant not in report['outputs']:
                continue
            item = report['outputs'][variant]
            relative = f"{case['id']}/{variant}.wav"
            audio = out/relative
            if hashlib.sha256(audio.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Audio hash mismatch before review publication')
            pending.append(dict(case_id=case['id'], variant=variant, audio=relative,
                audio_sha256=item['sha256'], content='pending', heard_text='',
                speaker_similarity=None, naturalness=None, notes=''))
            status = '正常 EOS' if item['ended_with_eos'] else '未正常 EOS：可能截断'
            title={'base':'底模 · 原版','adapter':'Adapter · 原版'}[variant]
            rows.append(f'<div><h3>{title}</h3><p><a href="{relative}">{relative}</a> · '
                f'SHA256 {item["sha256"][:12]}</p><audio controls preload="none" src="{relative}" aria-label="{title}"></audio>'
                f'<p>{item["seconds"]:.2f} 秒 · {status}</p>'
                f'<label>内容完整正确 <select data-index="{len(pending)-1}" data-field="content">'
                '<option value="pending">未评价</option><option value="pass">是</option>'
                '<option value="fail">否</option><option value="uncertain">不确定</option></select></label>'
                f'<label>实际听到的内容 <textarea data-index="{len(pending)-1}" data-field="heard_text"></textarea></label>'
                + ''.join(f'<label>{title} <select data-index="{len(pending)-1}" data-field="{field}">'
                          '<option value="">未评分</option>'+''.join(f'<option>{n}</option>' for n in range(1,6))+'</select></label>'
                          for field,title in [('speaker_similarity','与参考说话人的相似度（1低—5高）'),('naturalness','自然度（1低—5高）')])
                + f'<label>问题记录 <textarea data-index="{len(pending)-1}" data-field="notes"></textarea></label></div>')
        sections.append(f'<section><h2>{html.escape(case["id"])} · {html.escape(case["category"])}</h2>'+
            f'<p class="text">{html.escape(case["text"])}</p><p>检查：{html.escape(case["checks"])}</p>'
            '<div class="pair">'+''.join(rows)+'</div></section>')
    summary = dict(format_version=1, provenance=provenance, case_count=len(cases),
        audio_count=len(pending), eos_count=sum(v['ended_with_eos'] for r in reports for v in r['outputs'].values()),
        human_review_status='pending', quality_improvement=None, cases=reports)
    (out/'suite.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    review = dict(format_version=1, suite_sha256=hashlib.sha256((out/'suite.json').read_bytes()).hexdigest(),
                  reviewer='', status='pending', ratings=pending)
    (out/'review-template.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    embedded = json.dumps(review, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TTS 底模 / Adapter 对照</title><style>
body{font:16px/1.6 system-ui;margin:32px auto;padding:0 20px;max-width:1080px;color:#202b38;background:#f5f7fa}
section{background:white;padding:24px;margin:24px 0;border:1px solid #d8dee6;border-radius:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}audio,textarea{width:100%}label{display:block;margin:12px 0}textarea{min-height:58px;box-sizing:border-box}select,input,button{font:inherit;padding:6px} .text{font-size:20px}
@media(max-width:650px){.pair{grid-template-columns:1fr}}
</style><h1>底模 / Adapter 固定文本对照</h1>
<p>这是已标明模型身份的辅助试听，不是盲测。结构检查通过不代表内容正确或音质提升。评分没有自动填写；公开样本回归中的一次参数更新只验证流程，不代表正式训练收益。</p>
<p>先听参考，再检查漏字、重复、数字和句尾。中文使用英文参考，是跨语言探测；不能替代实际场景评估。完整保留目录，用浏览器打开本文件。</p>
<h2>共同参考语音</h2><audio controls preload="none" src="reference.wav"></audio>
<label>评价人 <input id="reviewer" placeholder="姓名或代号"></label><button id="export">下载评价 JSON</button><p id="message">填写后下载保存；关闭页面不会自动保存。</p>
''' + ''.join(sections) + '<script type="application/json" id="review-data">'+embedded+'''</script><script>
const result=JSON.parse(document.getElementById('review-data').textContent);
// Prevent simultaneous playback; direct file links remain available for independent listening.
document.querySelectorAll('audio').forEach(audio=>audio.addEventListener('play',()=>{
 document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause();});
}));
document.getElementById('export').onclick=()=>{
 result.reviewer=document.getElementById('reviewer').value.trim();
 if(!result.reviewer){document.getElementById('message').textContent='请填写评价人。';return;}
 document.querySelectorAll('[data-index]').forEach(el=>{
   const field=el.dataset.field;let value=el.value;
   if(['speaker_similarity','naturalness'].includes(field))value=value===''?null:Number(value);
   result.ratings[Number(el.dataset.index)][field]=value;
 });
 result.status=result.ratings.every(r=>r.content!=='pending'&&r.speaker_similarity!==null&&r.naturalness!==null)?'complete':'partial';
 result.reviewed_at=new Date().toISOString();
 const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById('message').textContent='已导出评价文件，请与本次结果目录一起保留。';
};</script></html>'''
    (out/'listening.html').write_text(page, encoding='utf-8')
    return summary

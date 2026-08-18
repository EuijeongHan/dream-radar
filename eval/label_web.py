"""모바일 라벨링 웹 UI.

    python -m eval.label_web            # 맥에서 실행 → 폰 브라우저로 접속

`eval/label.py`의 원장·의미론을 **그대로 재사용**합니다. 새 저장소를 만들지 않습니다 —
같은 `eval/labels.jsonl`에 같은 형식으로 기록하므로 터미널 도구와 섞어 써도
이어집니다 (순서도 같은 시드라 동일).

의미론 대응 (터미널 키 ↔ 버튼):
    n → [무관]           basis=title
    k → [보류]           2패스 대상
    a → [초록 보기]      펼친 뒤의 판정은 basis=abstract (terminal의 a와 동일)
    u → [취소]
    2패스(review)는 트리아지가 끝나면 자동으로 이어집니다. 초록이 항상 펼쳐지고
    메모 입력이 생깁니다.

보안: 인증이 없는 LAN 전용 서버입니다. 집 와이파이에서만 쓰세요. 노출되는 건
논문 제목·초록(공개 데이터)이고 받는 건 라벨뿐이며, 원장은 append-only라
무엇이 언제 기록됐는지 항상 추적됩니다.
"""

from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from eval import label as L

PORT = 8765


# ── API 코어 (HTTP와 분리 — 테스트는 이 함수들을 직접 부릅니다) ──────────


def api_status() -> dict[str, Any]:
    decided = L.latest_by_item(L.load_journal())
    days = []
    for date in L.available_dates():
        items = L.load_candidates(date)
        rows = [decided.get(i["id"]) for i in items]
        settled = sum(1 for r in rows if r and r["relevant"] is not None)
        pending = sum(1 for r in rows if r and r["relevant"] is None)
        days.append(
            {
                "date": date,
                "total": len(items),
                "settled": settled,
                "pending": pending,
                "relevant": sum(1 for r in rows if r and r["relevant"] is True),
            }
        )
    return {"days": days}


def api_next(date: str) -> dict[str, Any]:
    """다음 라벨링 대상 1건. 트리아지가 남았으면 트리아지, 아니면 2패스(보류분)."""
    items = L.load_candidates(date)
    translations = L.load_translations(date)
    decided = L.latest_by_item(L.load_journal())

    todo = [i for i in items if i["id"] not in decided]
    pending = [
        i
        for i in items
        if i["id"] in decided
        and decided[i["id"]]["date"] == date
        and decided[i["id"]]["relevant"] is None
    ]

    if todo:
        phase, item = "triage", todo[0]
    elif pending:
        phase, item = "review", pending[0]
    else:
        return {"phase": "done", "progress": _progress(date, items, decided)}

    tr = translations.get(item["id"], {})
    return {
        "phase": phase,
        "item": {
            "id": item["id"],
            "title": item["title"],
            "title_ko": tr.get("title_ko", ""),
            "abstract": item["abstract"],
            "abstract_ko": tr.get("abstract_ko", ""),
            "categories": item["categories"],
            "url": item["url"],
        },
        "progress": _progress(date, items, decided),
    }


def _progress(date: str, items: list[dict], decided: dict) -> dict[str, int]:
    rows = [decided.get(i["id"]) for i in items]
    return {
        "total": len(items),
        "done": sum(1 for r in rows if r is not None),
        "settled": sum(1 for r in rows if r and r["relevant"] is not None),
        "pending": sum(1 for r in rows if r and r["relevant"] is None),
    }


def api_submit(payload: dict[str, Any]) -> dict[str, Any]:
    """라벨 1건 기록. 터미널 도구와 같은 검증·같은 원장."""
    date = payload["date"]
    item_id = payload["item_id"]
    relevant = payload["relevant"]  # true | false | None(보류)
    basis = payload["basis"]

    if basis not in ("title", "abstract"):
        raise ValueError(f"basis 는 title|abstract 여야 합니다: {basis!r}")
    if relevant not in (True, False, None):
        raise ValueError(f"relevant 는 true|false|null 이어야 합니다: {relevant!r}")
    if relevant is None and basis != "title":
        # 보류는 항상 basis=title 로 남습니다 — 터미널 a→k 와 동일 (2패스가 abstract 로 덮음)
        basis = "title"

    items = {i["id"]: i for i in L.load_candidates(date)}
    if item_id not in items:
        raise ValueError(f"{date} 후보에 없는 항목: {item_id}")

    L.append_label(
        L.Label(item_id, date, items[item_id]["title"], relevant, basis, payload.get("note", ""))
    )
    return {"ok": True}


def api_undo() -> dict[str, Any]:
    removed = L.undo_last()
    return {"ok": True, "removed": removed["item_id"] if removed else None}


# ── HTTP ────────────────────────────────────────────────────────────────

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Radar 라벨링</title><style>
:root{--bg:#101418;--card:#1a2027;--fg:#e6e9ec;--dim:#8b949e;--green:#3fb950;--red:#f85149;--amber:#d29922;--blue:#58a6ff}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,system-ui,sans-serif;padding:14px;padding-bottom:120px}
.bar{height:6px;background:#2d333b;border-radius:3px;margin:10px 0 4px}.bar>i{display:block;height:100%;background:var(--blue);border-radius:3px}
.meta{color:var(--dim);font-size:13px;display:flex;justify-content:space-between}
.card{background:var(--card);border-radius:14px;padding:16px;margin-top:12px}
h2{font-size:17px;line-height:1.45}.ko{color:var(--green);font-size:16px;margin-top:8px;font-weight:600}
.cats{color:var(--dim);font-size:12px;margin-top:10px}
.abs{margin-top:12px;font-size:15px;display:none}.abs.show{display:block}
.abs .en{color:var(--dim);font-size:13px;margin-top:10px}
.note{width:100%;margin-top:10px;background:#0d1117;border:1px solid #2d333b;border-radius:8px;color:var(--fg);padding:8px;font-size:14px;display:none}
.note.show{display:block}
.btns{position:fixed;left:0;right:0;bottom:0;background:linear-gradient(transparent,var(--bg) 22%);padding:18px 14px 26px;display:flex;gap:10px}
button{flex:1;border:0;border-radius:12px;padding:16px 6px;font-size:16px;font-weight:700;color:#fff;background:#2d333b}
.bn{background:var(--red)}.bk{background:var(--amber)}.by{background:var(--green)}.ba{background:#30363d}.bu{flex:0 0 64px;background:#21262d;color:var(--dim)}
.phase{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.done{margin-top:40px;text-align:center;color:var(--dim)}
a{color:var(--blue);text-decoration:none;font-size:13px}
.days{display:flex;gap:8px;margin-bottom:4px}.days button{padding:9px;font-size:14px;font-weight:600}
.days .on{background:var(--blue)}
</style></head><body>
<div class="days" id="days"></div>
<div class="bar"><i id="fill" style="width:0%"></i></div>
<div class="meta"><span id="phase" class="phase"></span><span id="prog"></span></div>
<div id="main"></div>
<div class="btns" id="btns"></div>
<script>
let DATE=null, CUR=null, ABS=false;
const $=id=>document.getElementById(id);
async function j(url,opt){const r=await fetch(url,opt);return r.json()}
async function loadDays(){
  const s=await j('/api/status');
  $('days').innerHTML=s.days.map(d=>`<button class="${d.date===DATE?'on':''}" onclick="pick('${d.date}')">${d.date.slice(5)}<br><small>${d.settled}/${d.total}</small></button>`).join('');
  if(!DATE&&s.days.length){DATE=s.days[0].date}
}
function pick(d){DATE=d;next()}
async function next(){
  ABS=false;
  const r=await j('/api/next?date='+DATE);
  await loadDays();
  const p=r.progress;
  $('fill').style.width=(100*p.done/p.total)+'%';
  $('prog').textContent=`${p.done}/${p.total}·보류 ${p.pending}`;
  if(r.phase==='done'){
    $('phase').textContent='완료';
    $('main').innerHTML='<div class="done">이 날짜는 끝났습니다 🎉<br>recheck 는 맥 터미널에서 돌리세요.</div>';
    $('btns').innerHTML='';return;
  }
  CUR=r.item;
  $('phase').textContent=r.phase==='triage'?'1패스 — 제목만 보고':'2패스 — 초록 읽고 판정';
  $('main').innerHTML=`<div class="card">
    <h2>${esc(CUR.title)}</h2>
    ${CUR.title_ko?`<div class="ko">${esc(CUR.title_ko)}</div>`:''}
    <div class="cats">${CUR.categories.join(', ')}</div>
    <div class="abs" id="abs">
      ${CUR.abstract_ko?`<div>${esc(CUR.abstract_ko)}</div>`:''}
      <div class="en">${esc(CUR.abstract)}</div>
      <div style="margin-top:8px"><a href="${CUR.url}" target="_blank">arXiv 원문 ↗</a></div>
    </div>
    <input class="note" id="note" placeholder="메모 (경계 사례일 때만)">
  </div>`;
  if(r.phase==='review'){showAbs();render(['y','n','u'])}
  else{render(['n','k','a','u'])}
}
function showAbs(){ABS=true;$('abs').classList.add('show');$('note').classList.add('show')}
function render(keys){
  const map={n:['무관','bn',()=>send(false)],k:['보류','bk',()=>send(null)],
    a:['초록','ba',()=>{showAbs();render(['y','n','k','u'])}],
    y:['관련','by',()=>send(true)],u:['↩','bu',undo]};
  $('btns').innerHTML='';
  keys.forEach(k=>{const[t,c,f]=map[k];const b=document.createElement('button');
    b.textContent=t;b.className=c;b.onclick=f;$('btns').appendChild(b)});
}
async function send(rel){
  const basis=ABS?'abstract':'title';
  await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({date:DATE,item_id:CUR.id,relevant:rel,basis:basis,note:($('note')||{}).value||''})});
  next();
}
async function undo(){await j('/api/undo',{method:'POST'});next()}
function esc(s){const d=document.createElement('span');d.textContent=s;return d.innerHTML}
loadDays().then(next);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 조용히
        pass

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/status"):
                self._json(api_status())
            elif self.path.startswith("/api/next"):
                date = self.path.split("date=", 1)[-1][:10]
                self._json(api_next(date))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # 라벨링 중 서버가 죽으면 안 됩니다
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/api/label"):
                self._json(api_submit(payload))
            elif self.path.startswith("/api/undo"):
                self._json(api_undo())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.255.255", 1))  # 실제 트래픽 없음 — 라우팅 소스 IP 조회용
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    # flush=True — 백그라운드로 띄우면 stdout 버퍼링 때문에 주소가 안 보입니다
    print("모바일 라벨링 서버 시작", flush=True)
    print(f"  같은 와이파이의 폰에서 열기 →  http://{lan_ip()}:{PORT}", flush=True)
    print(f"  (맥에서 확인하려면 http://localhost:{PORT})", flush=True)
    print(f"  종료: Ctrl-C · 원장: {L.JOURNAL_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

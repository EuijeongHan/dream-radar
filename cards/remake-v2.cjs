const fs=require('fs'),path=require('path'),sharp=require('sharp');
const root=__dirname, C='#F4F0E7',G='#173B32',R='#DE5135';
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
function text(lines,x,y,size=40,color=G,weight=500,leading=1.45){return lines.map((s,i)=>`<text x="${x}" y="${y+i*size*leading}" font-size="${size}" font-weight="${weight}" fill="${color}">${esc(s)}</text>`).join('');}
function panel(lines,y,dark=false){return `<rect x="72" y="${y}" width="936" height="${lines.length*58+65}" rx="20" fill="${dark?G:'#E6E7DD'}"/>`+text(lines,105,y+65,36,dark?C:G);}
const issues=[{slug:'agentrewind',n:'001',title:['AgentRewind: Recoverable Execution','for Long-Horizon LLM Agents'],authors:['Yu Zhuang · Kefei Chen · Yitong Duan','Shuxin Zheng · Jian Li · Xu-Yao Zhang'],url:'https://arxiv.org/abs/2608.14380',basis:'논문 초록 기반 요약 · 실험 수치 미검증',pages:[
{tag:'AI에게 긴 작업을 맡기는 사람에게',head:['AI가 중간에 틀리면,','처음부터 다시','시켜야 할까?'],sub:['실수하기 전으로 돌아가는 AI를','AgentRewind 논문으로 읽습니다.'],boxes:['대화 기록 + 작업 환경','같은 시점으로 되돌리기']},
{tag:'01 / 왜 “다시 해”로 부족할까',head:['대화만 고쳐도','바뀐 파일은 남는다.'],sub:['긴 작업의 오류는 대화 기록과','환경 상태 양쪽으로 퍼질 수 있습니다.'],boxes:['이해를 돕는 예시','잘못 수정한 파일 위에서 계속 작업하면','다음 단계도 틀어질 수 있습니다.'],bottom:['핵심 질문','AI의 생각과 작업장을 함께 복구할 수 있을까?']},
{tag:'02 / 논문의 해결법',head:['게임의 저장 지점을','AI 작업에도.'],sub:['AgentRewind는 대화 맥락과 통제된 환경을','같은 시점의 체크포인트로 기록합니다.'],boxes:['① 같은 시점에 저장','② 오류가 생기면 이전 상태로 복귀','③ 앞선 시도의 정보를 활용해 재개'],bottom:['원리 비유: 게임 저장 지점','모든 외부 행동을 되돌릴 수 있다는 뜻은 아닙니다.']},
{tag:'03 / 어디까지 확인됐나',head:['성공 여부와','중간 진척도를 본다.'],sub:['연구진은 장기 엔지니어링 과제를 위한','MettleBench도 제안했습니다.'],boxes:['초록에서 보고한 결과','비교 기준보다 작업 성공률 개선','평균 체크리스트 진척도 개선'],bottom:['읽을 때 주의할 점','개선 폭과 세부 조건은 이번 요약에서 검증하지 않았습니다.']},
{tag:'04 / 저장해두고 해볼 실험',head:['실패를 한 번','일부러 넣어보세요.'],sub:['관측소의 적용 아이디어 · 논문 실험과 별개'],boxes:['테스트용 작업에서 중간 실패를 만들고,','처음부터 재실행 vs 저장 지점부터 재개를 비교.','완료 여부 · 소요 시간 · 중복 작업을 기록하세요.']}
]},{slug:'eal-bench',n:'002',title:['Agent Memory Is a Surface for','Endogenous Authorization Laundering'],authors:['Tommaso Cerruti · Mika Okamoto','Ansel Kaplan Erol'],url:'https://arxiv.org/abs/2609.01836',basis:'논문 본문 기반 요약 · 결과는 벤치마크 조건에 한정',pages:[
{tag:'승인받고 실행하는 AI를 만드는 사람에게',head:['“허락한 적 없는데”','AI는 왜','실행했을까?'],sub:['기억을 요약하다 권한까지 바뀌는 문제.','EAL-Bench 논문으로 읽습니다.'],boxes:['과거의 허용 ≠ 지금의 허용','기억에 남은 권한을 믿어도 될까?']},
{tag:'01 / 권한은 계속 바뀐다',head:['허용 → 범위 축소','→ 마지막엔 철회.'],sub:['기억을 압축하며 변경 기록을 놓치면','과거의 허용이 현재 권한처럼 남을 수 있습니다.'],boxes:['이해를 돕는 예시','“구매해도 돼” → “이번 구매는 취소해”','요약에 첫 문장만 남는다면?'],bottom:['논문이 다루는 실패','외부 공격 없이 기억 갱신 자체에서 생기는 권한 왜곡']},
{tag:'02 / 기억 오류가 행동으로 이어질까',head:['기억하는 AI와','실행하는 AI를 나눴다.'],sub:['구매 · 사이버보안 · 금융 영역에서','기억 작성 모델 5개, 실행 모델 2개를 평가했습니다.'],boxes:['권한 변경 이력','↓ 지속 기억으로 정리','↓ 그 기억을 보고 실행 여부 판단'],bottom:['검사하는 두 지점','거짓 권한이 생겼나? → 무권한 행동으로 이어졌나?']},
{tag:'03 / 숫자는 조건과 함께',head:['기억 속 거짓 권한,','행동으로 이어졌다.'],sub:['증분 기억 갱신 조건에서 관측한 결과'],stats:true,bottom:['한계','모델·기억 구성에 따라 다르며 실서비스 전체의 확률이 아닙니다.']},
{tag:'04 / 저장해두고 해볼 실험',head:['승인을 취소한 뒤,','AI를 다시 실행해보세요.'],sub:['관측소의 적용 아이디어 · 격리된 테스트 환경에서'],boxes:['승인 → 철회 → 실행 요청 순서로 테스트.','기억과 별도로 권한 원장을 두고 실행 직전 확인.','잘못 허용한 경우와 잘못 거부한 경우를 함께 기록.']}
]}];
async function run(){let thumbs=[];for(const issue of issues){const dir=path.join(root,issue.slug,'v2');fs.mkdirSync(dir,{recursive:true});for(let i=0;i<5;i++){const p=issue.pages[i],last=i===4;let body=`<rect width="1080" height="1350" fill="${C}"/>`;
body+=text([`FROM HAEWON · PAPER RADAR ${issue.n}`],72,72,25,G,700)+`<line x1="72" y1="104" x2="1008" y2="104" stroke="${G}" stroke-width="2"/>`;
body+=text([i===0?'논문 한 편, 5장으로 읽기':p.tag],72,160,28,R,700);
body+=text(p.head,72,265,last?57:66,G,800,1.3);
const sy=i===0?575:last?445:480;body+=text(p.sub,72,sy,34,G,500);
if(p.stats){body+=panel(['최대 50.2%','무권한 요청에 거짓 권한을 만든 비율'],600,true);body+=panel(['98.6%','거짓 권한이 이미 있을 때 실행한 비율'],820);}
else body+=panel(p.boxes,last?510: i===0?740:620,i===0);
if(p.bottom)body+=text(p.bottom,72,1110,25,G,500,1.6);
if(i===0){body+=text([p.tag],72,1050,28,R,700);body+=text(issue.title,72,1120,25,G,600);}
if(last){body+=`<line x1="72" y1="795" x2="1008" y2="795" stroke="${G}"/>`;body+=text(issue.title,72,843,29,G,700);body+=text(issue.authors,72,940,24);body+=text([issue.url,issue.basis],72,1030,24);body+=text(['© fromhaewon. All rights reserved.','권리 표기는 직접 작성한 요약·디자인에 한정됩니다.'],72,1150,22);}
body+=text(['한해원의 관측소'],72,1290,24)+text([`${String(i+1).padStart(2,'0')} / 05`],900,1290,24);
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1350"><g font-family="Arial, Apple SD Gothic Neo, sans-serif">${body}</g></svg>`;const base=path.join(dir,`0${i+1}`);fs.writeFileSync(base+'.svg',svg);await sharp(Buffer.from(svg)).png().toFile(base+'.png');thumbs.push(await sharp(Buffer.from(svg)).resize(270,338).png().toBuffer());}
const source=fs.readFileSync(path.join(root,issue.slug,'caption.md'),'utf8');const extra=issue.n==='001'?'실패를 넣은 테스트 작업에서 전체 재실행과 체크포인트 재개를 비교해보세요. 완료 여부, 시간, 중복 작업을 함께 기록하는 것이 관측소의 적용 제안입니다.':'격리된 테스트에서 승인 → 철회 → 실행 요청을 넣어보세요. 잘못 실행하는 경우뿐 아니라 정상 요청을 거부하는 경우도 세어보는 것이 관측소의 적용 제안입니다.';
const revised=source.replace(/#[\s\S]*$/,'').trim()+`\n\n🔬 직접 해볼 실험\n${extra}\n\n🔭 AI 논문 한 편에서 원리와 적용할 실험을 찾습니다. 다음 개발 전에 저장해두세요.\n\n#AI #AI에이전트 #${issue.n==='001'?'AgentRewind':'AgentMemory'} #논문요약 #한해원의관측소\n`;
fs.writeFileSync(path.join(dir,'caption.md'),revised.replace(issue.n==='001'?'AI 에이전트가 긴 작업 도중 실수하면, 왜 그냥 다음 행동으로 고치기 어려울까요?':'AI 에이전트의 기억이, 사용자가 준 적 없는 권한을 만들어낼 수 있다면 어떨까요?',issue.n==='001'?'AI가 중간에 틀리면, 처음부터 다시 시켜야 할까요?':'“허락한 적 없는데” AI는 왜 실행했을까요?'));
}await sharp({create:{width:1350,height:676,channels:3,background:C}}).composite(thumbs.map((input,i)=>({input,left:(i%5)*270,top:Math.floor(i/5)*338}))).png().toFile(path.join(root,'remake-v2-preview.png'));}
run().catch(e=>{console.error(e);process.exit(1)});

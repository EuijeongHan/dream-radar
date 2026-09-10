const fs=require('fs'),path=require('path'),sharp=require('sharp');
const root=__dirname,ink='#101114',white='#F7F8FA',blue='#3455FF',orange='#FF613B';
const e=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
const t=(a,x,y,z=40,c=ink,w=600,l=1.25)=>a.map((s,i)=>`<text x="${x}" y="${y+i*z*l}" font-size="${z}" font-weight="${w}" fill="${c}">${e(s)}</text>`).join('');
const rect=(x,y,w,h,c,r=0)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${c}"/>`;
const line=(x,y,X,Y,c,w=3)=>`<path d="M${x} ${y} L${X} ${Y}" stroke="${c}" stroke-width="${w}" fill="none"/>`;
const circle=(x,y,r,c)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`;
const arrow=(x,y,X,Y,c)=>line(x,y,X,Y,c,7)+`<path d="M${X-15} ${Y-15} L${X} ${Y} L${X-15} ${Y+15}" stroke="${c}" fill="none" stroke-width="7"/>`;
function oldfile(x,y,c,label){return `<g transform="translate(${x} ${y}) rotate(-6)">`+rect(0,0,220,260,c,18)+rect(25,55,170,12,'#ffffff77',6)+rect(25,92,125,12,'#ffffff77',6)+rect(25,130,150,12,'#ffffff77',6)+t([label],25,225,28,white)+`</g>`;}
function oldchip(x,y,color,kind){return `<g transform="translate(${x} ${y})">`+rect(0,0,260,260,color,45)+Array.from({length:5},(_,i)=>line(30+i*48,-28,30+i*48,0,color,12)+line(30+i*48,260,30+i*48,288,color,12)+line(-28,30+i*48,0,30+i*48,color,12)+line(260,30+i*48,288,30+i*48,color,12)).join('')+rect(40,40,180,180,ink,22)+t([kind],75,162,68,white,800)+`</g>`;}
function base(n,i,dark=false){return rect(0,0,1080,1350,dark?ink:white)+(dark?circle(600,735,520,n==='001'?'url(#halo)':'url(#heat)'):'')+t([`FROM HAEWON  /  PAPER RADAR ${n}`],64,65,24,dark?white:ink,700)+line(64,94,1016,94,dark?'#ffffff44':'#10111433',2)+t(['한해원의 관측소'],64,1295,24,dark?white:ink)+t([`${i+1} / 5`],945,1295,24,dark?white:ink);}

const defs=`<defs>
<linearGradient id="electric" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#D7DEFF"/><stop offset=".32" stop-color="#748AFF"/><stop offset=".65" stop-color="#3455FF"/><stop offset="1" stop-color="#142397"/></linearGradient>
<linearGradient id="fire" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#FFE8B5"/><stop offset=".32" stop-color="#FF9A5A"/><stop offset=".65" stop-color="#FF613B"/><stop offset="1" stop-color="#9E2418"/></linearGradient>
<linearGradient id="metal" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#FFF"/><stop offset=".45" stop-color="#CBD3EC"/><stop offset="1" stop-color="#7084BC"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#5F687F"/><stop offset=".5" stop-color="#242B3D"/><stop offset="1" stop-color="#10131F"/></linearGradient>
<radialGradient id="halo"><stop stop-color="#5669FF" stop-opacity=".4"/><stop offset="1" stop-color="#5669FF" stop-opacity="0"/></radialGradient>
<radialGradient id="heat"><stop stop-color="#FF613B" stop-opacity=".3"/><stop offset="1" stop-color="#FF613B" stop-opacity="0"/></radialGradient>
<filter id="shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="12" dy="22" stdDeviation="18" flood-color="#030611" flood-opacity=".5"/></filter>
</defs>`;
function file(x,y,c,label){return `<g transform="translate(${x} ${y}) rotate(-10)" filter="url(#shadow)">`+rect(18,22,220,260,'#263466',18)+rect(8,10,220,260,'#899CD9',18)+rect(0,0,220,260,'url(#metal)',18)+`<path d="M160 0 L220 60 H174 Q160 60 160 45Z" fill="#F4F8FF"/>`+rect(25,75,150,9,'#6479AF',4)+rect(25,104,110,9,'#6479AF',4)+rect(25,133,140,9,'#6479AF',4)+t([label],22,220,24,'#203365',800)+`</g>`;}
function chip(x,y,color,kind){let g=color===orange?'fire':'electric';return `<g transform="translate(${x} ${y})" filter="url(#shadow)">`+Array.from({length:5},(_,i)=>line(30+i*48,-28,30+i*48,0,'#ADB9D8',12)+line(30+i*48,260,30+i*48,288,'#6F7D9F',12)+line(-28,30+i*48,0,30+i*48,'#ADB9D8',12)+line(260,30+i*48,288,30+i*48,'#6F7D9F',12)).join('')+rect(9,14,260,260,'#161D35',32)+rect(0,0,260,260,'url(#'+g+')',32)+rect(27,27,206,206,'url(#glass)',22)+`<rect x="42" y="42" width="176" height="176" rx="15" stroke="#ffffff44" fill="none"/>`+t([kind],75,162,68,white,800)+`</g>`;}


const pages=[];
let s=base('003',0,true)+t(['논문 한 편, 5장으로 읽기'],64,160,28,'#9AA9FF')+t(['AI에게 두 시간,','탐색은 충분했을까?'],64,270,73,white,900)+chip(400,590,blue,'AI')+t(['02:00:00'],250,1050,100,'#9AA9FF',900)+t(['InferenceBench · 자율 최적화의 빈틈'],64,1180,34,white);pages.push(s);
s=base('003',1)+t(['빠르기만 해서는','통과할 수 없다.'],64,210,76,ink,900)+chip(100,500,blue,'GPU')+t(['H100 1장','실행당 2시간'],500,560,47,ink,800)+t(['모델: Mistral-7B-Instruct-v0.3'],64,920,35)+t(['속도 최적화 + 품질 검사 + 무결성 검사','작동하는 추론 서버를 제출하는 과제입니다.'],64,1040,35,ink,600,1.6);pages.push(s);
s=base('003',2,true)+t(['최고 에이전트보다','설정 탐색이 앞섰다.'],64,210,72,white,900)+t(['PyTorch 기준 대비 종합 속도 향상'],64,370,30,'#BCC0CB');
s+=rect(64,510,590,115,'url(#electric)',12)+t(['8.08×'],690,593,74,white,900)+t(['최고 에이전트 구성'],64,690,32,white)+rect(64,800,840,115,'url(#fire)',12)+t(['11.53×'],64,1020,92,white,900)+t(['SMAC 설정 탐색 · 같은 시간 예산'],64,1090,32,white)+t(['4개 시나리오의 기하평균 · 논문 Table 2'],64,1190,26,'#BCC0CB');pages.push(s);
s=base('003',3)+t(['아는 방법이 많아도,','시험은 적었다.'],64,210,71,ink,900)+t(['비기본 vLLM 설정 시도 수'],64,410,35)+t(['중앙값'],64,500,38,blue)+t(['1'],390,820,330,blue,900)+t(['개'],640,800,75,ink,900)+t(['2시간 실행에서 관측한 행동 분석'],64,1010,35)+t(['지식 부족보다 탐색·검증·최종 제출 과정이','병목일 수 있다는 것이 저자들의 해석입니다.'],64,1110,32,ink,500,1.5);pages.push(s);
s=base('003',4)+rect(0,110,1080,300,blue)+t(['다른 가설 3개부터.'],64,225,69,white,900)+t(['관측소 적용 제안 · 논문 결과와 별개'],64,330,29,white)+t(['후보 전략을 세 가지 만들고 같은 예산으로 비교.','가장 잘 작동한 설정을 저장한 뒤 재실행하세요.'],64,495,34,ink,600,1.6)+t(['한계: 특정 모델·장비·시간 예산의 평가입니다.','현재 모든 에이전트의 성능으로 일반화할 수 없습니다.'],64,660,28,ink,500,1.5)+line(64,780,1016,780,'#BBB');
s+=t(['InferenceBench: A Benchmark for Open-Ended','LLM Inference Optimization by AI Agents'],64,837,29,ink,800)+t(['Jehyeok Yeon · Ben Rank · Maksym Andriushchenko'],64,939,25)+t(['https://arxiv.org/abs/2607.20468','본문 기반 요약 · 이번 재검수: §3–5 및 공식 저장소'],64,1010,25,ink,500,1.5)+t(['© fromhaewon. All rights reserved.','직접 작성한 요약·디자인의 권리 표기입니다.'],64,1170,23,ink,500,1.5);pages.push(s);
(async()=>{const dir=path.join(root,'inferencebench','v4');fs.mkdirSync(dir,{recursive:true});let tiles=[];for(let i=0;i<5;i++){let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1350">'+defs+'<g font-family="Arial, Apple SD Gothic Neo, sans-serif">'+pages[i]+'</g></svg>';fs.writeFileSync(path.join(dir,'0'+(i+1)+'.svg'),svg);await sharp(Buffer.from(svg)).png().toFile(path.join(dir,'0'+(i+1)+'.png'));tiles.push(await sharp(Buffer.from(svg)).resize(324,405).png().toBuffer());}await sharp({create:{width:1620,height:405,channels:3,background:white}}).composite(tiles.map((input,i)=>({input,left:i*324,top:0}))).png().toFile(path.join(dir,'preview.png'));})();

import fs from 'fs';
import path from 'path';
const ROOT = path.resolve('D:/project/binance-predict/public/sites/atoms-dev-c15a3ca0/zh-d83abd2e');
const ASSETS = [
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Mike-TeamLeader-Avatar_origin.DmBYWaXT.webp",
  "kind": "img",
  "meta": "Mike - AI 团队领导 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Adrian-AdsAgent-Avatar.D1HVIhCr.png",
  "kind": "img",
  "meta": "Adrian - AI 广告专家 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Sarah-SEOSpecialist-Avatar_origin.DYHquUJp.webp",
  "kind": "img",
  "meta": "Sarah - AI SEO 专家 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Emma-ProductManager-Avatar_origin.BBeqkRr7.webp",
  "kind": "img",
  "meta": "Emma - AI 产品经理 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Bob-Architect-Avatar_origin.Cdi-oMPW.webp",
  "kind": "img",
  "meta": "Bob - AI 架构师 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Alex-Engineer-Avatar_origin.zHMG8gqX.webp",
  "kind": "img",
  "meta": "Alex - AI 工程师 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/David-DataAnalyst-Avatar_origin.CahzHabe.webp",
  "kind": "img",
  "meta": "David - AI 数据分析师 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Iris-DeepResearcher-Avatar_origin.uohFf0-y.webp",
  "kind": "img",
  "meta": "Iris - AI 深度研究员 Agent"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/why-do-people-trust-atoms/globe/card-background.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/why-do-people-trust-atoms/globe/lower-right-wash.svg?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/why-do-people-trust-atoms/globe/dotted-globe.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/why-do-people-trust-atoms/story/creating-courses-poster.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/community/avatars/avatar-1.webp",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/community/avatars/avatar-2.webp",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/community/avatars/avatar-3.webp",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/community/avatars/avatar-4.webp",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/community/avatars/avatar-5.webp",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/research/ICLR.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/research/arXiv.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/research/NeurIPS.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/research/ICML.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/validation/papers-fade-top.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/validation/papers-fade-bottom.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/validation/github-stars-chart.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/why-do-people-trust-atoms/product-hunt/card-background.webp?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/flag/flag_6.svg?url&no-inline",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/openai.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/openai.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/nvidia.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/nvidia.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/stanford.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/stanford.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/google.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/google.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/amazon.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/amazon.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/mit.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/mit.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/microsoft.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/microsoft.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/salesforce.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/salesforce.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/uc-berkeley.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/uc-berkeley.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/tesla.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/tesla.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light/samsung.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/logo_wall/light-hover/samsung.svg",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/iris.cWsNABAt.png",
  "kind": "img",
  "meta": "Deep Researcher"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/bob.CK05J5j-.png",
  "kind": "img",
  "meta": "Architect"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/adrian.B6ZAN_wb.png",
  "kind": "img",
  "meta": "Ads Specialist"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/david.BH0CUhJj.png",
  "kind": "img",
  "meta": "Data Analyst"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/2.cu10e4p4.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/3.xG7SjSLn.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/4.BPQEa1AB.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/5.ImuIZqcW.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/main.DGt_P6Ei.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/1.D886DBu4.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/2.DeoBqn1x.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/light-1.D-QZgQBZ.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/light-2.HQudE2Kh.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/light-3.BzA-Yc4H.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/light-4.CrMDqrfR.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/light-5.BoVltOAp.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/5.CvYp2tZM.png",
  "kind": "img",
  "meta": "preview"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Number_26.DJ2JRRvr.png",
  "kind": "img",
  "meta": "avatar"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/United%20States.tH2Ut8qB.png",
  "kind": "img",
  "meta": "United States"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/anusha-k.DMch3NTp.png",
  "kind": "img",
  "meta": "Anusha K"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/mike-judkins.D4wyzdUc.png",
  "kind": "img",
  "meta": "Mike Judkins"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/michel-harvey.Cn-GKO2f.png",
  "kind": "img",
  "meta": "Michel Harvey"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/kkangaces.Clqpf7bz.png",
  "kind": "img",
  "meta": "kkangaces210103101"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/hasan.DPTrSrRy.png",
  "kind": "img",
  "meta": "Hasan"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/producthunt.D6_ay3DX.svg",
  "kind": "img",
  "meta": "producthunt"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/kausik-lal.BLZi5wsx.png",
  "kind": "img",
  "meta": "Kausik Lal"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/mia.DZ7PRk0k.png",
  "kind": "img",
  "meta": "Mia"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/beau-carnes.Gvii-MEQ.png",
  "kind": "img",
  "meta": "Beau Carnes"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/stellar.BabQ1cMh.png",
  "kind": "img",
  "meta": "stellar"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/consistent-design72.1D9K2cjJ.png",
  "kind": "img",
  "meta": "Consistent_Design72"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/Union.CbOOVtn6.svg",
  "kind": "img",
  "meta": "Atoms"
 },
 {
  "u": "https://bat.bing.com/action/0?ti=343233335&Ver=2&mid=1e0ed20c-ad0e-4c03-8a25-5c0441dee499&bo=1&sid=62e618f09d7611f1995a13207a19e58c&vid=62e644e09d7611f181c6ebccdc62e4ea&vids=1&msclkid=N&uach=pv%3D19.0.0&pi=0&lg=zh-CN&sw=1440&sh=900&sc=24&nwd=1&tl=Atoms%EF%BC%9A%E7%94%A8%20AI%20%E6%9E%84%E5%BB%BA%E7%BD%91%E7%AB%99%E4%B8%8E%E5%BA%94%E7%94%A8%EF%BC%8C%E6%97%A0%E9%9C%80%E7%BC%96%E7%A0%81&p=https%3A%2F%2Fatoms.dev%2Fzh&r=&lt=2401&mtp=10&evt=pageLoad&sv=2&cdb=AQAQ&rn=183144",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://bat.bing.com/action/0?ti=343233335&Ver=2&mid=1e0ed20c-ad0e-4c03-8a25-5c0441dee499&bo=2&sid=62e618f09d7611f1995a13207a19e58c&vid=62e644e09d7611f181c6ebccdc62e4ea&vids=0&msclkid=N&ea=homepage_hero_case_exposure&en=Y&p=https%3A%2F%2Fatoms.dev%2Fzh&sw=1440&sh=900&sc=24&nwd=1&evt=custom&cdb=AQAQ&rn=538593",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://bat.bing.com/action/0?ti=343233335&Ver=2&mid=1e0ed20c-ad0e-4c03-8a25-5c0441dee499&bo=3&sid=62e618f09d7611f1995a13207a19e58c&vid=62e644e09d7611f181c6ebccdc62e4ea&vids=0&msclkid=N&ea=homepage_logo_wall_exposure&en=Y&p=https%3A%2F%2Fatoms.dev%2Fzh&sw=1440&sh=900&sc=24&nwd=1&evt=custom&cdb=AQAQ&rn=321993",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://bat.bing.com/action/0?ti=343233335&Ver=2&mid=1e0ed20c-ad0e-4c03-8a25-5c0441dee499&bo=4&sid=62e618f09d7611f1995a13207a19e58c&vid=62e644e09d7611f181c6ebccdc62e4ea&vids=0&msclkid=N&ea=homepage_third_party_queue_completed&en=Y&p=https%3A%2F%2Fatoms.dev%2Fzh&sw=1440&sh=900&sc=24&nwd=1&evt=custom&cdb=AQAQ&rn=490923",
  "kind": "img",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/nuxt-mgx/prod/assets/starts.X91XbzmB.png?url",
  "kind": "bg",
  "meta": "star blink"
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/saas-landing/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/independent-brands/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/film-studio/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/popular/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/creative-video/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://public-frontend-cos.metadl.com/commonfile/home/v1.3.x/founder-builds/shooting-game/poster.webp",
  "kind": "poster",
  "meta": ""
 },
 {
  "u": "https://atoms.dev/favicon.ico",
  "kind": "favicon",
  "meta": ""
 },
 {
  "u": "https://atoms-cos.metadl.com/cms_medias/logo_only_blue_d7ad9c419a.png",
  "kind": "og",
  "meta": ""
 }
];
fs.mkdirSync(ROOT,{recursive:true});
const safe=(u,i)=>{let n=u.split('?')[0].split('/').filter(Boolean).pop()||('asset-'+i);n=decodeURIComponent(n).replace(/[^a-zA-Z0-9._-]+/g,'_');return i+'-'+n;};
const results=[];
for(let i=0;i<ASSETS.length;i+=4){const batch=ASSETS.slice(i,i+4);await Promise.all(batch.map(async(a2,idx)=>{const name=safe(a2.u,i+idx);try{const r=await fetch(a2.u,{headers:{'User-Agent':'Mozilla/5.0'}});if(!r.ok)throw new Error('HTTP '+r.status);const buf=Buffer.from(await r.arrayBuffer());fs.writeFileSync(path.join(ROOT,name),buf);results.push({name,kind:a2.kind,bytes:buf.length,src:a2.u});}catch(e){results.push({name,kind:a2.kind,error:String(e.message),src:a2.u});}}));}
fs.writeFileSync(path.join(ROOT,'_download-results.json'),JSON.stringify(results,null,2));
console.log('downloaded',results.filter(r=>r.bytes).length,'failed',results.filter(r=>r.error).length);

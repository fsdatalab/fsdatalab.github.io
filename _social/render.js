const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1200,height:630},deviceScaleFactor:2});
await p.goto('file://'+process.argv[2]);await p.evaluate(()=>document.fonts.ready);
console.log(await p.evaluate(()=>[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family+f.weight).join(',')));
await p.screenshot({path:process.argv[3]});await b.close()})()

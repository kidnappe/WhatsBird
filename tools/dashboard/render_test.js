// 用桩 DOM 真跑一遍监控台的页面脚本，验证渲染路径不抛异常、状态条内容正确。
// 用法: node render_test.js <page.js> <payload.json>
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const PAYLOAD = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

const els = {};
function mk(id) {
  const o = {
    id, style: {}, dataset: {}, textContent: '', innerHTML: '', value: '',
    children: [], childElementCount: 0, tagName: 'DIV',
    // 真浏览器里 <select> 的 options 由 innerHTML 填充；桩必须给够，否则
    // rs.options[selectedIndex].text 会踩 undefined
    options: [{ text: '' }, { text: '' }, { text: '' }, { text: '' },
              { text: '' }, { text: '' }],
    selectedIndex: 0, disabled: false, firstElementChild: { style: {} },
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
                 contains(c) { return this._s.has(c); }, toggle() {} },
    getContext() { return new Proxy({}, { get: () => () => {} }); },
    appendChild(c) { this.children.push(c); },
    addEventListener() {}, removeEventListener() {},
  };
  return o;
}
global.document = {
  getElementById(id) { return els[id] || (els[id] = mk(id)); },
  createElement(t) { return mk('new:' + t); },
  querySelector() { return mk('q'); },
  querySelectorAll() { return []; },
  addEventListener() {},
  body: mk('body'),
};
global.window = {};
global.localStorage = { getItem: () => '', setItem: () => {} };
global.location = { hash: '#dl' };
global.fetch = async () => ({ json: async () => PAYLOAD });
const timers = [];
const realSetTimeout = global.setTimeout;      // 页面脚本会把 setTimeout 打桩，自己留一份
global.setTimeout = (fn) => { timers.push(fn); return timers.length; };
global.setInterval = () => 0;
global.requestAnimationFrame = () => 0;

let failed = false;
process.on('unhandledRejection', (e) => {          // tick() 是浮动 promise，异常只能这样抓
  failed = true;
  console.log('❌ tick() 未处理的 rejection:', (e && e.stack) || e);
});
process.on('uncaughtException', (e) => {
  failed = true;
  console.log('❌ 未捕获异常:', (e && e.stack) || e);
});
// tick() 的 catch-all 会把渲染异常吞成"(连接中断，重试中)"——正是这套面板的通病。
// 测试时把它掀开，否则永远看不到真因。
const patched = src.replace(
  "}catch(e){ $('sub').textContent='(连接中断，重试中)'; }",
  "}catch(e){ console.log('❌ tick 内部异常:', (e && e.stack) || e); throw e; }");
if (patched === src) console.log('⚠ 没能掀开 tick 的 catch-all（代码变了？）');
try {
  eval(patched);                   // 顶层：定义函数，最后自动 tick()
} catch (e) {
  failed = true;
  console.log('❌ 顶层执行抛异常:', e && e.stack || e);
}

realSetTimeout(() => {
  const strip = els['strip'] ? els['strip'].innerHTML : '(缺 #strip)';
  console.log('--- 状态条 innerHTML ---');
  console.log(strip.replace(/></g, '>\n<'));
  const chips = (strip.match(/class="chip/g) || []).length;
  const bad = (strip.match(/chip bad/g) || []).length;
  const ok = (strip.match(/chip ok/g) || []).length;
  // 卡片数**不能写死**：原来断言 ===4，加了东北（第 5 个区）之后测试就假失败。
  // 改成和载荷里 overview 的条数比对 —— 这才是"渲染完整"的真正含义。
  const wantChips = (PAYLOAD.overview || []).length;
  console.log(`\n状态条: ${chips} 个 chip（载荷 overview ${wantChips} 条；bad=${bad} ok=${ok}）`);
  console.log('fresh 文本     :', JSON.stringify(els['fresh'] && els['fresh'].textContent));
  console.log('bar2label 文本 :', JSON.stringify(els['bar2label'] && els['bar2label'].textContent));
  console.log('dlcaption 文本 :', JSON.stringify(els['dlcaption'] && els['dlcaption'].textContent));
  console.log('dl_eta 文本    :', JSON.stringify(els['dl_eta'] && els['dl_eta'].textContent));
  console.log('stateTxt 文本  :', JSON.stringify(els['stateTxt'] && els['stateTxt'].textContent));
  console.log('stopTrain      : disabled=' + (els['stopTrain'] && els['stopTrain'].disabled));
  const logHtml = String(els['log'] && els['log'].innerHTML);
  const lv = ['lv-ok', 'lv-warn', 'lv-err', 'lv-info'].filter(c => logHtml.includes(c));
  console.log('事件日志分色   :', lv.length ? lv.join(', ') : '(无 level class)');
  // 断言
  // 长数值必须自动降字号，否则被 .card .v 的 nowrap+ellipsis 截成省略号
  // （现场："+17,230 / 30,424" 在 26px 粗体下被吃掉）
  const rowsTxt = String(els['dl_rows'] && els['dl_rows'].textContent);
  const rowsPx = parseFloat(els['dl_rows'] && els['dl_rows'].style && els['dl_rows'].style.fontSize) || 26;
  console.log('dl_rows 文本   :', JSON.stringify(rowsTxt), '字号', rowsPx + 'px');
  const checks = [
    [`状态条渲染出全部 ${wantChips} 个 chip`, chips === wantChips && wantChips > 0],
    // ok/bad 卡片数与载荷一致（东北入库后五区可全绿，不能再假设至少一张 bad）
    [`ok 卡片数与载荷一致（${(PAYLOAD.overview||[]).filter(o=>o.lv==='ok').length}）`,
     ok === (PAYLOAD.overview||[]).filter(o=>o.lv==='ok').length],
    [`bad 卡片数与载荷一致（${(PAYLOAD.overview||[]).filter(o=>o.lv==='bad').length}）`,
     bad === (PAYLOAD.overview||[]).filter(o=>o.lv==='bad').length],
    ['进度条用物种口径', String(els['bar2label'] && els['bar2label'].textContent).includes('物种')],
    ['caption 含"计划"标注', String(els['dlcaption'] && els['dlcaption'].textContent).includes('计划')],
    // 停止按钮：训练已完成（done && final）应禁用（"✓ 已结束"），训练中必须可用
    // —— 页面设计行为。断言按载荷状态判定，两种载荷都能跑（2026-10-06 改，
    // 原断言硬编码 ===false，用真实已完成载荷回归时会误报）。
    ['停止按钮状态与载荷一致（完成禁用/进行中可用)',
     els['stopTrain'] && els['stopTrain'].disabled === !!(PAYLOAD.done && PAYLOAD.final)],
    ['事件日志带 level class', lv.length > 0],
    ['日志已 HTML 转义（无裸 <）', !/<(?!\/?(div|b|span)\b)/.test(logHtml)],
    [`长数值自动降字号（"${rowsTxt}" → ${rowsPx}px）`, rowsTxt.length <= 11 || rowsPx < 26],
    ['卡片不出现 "None"', !strip.includes('None')],
  ];
  let allOk = !failed;
  console.log('');
  for (const [name, okFlag] of checks) {
    console.log(`  ${okFlag ? '✓' : '✗'} ${name}`);
    if (!okFlag) allOk = false;
  }
  console.log(allOk ? '\n渲染测试通过' : '\n渲染测试失败');
  process.exit(allOk ? 0 : 1);
}, 50);

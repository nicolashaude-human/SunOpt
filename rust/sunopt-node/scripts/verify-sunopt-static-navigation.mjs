// RUN_SUNOPT_STATIC_NAV_VERIFY.cmd - read-only static-pages navigation validation.
// Writes a diagnostics JSON only to updates/outbox.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const read = name => fs.readFileSync(path.join(root,name),'utf8');
const source = read('public/scripts/sunopt-static-navigation.js');
const htmlProject = read('public/project-structure.html');
const htmlSlides = read('public/presentation/index.html');
const presentationPage = read('src/app/presentation/page.tsx');
const siteNav = read('src/components/site-navigation.tsx');
const layout = read('src/app/layout.tsx');
const theme = read('public/styles/sunopt-theme.css');
const docs = read('src/app/documentation/code-path/page.tsx');
const checks = [];
function check(name, test) { checks.push({name, pass: !!test}); }
function fakeDom(pathname) {
  class E {
    constructor(tag) {
      this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {};
      this.dataset = {}; this.className = ''; this.href = ''; this.tabIndex = -1;
      this.scrollLeft = 0; this.inert = false;
      this.classList = {toggle: (cls, enabled) => {
        const names = new Set(this.className.split(/\s+/).filter(Boolean));
        if (enabled) names.add(cls); else names.delete(cls);
        this.className = [...names].join(' ');
      }};
    }
    setAttribute(k,v) { this.attributes[k] = String(v); }
    getAttribute(k) { return this.attributes[k] ?? null; }
    addEventListener(k, fn) { (this.listeners[k]??=[]).push(fn); }
    dispatch(k, event) { for (const fn of this.listeners[k]??[]) fn(event); }
    append(...nodes) { this.children.push(...nodes); }
    appendChild(node) { this.append(node); }
    replaceChildren(...nodes) { this.children=nodes; }
    contains(node) { return this===node || this.children.some(child=>child.contains(node)); }
    focus() { this.focused = true; }
    setPointerCapture() { this.pointerCaptured=true; }
    hasPointerCapture() { return !!this.pointerCaptured; }
    releasePointerCapture() { this.pointerCaptured=false; }
  }
  const mount = new E('mount');
  const doc = {getElementById:id=>id==='sunopt-static-navigation'?mount:null,
    createElement:tag=>new E(tag)};
  vm.runInNewContext(source,{window:{location:{pathname}},document:doc});
  return mount.children[0];
}
for(const [route,expected] of [
  ['/project-structure.html','Project Structure'],
  ['/presentation/','Präsentation']]) {
  const header = fakeDom(route);
  const row = header.children[0];
  const main = row.children[1];
  const disclosure = row.children[2];
  const hover = header.children[1];
  const extra = hover.children[0].children[0];
  check(`${route}: three primary + selected fourth`,
    main.children.map(n=>n.textContent).join('|')===`Jahressimulation|Modellvergleich|Tankvergleich|${expected}`);
  check(`${route}: active button selected`, main.children.filter(n=>n.getAttribute('aria-current')==='page')
    .map(n=>n.textContent).join('')===expected);
  check(`${route}: hover opens dropdown`, (header.dispatch('pointerenter',{pointerType:'mouse',buttons:0}),
    hover.className.includes('is-open') && extra.tabIndex===0 && !hover.inert));
  check(`${route}: mouse leave closes dropdown`,(header.dispatch('pointerleave',{pointerType:'mouse'}),
    !hover.className.includes('is-open') && extra.tabIndex===-1 && hover.inert));
  check(`${route}: icon opens for touch`,(disclosure.dispatch('click',{}),
    disclosure.getAttribute('aria-expanded')==='true'));
  check(`${route}: Escape closes`, (header.dispatch('keydown',{key:'Escape'}),
    disclosure.getAttribute('aria-expanded')==='false'));
}
check('project structure mounts shared static navigation',
  htmlProject.includes('id="sunopt-static-navigation"') &&
    htmlProject.includes('src="/scripts/sunopt-static-navigation.js"'));
check('site navigation opens the embedded presentation',
  siteNav.includes('href: "/presentation"') && layout.includes('<SiteNavigation />') &&
    presentationPage.includes('src="/presentation/index.html"'));
check('project structure old bespoke links removed', !htmlProject.includes('class="site-nav"'));
check('standalone HTML retains the slide tabs',htmlSlides.includes('id="slide-tabs"') && htmlSlides.includes('id="slide-root"'));
check('floating hover menu never changes layout height',theme.includes('.sunopt-hover-pages { position: absolute'));
check('shared site theme and scroll',theme.includes('.sunopt-static-header') && theme.includes('.sunopt-drag-scroll'));
check('unsupported controller paragraph and stage removed',!docs.includes('Der Controller hält die 28 unabhängigen Tanks') &&
  !docs.includes('title: "GPU-Jahrescontroller"'));
check('unintelligible A/B description removed',!docs.includes('Der unveränderte Schalter im Tankvergleich wählt den präziseren'));
const result={tool:'sunopt-static-navigation-v3-verify',at:new Date().toISOString(),
  passed:checks.every(t=>t.pass),checks};
const out=path.join(root,'updates/outbox/sunopt-static-navigation-v3-verify.json');
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');
for(const c of checks) console.log(`${c.pass?'PASS':'FAIL'} ${c.name}`);
console.log(`Report: ${out}`);
if(!result.passed) process.exitCode=1;

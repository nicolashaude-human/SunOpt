/* public/presentation/annual-static-slide.js
 * Folie 2: Statisches Ergebnis einer EINMALIGEN Jahresberechnung.
 * Kein Aufruf einer Berechnungs-API, kein JIT und keine Simulation im Browser.
 * Bewusst als Overlay: aktuelle slides.js und presentation-data.js bleiben unangetastet.
 */
(() => {
  'use strict';
  const frame = document.getElementById('slide-frame');
  const counter = document.getElementById('slide-counter');
  if (!frame || !counter) return;
  let token = 0;
  function isYearSlide() {
    const c = (counter.textContent || '').trim();
    const tab = document.querySelector('#slide-tabs .slide-tab[aria-current="page"]');
    return /^2\s*\//.test(c) || (!c && tab && /^2\./.test(tab.textContent || ''));
  }
  function show() {
    if (!isYearSlide()) return;
    if (frame.querySelector('[data-sunopt-static-annual-slide="true"]')) return;
    const version = ++token;
    const article = document.createElement('article');
    article.setAttribute('data-sunopt-static-annual-slide','true');
    article.className = 'sunopt-annual-slide';
    article.innerHTML = '<p class="slide-kicker">Folie 2 · Jahressimulation</p>' +
      '<h1 class="slide-title">Jahressimulation · festgehaltene Ergebnisse</h1>' +
      '<p class="sunopt-annual-snapshot-status" role="status">Gespeichertes Jahresergebnis wird geladen …</p>';
    frame.replaceChildren(article);
    // Ausschliesslich eine STATISCHE Datei lesen, niemals die Simulations-API.
    fetch('/presentation/annual-simulation-static.json', {cache:'no-store'}).then(res => {
      if(!res.ok) throw new Error('Noch kein Snapshot vorhanden. Einmalig CAPTURE_SUNOPT_ANNUAL_SLIDE.cmd ausführen.');
      return res.json();
    }).then(snapshot => {
      if(version!==token || !isYearSlide() || !article.isConnected) return;
      if(snapshot?.schema !== 'sunopt-presentation-year-static-v1' || snapshot?.source?.dataPoints < 30000)
        throw new Error('Statische Jahresdaten sind unvollständig.');
      const iframe=document.createElement('iframe');
      iframe.title='Statische Jahressimulation – 2023';
      iframe.className='sunopt-annual-static-frame';
      iframe.setAttribute('loading','eager');
      iframe.src='/presentation/annual-simulation-static.html';
      article.replaceChildren(iframe);
    }).catch(error => {
      if(version!==token || !isYearSlide() || !article.isConnected) return;
      const p=article.querySelector('.sunopt-annual-snapshot-status');
      if(p) p.textContent = String(error instanceof Error ? error.message : error);
    });
  }
  const observer = new MutationObserver(() => {
    if (!isYearSlide()) { token++; return; }
    if (!frame.querySelector('[data-sunopt-static-annual-slide="true"]')) show();
  });
  observer.observe(frame,{childList:true});
  observer.observe(counter,{childList:true,characterData:true,subtree:true});
  show();
})();

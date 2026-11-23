/* The static SunOpt pages do not pass through Next.js RootLayout.
 * Provide the same primary menu, context tab and hover selection as SiteNavigation.
 * No changes to simulation code or to the presentation slide engine.
 */
(() => {
  'use strict';
  const mount = document.getElementById('sunopt-static-navigation');
  if (!mount) return;
  const primary = [
    { label: 'Jahressimulation', href: '/debug/hot-water' },
    { label: 'Modellvergleich', href: '/model-comparison' },
    { label: 'Tankvergleich', href: '/tank-comparison' }
  ];
  const secondary = [
    { label: 'Projektstatus', href: '/project-status' },
    { label: 'Beschleunigung A/B', href: '/acceleration-strategy' },
    { label: 'Ansatz B', href: '/approach-b' },
    { label: 'Präsentation', href: '/presentation' },
    { label: 'Codepfad', href: '/documentation/code-path' },
    { label: 'Physik', href: '/documentation/physics' },
    { label: 'Project Structure', href: '/project-structure.html' },
    { label: 'Project Cleanup', href: '/debug/project-cleanup' }
  ];
  const currentPath = window.location.pathname.replace(/\/$/, '') || '/';
  const current = [...primary, ...secondary].find(item => item.href === currentPath);
  const context = secondary.find(item => item.href === currentPath);
  const header = document.createElement('header');
  header.className = 'sunopt-static-header sunopt-site-header';
  const row = document.createElement('div');
  row.className = 'sunopt-static-nav-row';
  const brand = document.createElement('a');
  brand.href = '/';
  brand.className = 'sunopt-static-brand';
  brand.setAttribute('aria-label', 'SunOpt Startseite');
  const logo = document.createElement('img');
  logo.src = '/icon.png';
  logo.alt = '';
  logo.width = 38; logo.height = 38;
  const brandText = document.createElement('span');
  brandText.textContent = 'SunOpt';
  brand.append(logo, brandText);
  const mainNav = document.createElement('nav');
  mainNav.className = 'sunopt-static-nav-scroll sunopt-drag-scroll';
  mainNav.setAttribute('aria-label', 'SunOpt Hauptnavigation');
  mainNav.tabIndex = 0;
  const link = (item) => {
    const a = document.createElement('a');
    a.href = item.href;
    a.className = 'sunopt-nav-link';
    a.textContent = item.label;
    if (item.href === currentPath) a.setAttribute('aria-current', 'page');
    return a;
  };
  [...primary, ...(context ? [context] : [])].forEach(item => mainNav.appendChild(link(item)));
  const disclosure = document.createElement('button');
  disclosure.type = 'button';
  disclosure.className = 'sunopt-nav-disclosure';
  disclosure.textContent = '▾';
  disclosure.setAttribute('aria-label', 'Seitenauswahl öffnen');
  disclosure.setAttribute('aria-haspopup', 'true');
  disclosure.setAttribute('aria-controls', 'sunopt-static-hover-pages');
  disclosure.setAttribute('aria-expanded', 'false');
  row.append(brand, mainNav, disclosure);

  const hover = document.createElement('div');
  hover.id = 'sunopt-static-hover-pages';
  hover.className = 'sunopt-hover-pages';
  hover.setAttribute('aria-hidden', 'true');
  hover.inert = true;
  const hoverInner = document.createElement('div');
  hoverInner.className = 'sunopt-static-hover-inner';
  const extraNav = document.createElement('nav');
  extraNav.className = 'sunopt-static-nav-scroll sunopt-drag-scroll';
  extraNav.setAttribute('aria-label', 'Weitere SunOpt-Seiten');
  extraNav.tabIndex = -1;
  secondary.forEach(item => extraNav.appendChild(link(item)));
  hoverInner.appendChild(extraNav);
  hover.appendChild(hoverInner);
  header.append(row, hover);
  mount.replaceChildren(header);

  let open = false;
  function setOpen(value) {
    open = Boolean(value);
    hover.classList.toggle('is-open', open);
    hover.setAttribute('aria-hidden', String(!open));
    hover.inert = !open;
    extraNav.tabIndex = open ? 0 : -1;
    disclosure.setAttribute('aria-expanded', String(open));
  }
  header.addEventListener('pointerenter', event => {
    if (event.pointerType === 'mouse' && event.buttons === 0) setOpen(true);
  });
  header.addEventListener('pointerleave', event => {
    if (event.pointerType === 'mouse') setOpen(false);
  });
  header.addEventListener('focusin', () => setOpen(true));
  header.addEventListener('focusout', event => {
    if (!event.relatedTarget || !header.contains(event.relatedTarget)) setOpen(false);
  });
  header.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      disclosure.focus();
      setOpen(false);
    }
  });
  disclosure.addEventListener('click', () => setOpen(!open));
  extraNav.addEventListener('click', event => {
    if (event.target.closest('a')) setOpen(false);
  });

  // Mouse dragging for both rows. A drag is never mistaken for link activation.
  function attachDrag(element) {
    let state = null;
    let suppressClick = false;
    element.addEventListener('pointerdown', event => {
      if (event.pointerType !== 'mouse' || event.button !== 0) return;
      state = { id: event.pointerId, x: event.clientX, y: event.clientY,
        scrollLeft: element.scrollLeft, dragging: false };
      suppressClick = false;
    });
    element.addEventListener('pointermove', event => {
      if (!state || state.id !== event.pointerId) return;
      const dx = event.clientX - state.x;
      const dy = event.clientY - state.y;
      if (!state.dragging) {
        if (Math.abs(dx) < 6 || Math.abs(dx) < Math.abs(dy)) return;
        state.dragging = true;
        suppressClick = true;
        element.dataset.dragging = 'true';
        element.setPointerCapture(event.pointerId);
      }
      element.scrollLeft = state.scrollLeft - dx;
      event.preventDefault();
    });
    function stop(event) {
      if (!state || state.id !== event.pointerId) return;
      if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
      delete element.dataset.dragging;
      state = null;
    }
    element.addEventListener('pointerup', stop);
    element.addEventListener('pointercancel', stop);
    element.addEventListener('click', event => {
      if (!suppressClick) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      suppressClick = false;
    }, true);
  }
  attachDrag(mainNav);
  attachDrag(extraNav);
})();

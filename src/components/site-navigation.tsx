"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { FocusEvent, KeyboardEvent, PointerEvent } from "react";
import { SunOptBrand } from "@/components/sunopt-brand";
import { useDragToScroll } from "@/components/ui/use-drag-to-scroll";

const PRIMARY_ITEMS = [
  { label: "Jahressimulation", href: "/debug/hot-water" },
  { label: "Modellvergleich", href: "/model-comparison" },
  { label: "Tankvergleich", href: "/tank-comparison" },
] as const;
const SECONDARY_ITEMS = [
  { label: "Projektstatus", href: "/project-status" },
  { label: "Beschleunigung A/B", href: "/acceleration-strategy" },
  { label: "Ansatz B", href: "/approach-b" },
  { label: "Präsentation", href: "/presentation" },
  { label: "Codepfad", href: "/documentation/code-path" },
  { label: "Physik", href: "/documentation/physics" },
  { label: "Project Structure", href: "/project-structure.html" },
  { label: "Project Cleanup", href: "/debug/project-cleanup" },
] as const;

type NavItem = { readonly label: string; readonly href: string };
function NavLink({ item, active, onSelect }: {
  item: NavItem;
  active: boolean;
  onSelect?: () => void;
}) {
  if (item.href.endsWith(".html")) {
    return (
      <a href={item.href} className="sunopt-nav-link"
        aria-current={active ? "page" : undefined} onClick={onSelect}>
        {item.label}
      </a>
    );
  }
  return (
    <Link href={item.href} className="sunopt-nav-link"
      aria-current={active ? "page" : undefined} onClick={onSelect}>
      {item.label}
    </Link>
  );
}

/**
 * The three work pages always remain in place. A secondary selection occupies
 * the fourth position (immediately after Tankvergleich). Hover opens a
 * floating palette BELOW this bar: it never inserts a layout row and never
 * pushes the presentation's own slide tabs down.
 */
export function SiteNavigation() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [selectedHref, setSelectedHref] = useState<string | null>(null);
  const headerRef = useRef<HTMLElement>(null);
  const primaryDrag = useDragToScroll<HTMLElement>();
  const secondaryDrag = useDragToScroll<HTMLElement>();
  const secondary = SECONDARY_ITEMS.find((item) => item.href === pathname);
  // Immediately show the chosen item while Next.js performs the transition.
  // Once the path changes, the URL is the source of truth.
  const selected = SECONDARY_ITEMS.find((item) => item.href === selectedHref);
  const contextItem = selected ?? secondary;

  useEffect(() => { setOpen(false); setSelectedHref(null); }, [pathname]);
  function onHeaderPointerEnter(event: PointerEvent<HTMLElement>) {
    if (event.pointerType === "mouse" && event.buttons === 0) setOpen(true);
  }
  function onHeaderPointerLeave(event: PointerEvent<HTMLElement>) {
    if (event.pointerType === "mouse") setOpen(false);
  }
  function onBlur(event: FocusEvent<HTMLElement>) {
    const next = event.relatedTarget;
    if (!next || !headerRef.current?.contains(next as Node)) setOpen(false);
  }
  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      setOpen(false);
      headerRef.current?.querySelector<HTMLElement>(".sunopt-nav-link")?.focus();
    }
  }
  return (
    <header ref={headerRef}
      className="sunopt-site-header sticky top-0 z-50 border-b border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] shadow-sm"
      onPointerEnter={onHeaderPointerEnter} onPointerLeave={onHeaderPointerLeave}
      onFocusCapture={() => setOpen(true)} onBlurCapture={onBlur} onKeyDown={onKeyDown}>
      <div className="relative mx-auto flex min-h-16 max-w-7xl items-center gap-2 px-4 py-3">
        <Link href="/" className="mr-1 shrink-0 rounded-lg px-2 py-1 hover:bg-[var(--sunopt-panel-hover)]"
          aria-label="SunOpt Startseite"><SunOptBrand /></Link>
        <nav aria-label="SunOpt Hauptnavigation" tabIndex={0}
          className="sunopt-drag-scroll flex min-w-0 flex-1 items-center gap-1" {...primaryDrag}>
          {PRIMARY_ITEMS.map((item) => (
            <NavLink key={item.href} item={item} active={pathname === item.href} />
          ))}
          {contextItem && <NavLink key={contextItem.href} item={contextItem}
            active={pathname === contextItem.href} />}
        </nav>
        {/* Icon-only alternative for touch and keyboard; no extra "Weitere Seiten" navigation pill. */}
        <button type="button" className="sunopt-nav-disclosure" aria-label="Seitenauswahl öffnen"
          aria-haspopup="true" aria-controls="sunopt-hover-pages" aria-expanded={open}
          onPointerEnter={(event) => { if (event.pointerType === "mouse") setOpen(true); }}
          onClick={() => setOpen(true)}>
          <span aria-hidden="true">▾</span>
        </button>
      </div>
      <div id="sunopt-hover-pages" className={`sunopt-hover-pages ${open ? "is-open" : ""}`}
        aria-hidden={!open} inert={!open}>
        <div className="mx-auto max-w-7xl px-4 pb-3">
          <nav aria-label="Weitere SunOpt-Seiten" tabIndex={open ? 0 : -1}
            className="sunopt-drag-scroll flex items-center gap-1" {...secondaryDrag}>
            {SECONDARY_ITEMS.map((item) => (
              <NavLink key={item.href} item={item} active={pathname === item.href}
                onSelect={() => { setSelectedHref(item.href); setOpen(false); }} />
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}

export default SiteNavigation;

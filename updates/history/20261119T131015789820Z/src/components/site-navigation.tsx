"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { FocusEvent, KeyboardEvent } from "react";
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
  { label: "Präsentation", href: "/presentation/index.html" },
  { label: "Codepfad", href: "/documentation/code-path" },
  { label: "Physik", href: "/documentation/physics" },
  { label: "Project Structure", href: "/project-structure.html" },
  { label: "Project Cleanup", href: "/debug/project-cleanup" },
] as const;

function NavigationLink({ label, href, active }: {
  label: string; href: string; active: boolean;
}) {
  return (
    <Link className="sunopt-nav-link" href={href} aria-current={active ? "page" : undefined}>
      {label}
    </Link>
  );
}

export function SiteNavigation() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRoot = useRef<HTMLDivElement>(null);
  const lastPointerType = useRef<string>("keyboard");
  const primaryDrag = useDragToScroll<HTMLElement>();
  const secondaryDrag = useDragToScroll<HTMLElement>();
  const secondaryActive = SECONDARY_ITEMS.some((item) => item.href === pathname);

  // Route changes always close the submenu, including browser back/forward.
  useEffect(() => setMenuOpen(false), [pathname]);
  const closeOnBlur = (event: FocusEvent<HTMLDivElement>) => {
    const next = event.relatedTarget;
    if (!next || !menuRoot.current?.contains(next as Node)) setMenuOpen(false);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      setMenuOpen(false);
      menuRoot.current?.querySelector<HTMLButtonElement>("button")?.focus();
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--sunopt-border)] bg-[var(--sunopt-bg)]/95 shadow-sm backdrop-blur">
      <div className="mx-auto flex min-h-16 max-w-7xl items-center gap-2 px-4 py-3">
        <Link href="/" className="mr-1 shrink-0 rounded-lg px-2 py-1 hover:bg-[var(--sunopt-panel-hover)]" aria-label="SunOpt Startseite">
          <SunOptBrand />
        </Link>
        <nav aria-label="SunOpt Hauptnavigation" tabIndex={0}
          className="sunopt-drag-scroll flex min-w-0 flex-1 items-center gap-1"
          {...primaryDrag}
        >
          {PRIMARY_ITEMS.map((item) => (
            <NavigationLink key={item.href} {...item} active={pathname === item.href} />
          ))}
        </nav>
        <div ref={menuRoot} className="relative shrink-0" onPointerLeave={(event) => { if (event.pointerType === "mouse") setMenuOpen(false); }}
          onBlurCapture={closeOnBlur} onKeyDown={onKeyDown}>
          <button type="button" className="sunopt-nav-toggle"
            aria-haspopup="true" aria-expanded={menuOpen} aria-controls="sunopt-secondary-nav"
            onPointerEnter={(event) => { if (event.pointerType === "mouse") setMenuOpen(true); }}
            onClick={(event) => {
              // Hover already opened it for a mouse; keyboard/touch toggle it.
              if (event.detail === 0 || lastPointerType.current !== "mouse") {
                setMenuOpen((value) => !value);
              } else {
                setMenuOpen(true);
              }
            }}
            onPointerDown={(event) => { lastPointerType.current = event.pointerType; }}
          >
            {secondaryActive ? "Weitere Seiten · aktiv" : "Weitere Seiten"}
            <span aria-hidden="true">▾</span>
          </button>
          <div id="sunopt-secondary-nav"
            className={[
              "absolute right-0 top-full z-50 w-[min(90vw,780px)] border border-[var(--sunopt-border)]",
              "rounded-b-xl bg-[var(--sunopt-bg)] p-2 shadow-xl",
              menuOpen ? "visible" : "invisible pointer-events-none",
            ].join(" ")}
            aria-hidden={!menuOpen}
            onMouseEnter={() => setMenuOpen(true)}
          >
            <nav aria-label="SunOpt Dokumentation und Werkzeuge" tabIndex={menuOpen ? 0 : -1}
              className="sunopt-drag-scroll flex min-w-0 items-center gap-1"
              {...secondaryDrag}
            >
              {SECONDARY_ITEMS.map((item) => (
                <NavigationLink key={item.href} {...item} active={pathname === item.href} />
              ))}
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
}

export default SiteNavigation;

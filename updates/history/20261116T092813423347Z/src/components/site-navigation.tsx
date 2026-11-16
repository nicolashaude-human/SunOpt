// src/components/site-navigation.tsx

"use client";

import Link from "next/link";

import {
  usePathname,
} from "next/navigation";

import {
  SunOptBrand,
} from "@/components/sunopt-brand";


const PRIMARY_ITEMS = [
  {
    label:
      "Jahressimulation",
    href:
      "/debug/hot-water",
  },
  {
    label:
      "Modellvergleich",
    href:
      "/model-comparison",
  },
  {
    label:
      "Tankvergleich",
    href:
      "/tank-comparison",
  },
] as const;


const SECONDARY_ITEMS = [
  {
    label:
      "Projektstatus",
    href:
      "/project-status",
  },
  {
    label:
      "Beschleunigung A/B",
    href:
      "/acceleration-strategy",
  },
  {
    label:
      "Präsentation",
    href:
      "/presentation/index.html",
  },
  {
    label:
      "Project Structure",
    href:
      "/project-structure.html",
  },
  {
    label:
      "Project Cleanup",
    href:
      "/debug/project-cleanup",
  },
] as const;


function NavigationLink({
  label,
  href,
  active,
}: {
  label:
    string;
  href:
    string;
  active:
    boolean;
}) {

  return (
    <Link
      href={href}
      className={[
        "shrink-0 rounded-lg px-3 py-2",
        "text-sm font-semibold transition-colors",
        active
          ? "bg-[#ddb114] text-white"
          : "text-[#66543d] hover:bg-[#f3e6d2] hover:text-gray-950",
      ].join(
        " ",
      )}
    >
      {label}
    </Link>
  );
}


export function SiteNavigation() {

  const pathname =
    usePathname();


  return (
    <header
      className="
        group sticky top-0 z-50
        border-b border-[#d9c7aa]
        bg-[#fffaf2]/95 backdrop-blur
      "
    >
      <div
        className="
          mx-auto flex min-h-16 max-w-7xl
          items-center gap-3 px-4 py-3
        "
      >
        <Link
          href="/"
          className="
            mr-2 shrink-0
            rounded-lg px-2 py-1
            hover:bg-[#f3e6d2]
          "
        >
          <SunOptBrand/>
        </Link>

        <nav
          className="
            flex min-w-0 flex-1
            items-center gap-1
            overflow-x-auto
          "
          aria-label="SunOpt Hauptnavigation"
        >
          {PRIMARY_ITEMS.map(
            (
              item,
            ) => (
              <NavigationLink
                key={item.href}
                label={item.label}
                href={item.href}
                active={
                  pathname ===
                  item.href
                }
              />
            ),
          )}
        </nav>
      </div>

      <div
        className="
          pointer-events-none absolute
          left-0 right-0 top-full
          -translate-y-1 opacity-0
          border-b border-[#d9c7aa]
          bg-[#fffaf2]/98 shadow-md
          transition duration-150
          group-hover:pointer-events-auto
          group-hover:translate-y-0
          group-hover:opacity-100
          group-focus-within:pointer-events-auto
          group-focus-within:translate-y-0
          group-focus-within:opacity-100
        "
      >
        <nav
          className="
            mx-auto flex min-h-12 max-w-7xl
            items-center gap-1 overflow-x-auto
            px-4 py-2
          "
          aria-label="SunOpt Dokumentation und Werkzeuge"
        >
          {SECONDARY_ITEMS.map(
            (
              item,
            ) => (
              <NavigationLink
                key={item.href}
                label={item.label}
                href={item.href}
                active={
                  pathname ===
                  item.href
                }
              />
            ),
          )}
        </nav>
      </div>
    </header>
  );
}


export default SiteNavigation;

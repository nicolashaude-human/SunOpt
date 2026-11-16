// src/components/site-navigation.tsx

"use client";

import Link from "next/link";

import {
  usePathname,
} from "next/navigation";

import {
  SunOptBrand,
} from "@/components/sunopt-brand";


const NAVIGATION_ITEMS = [
  
  {
    label: "Hot-Water Debug",
    href: "/debug/hot-water",
  },
  {
    label:
      "Modellvergleich",
    href:
      "/model-comparison",
  },
  {
    label: "Tankvergleich",
    href: "/tank-comparison",
  },
  {
    label:
      "Projektstatus",
    href:
      "/project-status",
  },
  {
    label: "Präsentation",
    href: "/presentation",
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


export function SiteNavigation() {

  const pathname =
    usePathname();


  return (
    <header
      className="sticky top-0 z-50 border-b border-[#d9c7aa] bg-[#fffaf2]/95 backdrop-blur"
    >
      <div
        className="mx-auto flex min-h-16 max-w-7xl items-center gap-3 px-4 py-3"
      >
        <Link
          href="/"
          className="mr-2 shrink-0 rounded-lg px-2 py-1 hover:bg-[#f3e6d2]"
        >
          <SunOptBrand/>
          
        </Link>

        <nav
          className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto"
          aria-label="SunOpt navigation"
        >
          {NAVIGATION_ITEMS.map(
            (
              item,
            ) => {

              const active =
                pathname ===
                item.href;


              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={[
                    "shrink-0 rounded-lg px-3 py-2",
                    "text-sm font-semibold transition-colors",
                    active
                      // #ddb114
                      ? "bg-[#ddb114] text-white"
                      : "text-[#66543d] hover:bg-[#f3e6d2] hover:text-gray-950",
                  ].join(
                    " ",
                  )}
                >
                  {item.label}
                </Link>
              );
            },
          )}
        </nav>
      </div>
    </header>
  );
}


export default SiteNavigation;

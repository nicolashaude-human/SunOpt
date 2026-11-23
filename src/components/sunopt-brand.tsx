// src/components/sunopt-brand.tsx

import Image from "next/image";

import sunOptLogo from "@/app/icon.png";


interface SunOptBrandProps {
  logoSize?: number;
  textSize?: number;
  gap?: number;
  logoOffsetX?: number;
  logoOffsetY?: number;
  textOffsetX?: number;
  textOffsetY?: number;
  className?: string;
}


/*
 * Maximum recommended logoSize in the navigation bar: 40 px.
 * Together with the navigation padding this defines the normal
 * navigation-bar height. Larger values deliberately make the bar taller.
 */
const DEFAULT_LOGO_SIZE =
  55;


  export function SunOptBrand({
    logoSize = DEFAULT_LOGO_SIZE,
    textSize = 30,
    gap = 8,
    logoOffsetX = 0,
    logoOffsetY = -10,
    textOffsetX = -72,
    textOffsetY = 17,
    className = "",
  }: SunOptBrandProps) {
    return (
      <span
        className={[
          // Enforces a strict 40px height while retaining centered baseline equilibrium
          "inline-flex items-center h-[40px] whitespace-nowrap overflow-visible",
          className,
        ].join(" ")}
        style={{
          gap,
        }}
      >
        {/* Logo Container */}
        <span
          className="pl-(logoOffsetX) pb-[20] relative inline-block shrink-0"
          style={{
            width: logoSize,
            height: logoSize,
            left: logoOffsetX,
            top: logoOffsetY, // Visual displacement along the vertical axis
          }}
        >
          <Image
            src={sunOptLogo}
            alt="SunOpt logo"
            fill
            sizes={`${logoSize}px`}
            className="object-contain"
            priority
          />
        </span>
  
        {/* Typography Container */}
        <span
          className="relative inline-block font-bold tracking-tight text-[var(--sunopt-ink)]"
          style={{
            fontSize: textSize,
            lineHeight: 1,
            left: textOffsetX,
            top: textOffsetY, // Visual displacement along the vertical axis
          }}
        >
          SunOpt
        </span>
      </span>
    );
  }


export default SunOptBrand;

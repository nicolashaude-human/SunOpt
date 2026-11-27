import type { HTMLAttributes } from "react";

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      math: HTMLAttributes<Element> & { display?: "block" | "inline" };
      mi: HTMLAttributes<Element>;
      mn: HTMLAttributes<Element>;
      mo: HTMLAttributes<Element>;
      mrow: HTMLAttributes<Element>;
      msub: HTMLAttributes<Element>;
    }
  }
}

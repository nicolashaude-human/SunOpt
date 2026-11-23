// src/app/layout.tsx

import type {
  Metadata,
} from "next";

import "./globals.css";

import {
  SiteNavigation,
} from "@/components/site-navigation";


export const metadata:
  Metadata = {

    title:
      "SunOpt",

    description:
      "PV hot-water storage optimisation",
  };


export default function RootLayout(
  {
    children,
  }: Readonly<{
    children:
      React.ReactNode;
  }>,
) {

  return (
    <html lang="de">

      <head>
        <link rel="stylesheet" href="/styles/sunopt-theme.css" />
      </head>

      <body>

        <SiteNavigation />

        {children}

      </body>

    </html>
  );
}
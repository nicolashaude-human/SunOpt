import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SunOpt · Präsentation",
};

export default function PresentationPage() {
  return (
    <main className="sunopt-presentation-shell">
      <iframe
        src="/presentation/index.html"
        title="SunOpt-Präsentation"
        className="block h-full w-full border-0"
      />
    </main>
  );
}

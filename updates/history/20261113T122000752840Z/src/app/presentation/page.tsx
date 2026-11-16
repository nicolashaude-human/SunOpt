// src/app/presentation/page.tsx

export default function PresentationPage() {
  return (
    <main className="h-[calc(100dvh-65px)] w-full overflow-hidden bg-[#fffaf2] p-3">
      <iframe
        src="/presentation/index.html?v=20261030-presentation-v6"
        title="SunOpt Präsentation"
        className="h-full w-full rounded-xl border border-[#d9c7aa] bg-white"
      />
    </main>
  );
}

// src/app/presentation/page.tsx

export default function PresentationPage() {

  return (
    <main
      className="
        h-[calc(100vh-4rem)]
        min-h-[640px]
        bg-[#fffaf2]
      "
    >
      <iframe
        src="/presentation/index.html"
        title="SunOpt Präsentation"
        className="
          block h-full w-full
          border-0 bg-[#fffaf2]
        "
      />
    </main>
  );
}

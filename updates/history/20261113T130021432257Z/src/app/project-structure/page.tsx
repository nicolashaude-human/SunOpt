// src/app/project-structure/page.tsx

export default function ProjectStructurePage() {

  return (
    <main
      className="
        h-[calc(100vh-4rem)]
        min-h-[640px]
        bg-[#fffaf2]
      "
    >
      <iframe
        src="/project-structure.html"
        title="SunOpt Project Structure"
        className="
          block h-full w-full
          border-0 bg-white
        "
      />
    </main>
  );
}

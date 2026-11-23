import type { ReactNode } from "react";

export function SunOptDocHeader({ label, title, children }: {
  label: string; title: string; children: ReactNode;
}) {
  return (
    <header className="mb-8">
      <p className="sunopt-kicker">{label}</p>
      <h1 className="mt-2 text-3xl font-black tracking-tight text-[var(--sunopt-ink)] sm:text-4xl">{title}</h1>
      <div className="mt-4 max-w-5xl text-base leading-8 text-[var(--sunopt-muted)]">{children}</div>
    </header>
  );
}

export function SunOptPanel({ title, children, className = "" }: {
  title: string; children: ReactNode; className?: string;
}) {
  return (
    <section className={`sunopt-card p-5 shadow-sm sm:p-6 ${className}`}>
      <h2 className="text-xl font-extrabold text-[var(--sunopt-ink)]">{title}</h2>
      <div className="mt-3 leading-7 text-[var(--sunopt-muted)]">{children}</div>
    </section>
  );
}

export function SunOptFlowStep({ number, title, subtitle, description, file }: {
  number: number; title: string; subtitle: string; description: ReactNode; file?: string;
}) {
  return (
    <article className="sunopt-flow-step">
      <div className="sunopt-flow-node">
        <strong>{String(number).padStart(2, "0")} · {title}</strong>
        <span>{subtitle}</span>
        {file && <div className="sunopt-flow-source"><code>{file}</code></div>}
      </div>
      <div className="sunopt-flow-prose">{description}</div>
    </article>
  );
}

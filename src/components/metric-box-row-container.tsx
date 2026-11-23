// src/components/metric-box-row-container.tsx

import type {
  ReactNode,
} from "react";


interface MetricBoxRowContainerProps {
  title:
    string;

  eyebrow?:
    string;

  children:
    ReactNode;

  columnsClassName?:
    string;
}


export function MetricBoxRowContainer({
  title,
  eyebrow,
  children,
  columnsClassName = "sm:grid-cols-2 xl:grid-cols-6",
}: MetricBoxRowContainerProps) {

  return (
    <section
      className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] p-4 shadow-sm"
    >
      <div
        className="mb-4 flex flex-wrap items-baseline justify-between gap-2"
      >
        <h3
          className="text-lg font-bold text-[var(--sunopt-ink)]"
        >
          {title}
        </h3>

        {eyebrow && (
          <div
            className="text-xs font-semibold uppercase tracking-wide text-[var(--sunopt-muted)]"
          >
            {eyebrow}
          </div>
        )}
      </div>

      <div
        className={[
          "grid gap-3",
          columnsClassName,
        ].join(
          " ",
        )}
      >
        {children}
      </div>
    </section>
  );
}


export default MetricBoxRowContainer;

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
      className="rounded-2xl border border-[#dfcfb7] bg-[#fffaf2] p-4 shadow-sm"
    >
      <div
        className="mb-4 flex flex-wrap items-baseline justify-between gap-2"
      >
        <h3
          className="text-lg font-bold text-gray-950"
        >
          {title}
        </h3>

        {eyebrow && (
          <div
            className="text-xs font-semibold uppercase tracking-wide text-[#8b7355]"
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

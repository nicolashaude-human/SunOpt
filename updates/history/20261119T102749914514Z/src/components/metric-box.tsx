// src/components/metric-box.tsx

interface MetricBoxProps {

  label:
    string;

  value:
    string;

  detail?:
    string;

  size?:
    "normal" |
    "large";
}


export function MetricBox({
  label,
  value,
  detail,
  size = "normal",
}: MetricBoxProps) {

  const valueClassName =
    size === "large"
      ? "text-3xl"
      : "text-xl";


  return (
    <div
      className="
        min-w-0 rounded-xl
        border border-[#d8c5a5]
        bg-[#f8eedc]
        px-4 py-3
      "
    >
      <div
        className="
          text-xs font-semibold
          uppercase tracking-wide
          text-[#76644d]
        "
      >
        {label}
      </div>

      <div
        className={`
          mt-1 font-bold
          tabular-nums
          text-gray-950
          ${valueClassName}
        `}
      >
        {value}
      </div>

      {detail && (
        <div
          className="
            mt-1 text-xs
            leading-5
            text-[#76644d]
          "
        >
          {detail}
        </div>
      )}
    </div>
  );
}


export default MetricBox;
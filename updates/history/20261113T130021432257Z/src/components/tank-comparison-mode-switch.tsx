// src/components/tank-comparison-mode-switch.tsx

"use client";


export type TankComparisonCausalPath =
  | "A"
  | "B";


interface TankComparisonModeSwitchProps {
  value:
    TankComparisonCausalPath;

  disabled?:
    boolean;

  onChange:
    (
      value:
        TankComparisonCausalPath,
    ) => void;
}


const MODES:
  readonly TankComparisonCausalPath[] = [
    "A",
    "B",
  ];


export function TankComparisonModeSwitch({
  value,
  disabled =
    false,
  onChange,
}: TankComparisonModeSwitchProps) {

  return (
    <div
      className="
        ml-auto flex shrink-0
        items-center justify-end gap-2
      "
      role="group"
      aria-label="CUDA causal path"
    >
      {MODES.map(
        (
          mode,
        ) => {

          const active =
            value ===
            mode;


          return (
            <button
              key={mode}
              type="button"
              disabled={
                disabled
              }
              onClick={
                () =>
                  onChange(
                    mode,
                  )
              }
              aria-pressed={
                active
              }
              title={
                mode ===
                "A"
                  ? "Pfad A – FP64 Referenz"
                  : "Pfad B – FP32 Produktion"
              }
              className={[
                "flex h-10 min-w-12 items-center justify-center",
                "rounded-lg border-2 bg-gray-950 px-4",
                "text-sm font-bold text-white",
                "shadow-sm transition",
                "hover:bg-gray-800",
                "disabled:cursor-not-allowed disabled:opacity-50",
                active
                  ? [
                      "border-green-500",
                      "shadow-[0_0_10px_rgba(34,197,94,0.55)]",
                    ].join(
                      " ",
                    )
                  : "border-transparent",
              ].join(
                " ",
              )}
            >
              {mode}
            </button>
          );
        },
      )}
    </div>
  );
}


export default TankComparisonModeSwitch;

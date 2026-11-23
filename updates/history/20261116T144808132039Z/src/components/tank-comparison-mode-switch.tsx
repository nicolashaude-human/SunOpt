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
      aria-label="Tankvergleich Rechenpfad"
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
                  ? "Approach A – optimierter exakter FP64 CUDA-Lockstep"
                  : "Approach B – B14P FP32 Produktionssolver"
              }
              className={[
                "flex h-11 min-w-14 items-center justify-center",
                "rounded-lg border-4 bg-black px-4",
                "text-sm font-bold text-white",
                "transition",
                "hover:bg-gray-800",
                "disabled:cursor-not-allowed disabled:opacity-50",
                active
                  ? [
                      "border-green-500",
                      "shadow-[0_0_12px_rgba(34,197,94,0.65)]",
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

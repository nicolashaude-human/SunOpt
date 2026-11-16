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
        flex shrink-0 items-center gap-2
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
            <div
              key={mode}
              className={[
                "rounded-xl p-[2px] transition-shadow",
                "bg-green-500",
                active
                  ? "shadow-[0_0_14px_rgba(34,197,94,0.55)]"
                  : "shadow-sm",
              ].join(
                " ",
              )}
            >
              <button
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
                  "rounded-[10px] px-4",
                  "text-sm font-bold transition",
                  "disabled:cursor-not-allowed disabled:opacity-50",
                  active
                    ? "bg-green-50 text-green-950"
                    : "bg-white text-gray-800 hover:bg-green-50",
                ].join(
                  " ",
                )}
              >
                {mode}
              </button>
            </div>
          );
        },
      )}
    </div>
  );
}


export default TankComparisonModeSwitch;

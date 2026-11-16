// src/domain/default-hot-water-profile.ts

import type {
  HotWaterProfile,
} from "./hot-water";

export const DEFAULT_HOT_WATER_PROFILE: HotWaterProfile = {
  hourlyFractions: [
    0.01, // 00
    0.00, // 01
    0.00, // 02
    0.00, // 03
    0.00, // 04
    0.02, // 05

    0.08, // 06
    0.12, // 07
    0.08, // 08
    0.04, // 09
    0.03, // 10
    0.03, // 11

    0.04, // 12
    0.03, // 13
    0.03, // 14
    0.03, // 15
    0.04, // 16
    0.06, // 17

    0.09, // 18
    0.10, // 19
    0.08, // 20
    0.05, // 21
    0.03, // 22
    0.01, // 23
  ],
};
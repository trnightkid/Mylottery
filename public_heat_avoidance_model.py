#!/usr/bin/env python3
"""
Public heat avoidance model for Double Color Ball experiments.

This module treats the "avoid public picks" idea as a testable hypothesis:
first estimate what numbers the public may over-buy, then favor numbers that
remain structurally normal while having lower public heat.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


DATA_FILE = "lottery_data.csv"
OUTPUT_DIR = "output"
RED_COLS = ["red1", "red2", "red3", "red4", "red5", "red6"]
RED_NUMBERS = range(1, 34)
BLUE_NUMBERS = range(1, 17)


@dataclass(frozen=True)
class ModelWeights:
    public_heat_inverse: float = 0.30
    structure: float = 0.22
    neighbor_shift: float = 0.18
    omission: float = 0.15
    medium_frequency: float = 0.10
    noise: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "public_heat_inverse": self.public_heat_inverse,
            "structure": self.structure,
            "neighbor_shift": self.neighbor_shift,
            "omission": self.omission,
            "medium_frequency": self.medium_frequency,
            "noise": self.noise,
        }


class PublicHeatAvoidanceModel:
    def __init__(self, df: pd.DataFrame, weights: ModelWeights | None = None, seed: int = 20260715):
        self.df = self._prepare(df)
        self.weights = weights or ModelWeights()
        self.random = random.Random(seed)

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        required = ["period", *RED_COLS, "blue"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        clean = df.copy()
        for col in required:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna(subset=required)
        clean[required] = clean[required].astype(int)
        return clean.sort_values("period", ascending=True).reset_index(drop=True)

    def _recent_rows(self, end_index: int | None = None, window: int = 10) -> pd.DataFrame:
        if end_index is None:
            end_index = len(self.df)
        start = max(0, end_index - window)
        return self.df.iloc[start:end_index]

    @staticmethod
    def _normalize(values: Dict[int, float], default: float = 0.0) -> Dict[int, float]:
        if not values:
            return {}
        lo = min(values.values())
        hi = max(values.values())
        if math.isclose(lo, hi):
            return {k: default for k in values}
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}

    @staticmethod
    def _red_set(row: pd.Series) -> set[int]:
        return {int(row[col]) for col in RED_COLS}

    def red_frequency(self, end_index: int | None = None, window: int = 10) -> Dict[int, int]:
        rows = self._recent_rows(end_index, window)
        counter: Counter[int] = Counter()
        for _, row in rows.iterrows():
            counter.update(int(row[col]) for col in RED_COLS)
        return {num: counter.get(num, 0) for num in RED_NUMBERS}

    def blue_frequency(self, end_index: int | None = None, window: int = 20) -> Dict[int, int]:
        rows = self._recent_rows(end_index, window)
        counter = Counter(int(v) for v in rows["blue"].tolist())
        return {num: counter.get(num, 0) for num in BLUE_NUMBERS}

    def omission(self, end_index: int | None = None) -> Dict[int, int]:
        if end_index is None:
            end_index = len(self.df)
        rows = self.df.iloc[:end_index]
        result = {}
        for num in RED_NUMBERS:
            miss = 0
            for _, row in rows.iloc[::-1].iterrows():
                if num in self._red_set(row):
                    break
                miss += 1
            result[num] = miss
        return result

    def blue_omission(self, end_index: int | None = None) -> Dict[int, int]:
        if end_index is None:
            end_index = len(self.df)
        rows = self.df.iloc[:end_index]
        result = {}
        for num in BLUE_NUMBERS:
            miss = 0
            for blue in rows["blue"].iloc[::-1]:
                if int(blue) == num:
                    break
                miss += 1
            result[num] = miss
        return result

    def neighbor_shift(self, end_index: int | None = None, window: int = 10) -> Dict[int, float]:
        freq = self.red_frequency(end_index, window)
        scores = {num: 0.0 for num in RED_NUMBERS}

        for hot_num, count in freq.items():
            if count <= 0:
                continue
            for distance, weight in ((1, 1.0), (2, 0.55), (10, 0.35)):
                for candidate in (hot_num - distance, hot_num + distance):
                    if candidate in scores:
                        scores[candidate] += count * weight

            tail = hot_num % 10
            for candidate in RED_NUMBERS:
                if candidate != hot_num and candidate % 10 == tail:
                    scores[candidate] += count * 0.22

        return self._normalize(scores)

    def structure_score(self, end_index: int | None = None, window: int = 200) -> Dict[int, float]:
        rows = self._recent_rows(end_index, window)
        freq = self.red_frequency(end_index, window)
        norm_freq = self._normalize({k: float(v) for k, v in freq.items()}, default=0.5)

        region_counts = {"low": 0, "mid": 0, "high": 0}
        for _, row in rows.iterrows():
            for col in RED_COLS:
                value = int(row[col])
                if value <= 11:
                    region_counts["low"] += 1
                elif value <= 22:
                    region_counts["mid"] += 1
                else:
                    region_counts["high"] += 1

        total = sum(region_counts.values()) or 1
        region_ratio = {k: v / total for k, v in region_counts.items()}

        scores = {}
        for num in RED_NUMBERS:
            if num <= 11:
                region = region_ratio["low"]
            elif num <= 22:
                region = region_ratio["mid"]
            else:
                region = region_ratio["high"]
            region_fit = min(region / (1 / 3), 1.4) / 1.4
            scores[num] = norm_freq[num] * 0.65 + region_fit * 0.35
        return scores

    def public_heat_score(self, end_index: int | None = None) -> Dict[int, float]:
        freq5 = self._normalize({k: float(v) for k, v in self.red_frequency(end_index, 5).items()})
        freq10 = self._normalize({k: float(v) for k, v in self.red_frequency(end_index, 10).items()})
        freq20 = self._normalize({k: float(v) for k, v in self.red_frequency(end_index, 20).items()})
        neighbor = self.neighbor_shift(end_index, 10)

        scores = {}
        lucky = {6, 8, 9, 16, 18, 28, 29}
        for num in RED_NUMBERS:
            birthday_bias = 1.0 if num <= 31 else 0.15
            calendar_bias = 1.0 if num <= 12 else 0.45 if num <= 24 else 0.20
            lucky_bias = 1.0 if num in lucky else 0.0
            round_bias = 0.7 if num in {10, 20, 30} else 0.0

            scores[num] = (
                freq5[num] * 0.22
                + freq10[num] * 0.18
                + freq20[num] * 0.12
                + neighbor[num] * 0.18
                + birthday_bias * 0.12
                + calendar_bias * 0.08
                + lucky_bias * 0.06
                + round_bias * 0.04
            )
        return self._normalize(scores)

    def red_scores(self, end_index: int | None = None) -> Dict[int, Dict[str, float]]:
        public_heat = self.public_heat_score(end_index)
        structure = self.structure_score(end_index)
        neighbor = self.neighbor_shift(end_index)
        omission = self._normalize({k: float(v) for k, v in self.omission(end_index).items()})
        medium = self._normalize({k: float(v) for k, v in self.red_frequency(end_index, 50).items()})

        result = {}
        for num in RED_NUMBERS:
            draw_score = (
                (1 - public_heat[num]) * self.weights.public_heat_inverse
                + structure[num] * self.weights.structure
                + neighbor[num] * self.weights.neighbor_shift
                + omission[num] * self.weights.omission
                + medium[num] * self.weights.medium_frequency
                + self.random.random() * self.weights.noise
            )
            result[num] = {
                "draw_score": draw_score,
                "public_heat": public_heat[num],
                "structure": structure[num],
                "neighbor_shift": neighbor[num],
                "omission": omission[num],
                "medium_frequency": medium[num],
            }
        return result

    def blue_scores(self, end_index: int | None = None) -> Dict[int, Dict[str, float]]:
        freq10 = self._normalize({k: float(v) for k, v in self.blue_frequency(end_index, 10).items()})
        freq50 = self._normalize({k: float(v) for k, v in self.blue_frequency(end_index, 50).items()})
        omission = self._normalize({k: float(v) for k, v in self.blue_omission(end_index).items()})

        result = {}
        for num in BLUE_NUMBERS:
            public_heat = freq10[num] * 0.45 + freq50[num] * 0.25 + (1.0 if num in {6, 8, 9, 16} else 0.0) * 0.30
            draw_score = (1 - public_heat) * 0.45 + omission[num] * 0.30 + freq50[num] * 0.20 + self.random.random() * 0.05
            result[num] = {
                "draw_score": draw_score,
                "public_heat": public_heat,
                "omission": omission[num],
                "medium_frequency": freq50[num],
            }
        return result

    @staticmethod
    def _consecutive_pairs(combo: Sequence[int]) -> int:
        values = sorted(combo)
        return sum(1 for a, b in zip(values, values[1:]) if b - a == 1)

    @staticmethod
    def _shape_penalty(combo: Sequence[int]) -> float:
        values = sorted(combo)
        odd = sum(1 for value in values if value % 2)
        low = sum(1 for value in values if value <= 11)
        mid = sum(1 for value in values if 12 <= value <= 22)
        high = 6 - low - mid
        total = sum(values)
        consecutive = PublicHeatAvoidanceModel._consecutive_pairs(values)

        penalty = 0.0
        if odd in {0, 1, 5, 6}:
            penalty += 0.18
        if 0 in {low, mid, high}:
            penalty += 0.16
        if total < 70 or total > 140:
            penalty += 0.12
        if consecutive > 2:
            penalty += 0.10
        return penalty

    @staticmethod
    def _shape_bonus(combo: Sequence[int]) -> float:
        consecutive = PublicHeatAvoidanceModel._consecutive_pairs(combo)

        # Consecutive numbers are common enough that the model should not erase
        # them. One pair is healthy; two pairs are acceptable; only excessive
        # chains are handled by _shape_penalty.
        if consecutive == 1:
            return 0.035
        if consecutive == 2:
            return 0.018
        return 0.0

    def generate_predictions(
        self,
        end_index: int | None = None,
        groups: int = 5,
        candidate_pool_size: int = 22,
    ) -> Dict[str, object]:
        red_scores = self.red_scores(end_index)
        blue_scores = self.blue_scores(end_index)
        sorted_red = sorted(red_scores.items(), key=lambda item: item[1]["draw_score"], reverse=True)
        pool = [num for num, _ in sorted_red[:candidate_pool_size]]

        combos: List[Tuple[float, Tuple[int, ...]]] = []
        for combo in combinations(pool, 6):
            score = sum(red_scores[num]["draw_score"] for num in combo) / 6
            heat = sum(red_scores[num]["public_heat"] for num in combo) / 6
            score = score + self._shape_bonus(combo) - self._shape_penalty(combo) - heat * 0.08
            combos.append((score, tuple(sorted(combo))))

        combos.sort(reverse=True, key=lambda item: item[0])
        selected = []
        seen = set()
        usage = defaultdict(int)
        max_scan = min(len(combos), 12000)
        combo_candidates = combos[:max_scan]

        while len(selected) < groups and combo_candidates:
            best_index = None
            best_adjusted = None
            best_payload = None

            for idx, (score, combo) in enumerate(combo_candidates):
                if combo in seen:
                    continue
                overlap_penalty = sum(usage[num] for num in combo) * 0.055
                duplicate_pair_penalty = 0.0
                for item in selected:
                    overlap = len(set(combo) & set(item["red"]))
                    if overlap > 2:
                        duplicate_pair_penalty += (overlap - 2) * 0.075
                adjusted = score - overlap_penalty - duplicate_pair_penalty
                if best_adjusted is None or adjusted > best_adjusted:
                    best_index = idx
                    best_adjusted = adjusted
                    best_payload = (score, combo)

            if best_index is None or best_payload is None:
                break

            score, combo = best_payload
            selected.append({
                "red": list(combo),
                "combo_score": score,
                "consecutive_pairs": self._consecutive_pairs(combo),
            })
            seen.add(combo)
            for num in combo:
                usage[num] += 1
            combo_candidates.pop(best_index)

        sorted_blue = sorted(blue_scores.items(), key=lambda item: item[1]["draw_score"], reverse=True)
        for idx, item in enumerate(selected):
            item["blue"] = sorted_blue[idx % len(sorted_blue)][0]

        return {
            "red_scores": red_scores,
            "blue_scores": blue_scores,
            "public_hot_reds": [num for num, _ in sorted(red_scores.items(), key=lambda item: item[1]["public_heat"], reverse=True)[:10]],
            "avoidance_candidates": pool,
            "predictions": selected,
        }

    @staticmethod
    def dantuo_bet_count(dan_count: int, tuo_count: int, blue_count: int) -> int:
        if dan_count < 0 or dan_count > 5:
            return 0
        need_from_tuo = 6 - dan_count
        if tuo_count < need_from_tuo or blue_count < 1:
            return 0
        return math.comb(tuo_count, need_from_tuo) * blue_count

    @classmethod
    def dantuo_cost(cls, dan_count: int, tuo_count: int, blue_count: int) -> int:
        return cls.dantuo_bet_count(dan_count, tuo_count, blue_count) * 2

    def generate_purchase_plans(
        self,
        budget: int = 100,
        max_plans: int = 8,
        candidate_pool_size: int = 24,
        end_index: int | None = None,
    ) -> List[Dict[str, object]]:
        current = self.generate_predictions(groups=5, candidate_pool_size=candidate_pool_size, end_index=end_index)
        red_scores = current["red_scores"]
        blue_scores = current["blue_scores"]

        red_rank = [num for num, _ in sorted(red_scores.items(), key=lambda item: item[1]["draw_score"], reverse=True)]
        low_public_rank = [num for num, _ in sorted(red_scores.items(), key=lambda item: item[1]["public_heat"])]
        omission_rank = [num for num, _ in sorted(red_scores.items(), key=lambda item: item[1]["omission"], reverse=True)]
        neighbor_rank = [num for num, _ in sorted(red_scores.items(), key=lambda item: item[1]["neighbor_shift"], reverse=True)]
        blue_rank = [num for num, _ in sorted(blue_scores.items(), key=lambda item: item[1]["draw_score"], reverse=True)]
        blue_low_rank = [num for num, _ in sorted(blue_scores.items(), key=lambda item: item[1]["public_heat"])]
        blue_omission_rank = [num for num, _ in sorted(blue_scores.items(), key=lambda item: item[1]["omission"], reverse=True)]

        plans = []
        ranking_sources = [
            red_rank,
            neighbor_rank,
            low_public_rank,
            omission_rank,
            current["avoidance_candidates"],
        ]

        def add_unique(target: List[int], source: Iterable[int], limit: int) -> None:
            for num in source:
                if num not in target:
                    target.append(num)
                if len(target) >= limit:
                    break

        def build_layered_pool(total_count: int, offset: int = 0) -> List[int]:
            pool: List[int] = []
            quotas = [max(1, total_count // 4), max(1, total_count // 4), max(1, total_count // 4)]
            quotas.append(total_count - sum(quotas))
            for source, quota in zip(ranking_sources, quotas):
                rotated = source[offset:] + source[:offset]
                add_unique(pool, rotated, len(pool) + quota)
            add_unique(pool, current["avoidance_candidates"], total_count)
            add_unique(pool, red_rank, total_count)
            return pool[:total_count]

        def append_plan(dans: List[int], tuos: List[int], blues: List[int], plan_type: str) -> None:
            dan_count = len(dans)
            tuo_count = len(tuos)
            blue_count = len(blues)
            if dan_count == 0:
                if tuo_count < 6:
                    return
                bet_count = math.comb(tuo_count, 6) * blue_count
            else:
                bet_count = self.dantuo_bet_count(dan_count, tuo_count, blue_count)
            cost = bet_count * 2
            if bet_count <= 0 or cost > budget:
                return

            red_pool = list(dict.fromkeys([*dans, *tuos]))
            avg_dan = sum(red_scores[num]["draw_score"] for num in dans) / len(dans) if dans else 0.0
            avg_tuo = sum(red_scores[num]["draw_score"] for num in tuos) / len(tuos)
            avg_blue = sum(blue_scores[num]["draw_score"] for num in blues) / len(blues)
            avg_public_heat = sum(red_scores[num]["public_heat"] for num in red_pool) / len(red_pool)
            consecutive_capacity = self._consecutive_pairs(sorted(red_pool))
            regions = [
                sum(1 for num in red_pool if num <= 11),
                sum(1 for num in red_pool if 12 <= num <= 22),
                sum(1 for num in red_pool if num >= 23),
            ]
            balance_bonus = 0.0
            if min(regions) > 0:
                balance_bonus += 0.10
            if consecutive_capacity >= 1:
                balance_bonus += 0.06
            if dan_count <= 1:
                balance_bonus += 0.08
            if "segment" in plan_type:
                balance_bonus += 0.22
            if "late_segment" in plan_type:
                balance_bonus += 0.30
            if blue_count >= 3:
                balance_bonus += 0.08
            if blue_count >= 4:
                balance_bonus += 0.04
            if any(blue in {1, 2, 3, 4} for blue in blues):
                balance_bonus += 0.08
            if len(red_pool) >= 8:
                balance_bonus += 0.20
            if len(red_pool) >= 9:
                balance_bonus += 0.08
            if len(red_pool) < 8 and blue_count > 4:
                balance_bonus -= 0.28
            if len(red_pool) >= 8 and 3 <= blue_count <= 4:
                balance_bonus += 0.24
            if blue_count == 5 and len(red_pool) == 7:
                balance_bonus += 0.42
            if blue_count < 4:
                balance_bonus -= 0.45
            if blue_count > 5:
                balance_bonus -= 0.08

            coverage_ratio = len(red_pool) / 33
            blue_coverage_ratio = blue_count / 16
            efficiency = (
                avg_dan * 0.08
                + avg_tuo * 0.18
                + avg_blue * 0.12
                + math.log1p(bet_count) * 0.10
                + coverage_ratio * 0.26
                + blue_coverage_ratio * 0.12
                + (1 - avg_public_heat) * 0.10
                + balance_bonus
            ) / max(1.0, cost / 85)

            plans.append({
                "type": plan_type,
                "dan": sorted(dans),
                "tuo": sorted(tuos),
                "blue": sorted(blues),
                "dan_count": dan_count,
                "tuo_count": tuo_count,
                "blue_count": blue_count,
                "bet_count": bet_count,
                "cost": cost,
                "efficiency_score": efficiency,
                "red_pool_consecutive_pairs": consecutive_capacity,
                "region_counts": {
                    "low_01_11": regions[0],
                    "mid_12_22": regions[1],
                    "high_23_33": regions[2],
                },
            })

        def blue_options(count: int) -> List[List[int]]:
            options: List[List[int]] = []
            sources = [
                blue_rank,
                blue_omission_rank,
                blue_low_rank,
                [1, 2, 3, 4, *blue_rank],
            ]
            for source in sources:
                values: List[int] = []
                add_unique(values, source, count)
                if len(values) == count and values not in options:
                    options.append(values)
            return options

        # Red full-entry plans: no dan risk, useful when candidate-pool coverage is
        # better than top-score certainty.
        for total_red_count in range(7, 10):
            for offset in range(4):
                red_pool = build_layered_pool(total_red_count, offset=offset)
                for blue_count in range(1, min(7, len(blue_rank)) + 1):
                    for blues in blue_options(blue_count):
                        append_plan([], red_pool, blues, "layered_red_full_blue_multi")

        avoidance = current["avoidance_candidates"]
        for total_red_count in range(7, 10):
            for offset in (0, 4, 8, 12, 15):
                segment = avoidance[offset: offset + total_red_count]
                if len(segment) < total_red_count:
                    segment = segment + avoidance[: total_red_count - len(segment)]
                segment = list(dict.fromkeys(segment))
                add_unique(segment, low_public_rank, total_red_count)
                add_unique(segment, omission_rank, total_red_count)
                plan_type = "candidate_late_segment_red_full_blue_multi" if offset >= 12 else "candidate_segment_red_full_blue_multi"
                for blue_count in range(1, min(7, len(blue_rank)) + 1):
                    for blues in blue_options(blue_count):
                        append_plan([], segment[:total_red_count], blues, plan_type)

        dan_options = [
            red_rank[:1],
            red_rank[:2],
            [red_rank[0], red_rank[2]] if len(red_rank) > 2 else red_rank[:2],
            [red_rank[0], low_public_rank[0]],
        ]

        for dans in dan_options:
            dans = list(dict.fromkeys(dans))
            if not dans or len(dans) > 5:
                continue
            for total_red_count in range(max(8, len(dans) + 5), min(candidate_pool_size, 18) + 1):
                red_pool = list(dans)
                target_tuo = total_red_count - len(dans)
                high_quota = max(2, target_tuo // 3)
                mid_quota = max(2, target_tuo // 3)
                low_quota = target_tuo - high_quota - mid_quota

                add_unique(red_pool, red_rank[:12], len(dans) + high_quota)
                add_unique(red_pool, neighbor_rank[:18], len(dans) + high_quota + mid_quota)
                add_unique(red_pool, low_public_rank[:24], len(dans) + high_quota + mid_quota + max(1, low_quota))
                add_unique(red_pool, omission_rank[:24], total_red_count)
                add_unique(red_pool, red_rank, total_red_count)

                red_pool = red_pool[:total_red_count]
                tuos = [num for num in red_pool if num not in dans]
                if len(tuos) < 6 - len(dans):
                    continue

                for blue_count in range(1, min(8, len(blue_rank)) + 1):
                    for blues in blue_options(blue_count):
                        append_plan(dans, tuos, blues, "layered_red_dantuo_blue_multi")

        plans.sort(key=lambda item: (item["efficiency_score"], item["bet_count"]), reverse=True)
        diverse = []
        seen_shapes = set()
        for plan in plans:
            red_signature = tuple(plan["dan"] + plan["tuo"])
            shape = (plan["type"], plan["dan_count"], plan["tuo_count"], plan["blue_count"], plan["cost"], red_signature)
            if shape in seen_shapes:
                continue
            red_pool = set(plan["dan"]) | set(plan["tuo"])
            if diverse:
                max_overlap = max(len(red_pool & (set(item["dan"]) | set(item["tuo"]))) for item in diverse)
                if max_overlap >= len(red_pool) - 1:
                    continue
            seen_shapes.add(shape)
            diverse.append(plan)
            if len(diverse) >= max_plans:
                break
        return diverse

    def backtest(
        self,
        periods: int = 200,
        groups: int = 5,
        warmup: int = 80,
        candidate_pool_size: int = 22,
    ) -> Dict[str, object]:
        max_periods = max(0, len(self.df) - warmup)
        periods = min(periods, max_periods)
        if periods <= 0:
            raise ValueError("Not enough historical rows for backtest")

        start = len(self.df) - periods
        details = []
        hit_counter: Counter[int] = Counter()
        best_group_hits = []
        candidate_hits = []
        random_candidate_hits = []
        random_best_hits = []

        for index in range(start, len(self.df)):
            actual_row = self.df.iloc[index]
            actual_reds = self._red_set(actual_row)
            actual_blue = int(actual_row["blue"])

            prediction = self.generate_predictions(end_index=index, groups=groups, candidate_pool_size=candidate_pool_size)
            candidate_pool = set(prediction["avoidance_candidates"])
            candidate_hits.append(len(candidate_pool & actual_reds))
            random_candidate_pool = set(self.random.sample(list(RED_NUMBERS), candidate_pool_size))
            random_candidate_hits.append(len(random_candidate_pool & actual_reds))

            group_hits = []
            blue_hit = False
            for group in prediction["predictions"]:
                red_hit = len(set(group["red"]) & actual_reds)
                group_hits.append(red_hit)
                blue_hit = blue_hit or int(group["blue"]) == actual_blue

            best_hit = max(group_hits) if group_hits else 0
            hit_counter[best_hit] += 1
            best_group_hits.append(best_hit)

            random_hits = []
            for _ in range(groups):
                random_group = set(self.random.sample(list(RED_NUMBERS), 6))
                random_hits.append(len(random_group & actual_reds))
            random_best_hits.append(max(random_hits))

            details.append({
                "period": int(actual_row["period"]),
                "actual_red": sorted(actual_reds),
                "actual_blue": actual_blue,
                "best_red_hit": best_hit,
                "group_hits": group_hits,
                "candidate_pool_hit": candidate_hits[-1],
                "blue_hit": blue_hit,
            })

        avg_best = sum(best_group_hits) / len(best_group_hits)
        avg_random_best = sum(random_best_hits) / len(random_best_hits)
        avg_candidate = sum(candidate_hits) / len(candidate_hits)
        avg_random_candidate = sum(random_candidate_hits) / len(random_candidate_hits)

        return {
            "periods": periods,
            "groups_per_period": groups,
            "candidate_pool_size": candidate_pool_size,
            "avg_best_red_hit": avg_best,
            "avg_random_best_red_hit": avg_random_best,
            "avg_candidate_pool_hit": avg_candidate,
            "avg_random_candidate_pool_hit": avg_random_candidate,
            "hit_distribution": {str(k): hit_counter.get(k, 0) for k in range(7)},
            "three_plus_rate": sum(1 for h in best_group_hits if h >= 3) / len(best_group_hits),
            "four_plus_rate": sum(1 for h in best_group_hits if h >= 4) / len(best_group_hits),
            "random_three_plus_rate": sum(1 for h in random_best_hits if h >= 3) / len(random_best_hits),
            "recent_details": details[-20:],
        }

    def backtest_purchase_plans(
        self,
        periods: int = 120,
        budget: int = 100,
        max_plans: int = 5,
        candidate_pool_size: int = 24,
        warmup: int = 80,
    ) -> Dict[str, object]:
        max_periods = max(0, len(self.df) - warmup)
        periods = min(periods, max_periods)
        if periods <= 0:
            raise ValueError("Not enough historical rows for purchase plan backtest")

        start = len(self.df) - periods
        details = []
        best_red_pool_hits = []
        best_dan_hits = []
        blue_hits = []
        advance_path_hits = []
        blue_or_four_plus_hits = []
        full_red_cover = 0
        actual_combo_cover = 0
        total_costs = []

        for index in range(start, len(self.df)):
            actual_row = self.df.iloc[index]
            actual_reds = self._red_set(actual_row)
            actual_blue = int(actual_row["blue"])
            plans = self.generate_purchase_plans(
                budget=budget,
                max_plans=max_plans,
                candidate_pool_size=candidate_pool_size,
                end_index=index,
            )
            if not plans:
                continue
            selected_plans = []
            spent = 0
            for plan in plans:
                if spent + int(plan["cost"]) <= budget:
                    selected_plans.append(plan)
                    spent += int(plan["cost"])
                if len(selected_plans) >= max_plans:
                    break
            if not selected_plans:
                selected_plans = [plans[0]]

            plan_results = []
            for plan in selected_plans:
                red_pool = set(plan["dan"]) | set(plan["tuo"])
                dan_set = set(plan["dan"])
                blue_set = set(plan["blue"])
                red_pool_hit = len(red_pool & actual_reds)
                dan_hit = len(dan_set & actual_reds)
                blue_hit = actual_blue in blue_set
                red_cover = actual_reds <= red_pool
                combo_cover = red_cover and blue_hit
                plan_results.append({
                    "cost": plan["cost"],
                    "bet_count": plan["bet_count"],
                    "dan": plan["dan"],
                    "tuo": plan["tuo"],
                    "blue": plan["blue"],
                    "red_pool_hit": red_pool_hit,
                    "dan_hit": dan_hit,
                    "blue_hit": blue_hit,
                    "red_cover": red_cover,
                    "combo_cover": combo_cover,
                })

            best_by_pool = max(plan_results, key=lambda item: (item["red_pool_hit"], item["blue_hit"]))
            best_red_pool_hits.append(best_by_pool["red_pool_hit"])
            best_dan_hits.append(max(item["dan_hit"] for item in plan_results))
            any_blue_hit = any(item["blue_hit"] for item in plan_results)
            any_four_plus = any(item["red_pool_hit"] >= 4 for item in plan_results)
            blue_hits.append(any_blue_hit)
            advance_path_hits.append(any_blue_hit or any_four_plus)
            blue_or_four_plus_hits.append({
                "blue": any_blue_hit,
                "four_plus": any_four_plus,
            })
            full_red_cover += 1 if any(item["red_cover"] for item in plan_results) else 0
            actual_combo_cover += 1 if any(item["combo_cover"] for item in plan_results) else 0
            total_costs.append(sum(item["cost"] for item in plan_results))

            details.append({
                "period": int(actual_row["period"]),
                "actual_red": sorted(actual_reds),
                "actual_blue": actual_blue,
                "best_red_pool_hit": best_by_pool["red_pool_hit"],
                "best_dan_hit": max(item["dan_hit"] for item in plan_results),
                "any_blue_hit": any_blue_hit,
                "any_four_plus_red_pool": any_four_plus,
                "advance_path_hit": any_blue_hit or any_four_plus,
                "any_full_red_cover": any(item["red_cover"] for item in plan_results),
                "any_combo_cover": any(item["combo_cover"] for item in plan_results),
                "plans": plan_results,
            })

        total = len(details) or 1
        return {
            "periods": len(details),
            "budget": budget,
            "plans_per_period": max_plans,
            "avg_total_cost": sum(total_costs) / total if total_costs else 0,
            "avg_best_red_pool_hit": sum(best_red_pool_hits) / total,
            "avg_best_dan_hit": sum(best_dan_hits) / total,
            "blue_hit_rate": sum(1 for hit in blue_hits if hit) / total,
            "advance_path_rate": sum(1 for hit in advance_path_hits if hit) / total,
            "full_red_cover_rate": full_red_cover / total,
            "combo_cover_rate": actual_combo_cover / total,
            "three_plus_pool_rate": sum(1 for hit in best_red_pool_hits if hit >= 3) / total,
            "four_plus_pool_rate": sum(1 for hit in best_red_pool_hits if hit >= 4) / total,
            "blue_and_four_plus_rate": sum(1 for item in blue_or_four_plus_hits if item["blue"] and item["four_plus"]) / total,
            "recent_details": details[-20:],
        }

    @staticmethod
    def _random_weights(rng: random.Random) -> ModelWeights:
        raw = [rng.uniform(0.05, 0.45) for _ in range(5)]
        total = sum(raw)
        scaled = [value / total * 0.95 for value in raw]
        return ModelWeights(
            public_heat_inverse=scaled[0],
            structure=scaled[1],
            neighbor_shift=scaled[2],
            omission=scaled[3],
            medium_frequency=scaled[4],
            noise=0.05,
        )

    @staticmethod
    def _mutate_weights(weights: ModelWeights, rng: random.Random, rate: float = 0.18) -> ModelWeights:
        values = [
            weights.public_heat_inverse,
            weights.structure,
            weights.neighbor_shift,
            weights.omission,
            weights.medium_frequency,
        ]
        mutated = [max(0.02, value + rng.uniform(-rate, rate)) for value in values]
        total = sum(mutated)
        scaled = [value / total * 0.95 for value in mutated]
        return ModelWeights(
            public_heat_inverse=scaled[0],
            structure=scaled[1],
            neighbor_shift=scaled[2],
            omission=scaled[3],
            medium_frequency=scaled[4],
            noise=0.05,
        )

    @staticmethod
    def _crossover(a: ModelWeights, b: ModelWeights, rng: random.Random) -> ModelWeights:
        values_a = [
            a.public_heat_inverse,
            a.structure,
            a.neighbor_shift,
            a.omission,
            a.medium_frequency,
        ]
        values_b = [
            b.public_heat_inverse,
            b.structure,
            b.neighbor_shift,
            b.omission,
            b.medium_frequency,
        ]
        mixed = [rng.choice((left, right)) for left, right in zip(values_a, values_b)]
        total = sum(mixed)
        scaled = [value / total * 0.95 for value in mixed]
        return ModelWeights(
            public_heat_inverse=scaled[0],
            structure=scaled[1],
            neighbor_shift=scaled[2],
            omission=scaled[3],
            medium_frequency=scaled[4],
            noise=0.05,
        )

    @staticmethod
    def _fitness(backtest: Dict[str, object]) -> float:
        avg_best = float(backtest["avg_best_red_hit"])
        random_best = float(backtest["avg_random_best_red_hit"])
        three_plus = float(backtest["three_plus_rate"])
        four_plus = float(backtest["four_plus_rate"])
        candidate_delta = float(backtest["avg_candidate_pool_hit"]) - float(backtest["avg_random_candidate_pool_hit"])
        return avg_best + three_plus * 1.2 + four_plus * 2.0 + candidate_delta * 0.12 - max(0.0, random_best - avg_best) * 0.35

    @classmethod
    def optimize_weights(
        cls,
        df: pd.DataFrame,
        generations: int = 4,
        population_size: int = 10,
        periods: int = 80,
        groups: int = 5,
        candidate_pool_size: int = 22,
        seed: int = 20260715,
    ) -> Dict[str, object]:
        rng = random.Random(seed)
        population = [ModelWeights()] + [cls._random_weights(rng) for _ in range(max(1, population_size - 1))]
        history = []
        best_record = None

        for generation in range(generations):
            scored = []
            for idx, weights in enumerate(population):
                model = cls(df, weights=weights, seed=seed + generation * 100 + idx)
                backtest = model.backtest(periods=periods, groups=groups, candidate_pool_size=candidate_pool_size)
                fitness = cls._fitness(backtest)
                scored.append((fitness, weights, backtest))

            scored.sort(key=lambda item: item[0], reverse=True)
            top_fitness, top_weights, top_backtest = scored[0]
            generation_record = {
                "generation": generation + 1,
                "fitness": top_fitness,
                "weights": top_weights.as_dict(),
                "avg_best_red_hit": top_backtest["avg_best_red_hit"],
                "three_plus_rate": top_backtest["three_plus_rate"],
                "four_plus_rate": top_backtest["four_plus_rate"],
            }
            history.append(generation_record)

            if best_record is None or top_fitness > best_record["fitness"]:
                best_record = {
                    "fitness": top_fitness,
                    "weights": top_weights,
                    "backtest": top_backtest,
                }

            elites = [weights for _, weights, _ in scored[: max(2, population_size // 3)]]
            next_population = elites[:]
            while len(next_population) < population_size:
                parent_a, parent_b = rng.sample(elites, 2)
                child = cls._crossover(parent_a, parent_b, rng)
                child = cls._mutate_weights(child, rng)
                next_population.append(child)
            population = next_population

        assert best_record is not None
        return {
            "best_fitness": best_record["fitness"],
            "best_weights": best_record["weights"].as_dict(),
            "best_backtest": best_record["backtest"],
            "history": history,
            "settings": {
                "generations": generations,
                "population_size": population_size,
                "periods": periods,
                "groups": groups,
                "candidate_pool_size": candidate_pool_size,
                "seed": seed,
            },
        }


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def save_outputs(result: Dict[str, object], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "public_heat_avoidance_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Public heat avoidance model")
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=22)
    parser.add_argument("--backtest-periods", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--optimize", action="store_true", help="run a lightweight genetic weight search")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--optimization-periods", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100, help="purchase plan budget in RMB")
    parser.add_argument("--purchase-plans", type=int, default=8)
    args = parser.parse_args()

    df = load_data(args.data)
    optimization = None
    weights = None
    if args.optimize:
        optimization = PublicHeatAvoidanceModel.optimize_weights(
            df,
            generations=args.generations,
            population_size=args.population,
            periods=args.optimization_periods,
            groups=args.groups,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
        )
        weights = ModelWeights(**optimization["best_weights"])

    model = PublicHeatAvoidanceModel(df, weights=weights, seed=args.seed)
    current = model.generate_predictions(groups=args.groups, candidate_pool_size=args.candidate_pool_size)
    purchase_plans = model.generate_purchase_plans(
        budget=args.budget,
        max_plans=args.purchase_plans,
        candidate_pool_size=max(args.candidate_pool_size, 24),
    )
    backtest = model.backtest(
        periods=args.backtest_periods,
        groups=args.groups,
        candidate_pool_size=args.candidate_pool_size,
    )
    purchase_backtest = model.backtest_purchase_plans(
        periods=min(args.backtest_periods, 120),
        budget=args.budget,
        max_plans=min(args.purchase_plans, 5),
        candidate_pool_size=max(args.candidate_pool_size, 24),
    )

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": args.data,
        "data_rows": len(model.df),
        "period_range": [int(model.df["period"].min()), int(model.df["period"].max())],
        "model_note": "Hypothesis test: favor structurally normal numbers with lower estimated public buying heat.",
        "weights": model.weights.as_dict(),
        "current": current,
        "purchase_plans": purchase_plans,
        "backtest": backtest,
        "purchase_backtest": purchase_backtest,
        "optimization": optimization,
    }
    output_path = save_outputs(result, args.output)

    print("=" * 72)
    print("Public heat avoidance model")
    print("=" * 72)
    print(f"Rows: {len(model.df)}")
    print(f"Period range: {model.df['period'].min()} ~ {model.df['period'].max()}")
    if optimization:
        print("Optimized weights:")
        for key, value in model.weights.as_dict().items():
            print(f"  {key}: {value:.4f}")
    print()
    print("Estimated public hot reds:")
    print("  " + " ".join(f"{n:02d}" for n in current["public_hot_reds"]))
    print("Avoidance candidate pool:")
    print("  " + " ".join(f"{n:02d}" for n in current["avoidance_candidates"]))
    print()
    print("Predictions:")
    for idx, group in enumerate(current["predictions"], 1):
        reds = " ".join(f"{n:02d}" for n in group["red"])
        print(
            f"  {idx}. red [{reds}] + blue {group['blue']:02d} "
            f"score={group['combo_score']:.4f} consecutive={group['consecutive_pairs']}"
        )
    print()
    print(f"Purchase plans under RMB {args.budget}:")
    for idx, plan in enumerate(purchase_plans[: args.purchase_plans], 1):
        dan = " ".join(f"{n:02d}" for n in plan["dan"])
        tuo = " ".join(f"{n:02d}" for n in plan["tuo"])
        blue = " ".join(f"{n:02d}" for n in plan["blue"])
        print(
            f"  {idx}. dan [{dan}] tuo [{tuo}] blue [{blue}] "
            f"bets={plan['bet_count']} cost={plan['cost']} score={plan['efficiency_score']:.4f}"
        )
    print()
    print("Backtest:")
    print(f"  periods: {backtest['periods']}")
    print(f"  avg best red hit: {backtest['avg_best_red_hit']:.3f}")
    print(f"  random baseline avg best red hit: {backtest['avg_random_best_red_hit']:.3f}")
    print(f"  avg candidate pool hit: {backtest['avg_candidate_pool_hit']:.3f}")
    print(f"  random candidate pool hit: {backtest['avg_random_candidate_pool_hit']:.3f}")
    print(f"  3+ red hit rate: {backtest['three_plus_rate']:.2%}")
    print(f"  random 3+ red hit rate: {backtest['random_three_plus_rate']:.2%}")
    print(f"  4+ red hit rate: {backtest['four_plus_rate']:.2%}")
    print()
    print("Purchase plan backtest:")
    print(f"  periods: {purchase_backtest['periods']}")
    print(f"  avg total listed cost: {purchase_backtest['avg_total_cost']:.2f}")
    print(f"  avg best red pool hit: {purchase_backtest['avg_best_red_pool_hit']:.3f}")
    print(f"  blue hit rate: {purchase_backtest['blue_hit_rate']:.2%}")
    print(f"  advance path rate blue or 4+ red: {purchase_backtest['advance_path_rate']:.2%}")
    print(f"  4+ red pool rate: {purchase_backtest['four_plus_pool_rate']:.2%}")
    print(f"  blue and 4+ red rate: {purchase_backtest['blue_and_four_plus_rate']:.2%}")
    print(f"  full red cover rate: {purchase_backtest['full_red_cover_rate']:.2%}")
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""Mix catalogue: every two-filament blend the library can produce.

Bambu Studio's *add mixed filament* dialog mixes two filaments whose percentages
run from 10 % to 90 %, which is 81 distinct ratios per pair.  For a library of
``n`` spools that gives ``C(n, 2) * 81`` predicted colours:

===========  =========  ==========
spools (n)   pairs      mixes
===========  =========  ==========
2            1          81
3            3          243
4            6          486
5            10         810
6            15         1215
8            28         2268
10           45         3645
===========  =========  ==========

Each entry keeps a link back to the two parent spools and their percentages, so
the UI can answer "which two filaments, in what ratio?" for any swatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..spectral import color as _color
from .engines import DEFAULT_ENGINE, MixEngine, get_engine

#: Percentages of the first filament: 10, 11, ... 90 — the 81 Bambu ratios.
MIX_RATIOS: tuple[int, ...] = tuple(range(10, 91))
RATIO_COUNT = len(MIX_RATIOS)

#: How many pair-groups to mix in a single NumPy batch.
_PAIR_CHUNK = 128

SORT_RGB = "rgb"
SORT_LIGHTNESS = "lightness"
SORT_HUE = "hue"
SORT_PAIR = "pair"
SORT_LABEL = "label"

SORT_CHOICES: tuple[tuple[str, str], ...] = (
    (SORT_RGB, "按 RGB 排列"),
    (SORT_LIGHTNESS, "按亮度 L*"),
    (SORT_HUE, "按色相 H"),
    (SORT_PAIR, "按母材组合"),
    (SORT_LABEL, "按名称"),
)


@dataclass(frozen=True)
class MixRecipe:
    """One predicted blend of two filaments at a fixed ratio."""

    pair_index: int
    a_id: str
    b_id: str
    percent_a: int
    percent_b: int
    color_hex: str
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]
    engine: str = DEFAULT_ENGINE

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.a_id, self.b_id)

    @property
    def key(self) -> str:
        return f"{self.a_id}|{self.b_id}|{self.percent_a}"

    @property
    def lightness(self) -> float:
        return self.lab[0]

    @property
    def hue(self) -> float:
        """Hue angle in degrees, 0 for achromatic colours."""
        _, a, b = self.lab
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            return 0.0
        return float(np.degrees(np.arctan2(b, a)) % 360.0)

    @property
    def chroma(self) -> float:
        _, a, b = self.lab
        return float(np.hypot(a, b))

    @property
    def ratio_text(self) -> str:
        return f"{self.percent_a}% + {self.percent_b}%"


class MixCatalog:
    """All predicted blends for a filament library.

    Building is vectorised: the reflectance spectrum of each spool is estimated
    once, and every pair is then a weighted K/S blend of two spectra.
    """

    def __init__(
        self,
        filaments: Sequence,
        ratios: Iterable[int] = MIX_RATIOS,
        engine: str | MixEngine = DEFAULT_ENGINE,
    ) -> None:
        self._filaments = list(filaments)
        self._ratios = tuple(int(r) for r in ratios)
        self._engine = get_engine(engine)
        self._recipes: list[MixRecipe] = []
        self._pairs: list[tuple[str, str]] = []
        self._by_pair: dict[tuple[str, str], list[MixRecipe]] = {}
        self._by_id: dict[str, object] = {}
        self._built = False

    # -- construction -------------------------------------------------------------
    @property
    def engine(self) -> MixEngine:
        return self._engine

    @property
    def ratios(self) -> tuple[int, ...]:
        return self._ratios

    @property
    def recipe_count(self) -> int:
        return len(self._recipes)

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    @property
    def filaments(self) -> list:
        return list(self._filaments)

    @property
    def recipes(self) -> list[MixRecipe]:
        return list(self._recipes)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return list(self._pairs)

    def build(self) -> "MixCatalog":
        """(Re)compute every recipe. Cheap enough to call on each library edit."""
        self._recipes = []
        self._by_pair = {}
        self._by_id = {f.id: f for f in self._filaments}
        self._pairs = []
        self._built = True

        count = len(self._filaments)
        if count < 2:
            return self

        colours = np.array([f.rgb for f in self._filaments], dtype=np.float64)

        ratios = np.asarray(self._ratios, dtype=np.float64)
        weights_a = ratios / 100.0
        weights_b = 1.0 - weights_a
        steps = len(ratios)

        pair_list = [(i, j) for i in range(count) for j in range(i + 1, count)]
        for start in range(0, len(pair_list), _PAIR_CHUNK):
            block = pair_list[start:start + _PAIR_CHUNK]
            parents = np.stack(
                [
                    colours[[i for i, _ in block]],
                    colours[[j for _, j in block]],
                ],
                axis=1,
            )                                                        # (P, 2, 3)
            weights = np.empty((len(block), steps, 2), dtype=np.float64)
            weights[:, :, 0] = weights_a
            weights[:, :, 1] = weights_b

            rgbs = self._engine.mix_rgb(parents, weights)             # (P, R, 3)
            labs = _color.lab_from_rgb(rgbs)

            for local, (i, j) in enumerate(block):
                pair_index = start + local
                fa, fb = self._filaments[i], self._filaments[j]
                self._pairs.append((fa.id, fb.id))
                group: list[MixRecipe] = []
                for r_index, percent_a in enumerate(self._ratios):
                    r, g, b = (int(v) for v in rgbs[local, r_index])
                    recipe = MixRecipe(
                        pair_index=pair_index,
                        a_id=fa.id,
                        b_id=fb.id,
                        percent_a=int(percent_a),
                        percent_b=100 - int(percent_a),
                        color_hex=f"#{r:02X}{g:02X}{b:02X}",
                        lab=(float(labs[local, r_index, 0]),
                             float(labs[local, r_index, 1]),
                             float(labs[local, r_index, 2])),
                        rgb=(r, g, b),
                        engine=self._engine.id,
                    )
                    group.append(recipe)
                    self._recipes.append(recipe)
                self._by_pair[(fa.id, fb.id)] = group

        return self

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    # -- queries ------------------------------------------------------------------
    def pair_recipes(self, a_id: str, b_id: str) -> list[MixRecipe]:
        """The 81 recipes of one unordered pair, in ascending ratio order."""
        self._ensure_built()
        if a_id == b_id:
            return []
        return list(self._by_pair.get((a_id, b_id)) or self._by_pair.get((b_id, a_id)) or [])

    def nearest_ratio(self, a_id: str, b_id: str, target_hex: str) -> MixRecipe | None:
        """The recipe of a pair whose predicted colour is closest to ``target_hex``."""
        group = self.pair_recipes(a_id, b_id)
        if not group:
            return None
        target = _color.lab_from_rgb(_color.hex_to_rgb(target_hex))
        best = None
        best_distance = float("inf")
        for recipe in group:
            distance = _color.delta_e_2000(recipe.lab, target)
            if distance < best_distance:
                best_distance = distance
                best = recipe
        return best

    def find_recipe(self, a_id: str, b_id: str, percent_a: int) -> MixRecipe | None:
        for recipe in self.pair_recipes(a_id, b_id):
            if recipe.percent_a == percent_a:
                return recipe
        return None

    def sorted_recipes(self, key: str = SORT_RGB) -> list[MixRecipe]:
        """Recipes ordered by ``key``; see :data:`SORT_CHOICES`."""
        self._ensure_built()
        recipes = self._recipes
        if key == SORT_RGB:
            return sorted(recipes, key=lambda m: m.rgb)
        if key == SORT_LIGHTNESS:
            return sorted(recipes, key=lambda m: (-m.lightness, m.hue))
        if key == SORT_HUE:
            return sorted(recipes, key=lambda m: (m.hue, -m.chroma, -m.lightness))
        if key == SORT_PAIR:
            return sorted(recipes, key=lambda m: (m.pair_index, m.percent_a))
        if key == SORT_LABEL:
            return sorted(recipes, key=lambda m: (self._label(m.a_id), self._label(m.b_id), m.percent_a))
        raise ValueError(f"unknown sort key {key!r}")

    def _label(self, filament_id: str):
        filament = self._by_id.get(filament_id)
        return filament.display_name.casefold() if filament is not None else filament_id

    def describe(self, recipe: MixRecipe) -> str:
        """Human sentence naming both spools and the ratio."""
        fa = self._by_id.get(recipe.a_id)
        fb = self._by_id.get(recipe.b_id)
        name_a = fa.display_name if fa is not None else recipe.a_id
        name_b = fb.display_name if fb is not None else recipe.b_id
        hex_a = fa.color_hex if fa is not None else "?"
        hex_b = fb.color_hex if fb is not None else "?"
        return (
            f"{name_a} ({hex_a}) {recipe.percent_a}%  +  "
            f"{name_b} ({hex_b}) {recipe.percent_b}%"
        )

    def stats(self) -> dict:
        self._ensure_built()
        return {
            "filaments": len(self._filaments),
            "pairs": len(self._pairs),
            "recipes": len(self._recipes),
            "ratios": len(self._ratios),
        }


def expected_recipe_count(filament_count: int, ratio_count: int = RATIO_COUNT) -> int:
    """``C(n, 2) * 81`` — used by the tests and by the UI's status line."""
    if filament_count < 2:
        return 0
    return filament_count * (filament_count - 1) // 2 * ratio_count

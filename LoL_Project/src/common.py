from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "ally_Ahri",
    "ally_Ashe",
    "ally_Darius",
    "ally_Hecarim",
    "ally_Khazix",
    "enemy_Nasus",
    "enemy_Sejuani",
    "enemy_Warwick",
    "enemy_XinZhao",
    "enemy_Ziggs",
]


def compute_minimap_crop(
    frame_width: int,
    frame_height: int,
    minimap_w: int,
    minimap_h: int,
    left_offset: int,
    top_offset: int,
    right_offset: int,
    bottom_offset: int,
) -> tuple[int, int, int, int]:
    left_crop = frame_width - minimap_w + left_offset
    top_crop = frame_height - minimap_h + top_offset
    right_crop = frame_width - right_offset
    bottom_crop = frame_height - bottom_offset

    left_crop = max(0, left_crop)
    top_crop = max(0, top_crop)
    right_crop = min(frame_width, right_crop)
    bottom_crop = min(frame_height, bottom_crop)
    return left_crop, top_crop, right_crop, bottom_crop


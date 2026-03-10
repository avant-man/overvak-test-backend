import re

ACTIONS = [
    "climbing", "crying", "falling",
    "hitting", "jumping", "laughing",
    "pushing", "running"
]

GROUND_TRUTH: dict[str, dict[str, int]] = {
    "VkhiJBpoofM": {
        "climbing": 0, "crying": 0, "falling": 0, "hitting": 0,
        "jumping": 0, "laughing": 1, "pushing": 0, "running": 1,
    },
    "qzvfds5Nqus": {
        "climbing": 0, "crying": 1, "falling": 1, "hitting": 1,
        "jumping": 0, "laughing": 1, "pushing": 0, "running": 0,
    },
    "D0x6P-UhGgs": {
        "climbing": 0, "crying": 1, "falling": 0, "hitting": 1,
        "jumping": 0, "laughing": 0, "pushing": 0, "running": 0,
    },
    "v8RPGCc8p9o": {
        "climbing": 1, "crying": 1, "falling": 1, "hitting": 1,
        "jumping": 0, "laughing": 1, "pushing": 0, "running": 0,
    },
    "aGw3sH8aVo8": {
        "climbing": 0, "crying": 1, "falling": 0, "hitting": 0,
        "jumping": 0, "laughing": 0, "pushing": 1, "running": 0,
    },
}

def get_ground_truth(video_id: str) -> dict[str, int] | None:
    return GROUND_TRUTH.get(video_id)

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None
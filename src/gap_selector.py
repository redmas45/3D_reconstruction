import random


def four_chunks(total_frames: int) -> list[tuple[int, int]]:
    """Returns four inclusive frame ranges covering the whole video."""
    if total_frames < 4:
        raise ValueError("Video must have at least 4 frames")
    chunks = []
    for idx in range(4):
        start = round(idx * total_frames / 4)
        end = round((idx + 1) * total_frames / 4) - 1
        chunks.append((start, min(total_frames - 1, end)))
    return chunks


def choose_hidden_chunk(total_frames: int, rng: random.Random) -> dict:
    chunks = four_chunks(total_frames)
    hidden_index = rng.randrange(4)
    hidden = chunks[hidden_index]
    visible = [chunk for idx, chunk in enumerate(chunks) if idx != hidden_index]
    return {
        "chunks": chunks,
        "hidden_index": hidden_index,
        "hidden_range": hidden,
        "visible_ranges": visible,
    }

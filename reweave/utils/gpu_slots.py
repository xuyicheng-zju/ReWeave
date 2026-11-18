"""Stable GPU/lane identities that retain the original four-lane lock names."""


def encode_slot(device: int, lane: int) -> int:
    if type(device) is not int or not 0 <= device < 8:
        raise ValueError("Invalid pooled GPU device")
    if type(lane) is not int or not 0 <= lane < 16:
        raise ValueError("Invalid pooled GPU lane")
    # The first four lanes keep IDs gpu*4+lane. Further four-lane blocks use
    # new IDs, so increasing capacity never reassigns an existing physical lock.
    return (device // 4) * 64 + (lane // 4) * 16 + (device % 4) * 4 + lane % 4


def decode_slot(slot: int) -> tuple[int, int]:
    if type(slot) is not int or not 0 <= slot < 128:
        raise ValueError("Invalid pooled GPU slot")
    return (slot // 64) * 4 + (slot % 16) // 4, ((slot % 64) // 16) * 4 + slot % 4

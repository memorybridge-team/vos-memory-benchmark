"""What the installed SAM 2 actually calls things.

The point of this module is that every version-dependent assumption in the
wrapper is stated *here*, once, and checked against the live object instead of
being spread through the code as a remembered fact.

SAM 2 and SAM 2.1 differ in how `inference_state` is organised (a combined
`output_dict` in the earlier layout, per-object dicts and
`frames_tracked_per_obj` in the later one), and they differ in memory behaviour.
A wrapper that guessed wrong would not crash -- it would export a state that is
missing entries and produce numbers that look plausible.  So the probe runs
first and refuses to continue when it does not recognise what it is holding.

`scripts/dump_sam2_state.py` prints this probe's findings into
`docs/sam2_state_layout.md`.  Run it once per environment, before anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["StateLayout", "probe_layout", "MEMORY_KEYS"]

#: Keys of one frame's output that the handoff cares about by name.  Anything
#: else the model stored travels in `MemorySlot.extras`, untouched.
MEMORY_KEYS = ("maskmem_features", "obj_ptr", "object_score_logits")

#: Deliberately *not* carried across: the target re-issues its own.
REGENERATED_KEYS = ("maskmem_pos_enc",)


@dataclass
class StateLayout:
    per_obj: bool                  # output_dict_per_obj present
    combined: bool                 # a single output_dict present
    frames_tracked_per_obj: bool   # SAM 2.1 style tracking bookkeeping
    obj_maps: bool                 # obj_id_to_idx / obj_idx_to_id present
    keys: tuple[str, ...] = ()
    frame_out_keys: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    def require(self) -> "StateLayout":
        if not self.per_obj:
            raise RuntimeError(
                "this SAM 2 build has no inference_state['output_dict_per_obj']; "
                "the wrapper exports memory per object and cannot work from a "
                "combined dict alone. Run scripts/dump_sam2_state.py and adapt "
                "hoeval/models/sam2_layout.py before going further."
            )
        if not self.obj_maps:
            raise RuntimeError(
                "no obj_id_to_idx/obj_idx_to_id in inference_state: without the "
                "object index mapping a transferred memory can be attached to "
                "the wrong object, which is silent and catastrophic."
            )
        missing = [k for k in MEMORY_KEYS if self.frame_out_keys and k not in self.frame_out_keys]
        if missing:
            raise RuntimeError(
                "frame outputs are missing %s. object_score_logits in "
                "particular is the occlusion state -- dropping it makes the "
                "target treat an object the source had lost as fully visible."
                % (missing,)
            )
        return self


def probe_layout(inference_state) -> StateLayout:
    """Inspect a live `inference_state`. Cheap; call it once per session."""
    keys = tuple(sorted(inference_state.keys()))
    per_obj = "output_dict_per_obj" in inference_state
    frame_out_keys: tuple[str, ...] = ()
    if per_obj:
        for per in inference_state["output_dict_per_obj"].values():
            for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for out in per.get(bucket, {}).values():
                    frame_out_keys = tuple(sorted(out.keys()))
                    break
                if frame_out_keys:
                    break
            if frame_out_keys:
                break
    return StateLayout(
        per_obj=per_obj,
        combined="output_dict" in inference_state,
        frames_tracked_per_obj="frames_tracked_per_obj" in inference_state,
        obj_maps="obj_id_to_idx" in inference_state and "obj_idx_to_id" in inference_state,
        keys=keys,
        frame_out_keys=frame_out_keys,
        notes={
            "num_frames": inference_state.get("num_frames"),
            "storage_device": str(inference_state.get("storage_device", "?")),
            "device": str(inference_state.get("device", "?")),
            "objects": len(inference_state.get("obj_ids", []) or []),
        },
    )

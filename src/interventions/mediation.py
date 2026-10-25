from __future__ import annotations

from src.hooks.core import first_tensor
from src.interventions.hooks import make_patch_hook
from src.interventions.patching import capture_module_vector, margin_with_hook
from src.utils.module_access import ModuleSpec, resolve_module


def downstream_mediation_patch(bundle, clean_prompt: str, corrupt_prompt: str,
                               rule_answer: str, shortcut_answer: str,
                               source_spec: ModuleSpec, downstream_spec: ModuleSpec,
                               position: int=-1):
    if source_spec.layer >= downstream_spec.layer:
        raise ValueError("source layer should precede downstream layer")
    src=resolve_module(bundle.model,source_spec); down=resolve_module(bundle.model,downstream_spec)
    src_clean=capture_module_vector(bundle,clean_prompt,source_spec,position)
    actual_pos = position
    if position == -1:
        actual_pos = len(bundle.tokenizer(corrupt_prompt, add_special_tokens=False)["input_ids"]) - 1
    holder={}
    def capture_down(_m,_inp,output):
        h=first_tensor(output); p=actual_pos; holder["v"]=h[:,p,:].detach().float().cpu()[0]
    h1=src.register_forward_hook(make_patch_hook(src_clean,actual_pos)); h2=down.register_forward_hook(capture_down)
    try:
        total_source_patch=margin_with_hook(bundle,corrupt_prompt,rule_answer,shortcut_answer)
    finally:
        h2.remove(); h1.remove()
    mediated_vec=holder["v"]
    baseline=margin_with_hook(bundle,corrupt_prompt,rule_answer,shortcut_answer)
    mediated=margin_with_hook(bundle,corrupt_prompt,rule_answer,shortcut_answer,hooks=[(down,make_patch_hook(mediated_vec,actual_pos))])
    clean=margin_with_hook(bundle,clean_prompt,rule_answer,shortcut_answer)
    return {"clean_margin":clean,"corrupt_margin":baseline,"source_patched_margin":total_source_patch,"mediated_margin":mediated,"total_source_effect":total_source_patch-baseline,"downstream_mediated_effect":mediated-baseline}

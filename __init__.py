NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge_module(module_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"])
    except Exception as exc:
        print(f"[KIE] 跳过 {module_name}: {exc}")
        return

    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


for _module_name in ("kie_image_nodes", "kie_sora2_nodes", "kie_universal_nodes"):
    _merge_module(_module_name)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

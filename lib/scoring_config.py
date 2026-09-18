# Scoring config validation
"""Config validation for scoring."""
from typing import Dict, Any

def validate_scoring_config(cfg: Dict, score_name: str, pc_id: str) -> Dict[str, Any]:
    """Validate scoring config."""
    print(f"\n* Validating: {score_name}")
    if "scoring" not in cfg or score_name not in cfg["scoring"]:
        raise ValueError(f"Config '{score_name}' not found")
    score_cfg = cfg["scoring"][score_name]
    paths = cfg["machines"][pc_id]["paths"]
    output_base = paths["output_path"]
    target_method = cfg.get("model", {}).get("target_method", "direct")
    glm_source = score_cfg.get("glm_source", "lookup")
    
    # Validate residual method requirements
    if target_method == "residual":
        glm_col = cfg.get("model", {}).get("glm_prediction_column")
        if not glm_col:
            raise ValueError("model.glm_prediction_column required for target_method=residual")
        print(f"  GLM column: {glm_col}")
    
    print(f"  Source: {score_cfg['source']}, Method: {target_method}")
    return {"score_cfg": score_cfg, "output_base": output_base, "target_method": target_method, "glm_source": glm_source, "paths": paths}

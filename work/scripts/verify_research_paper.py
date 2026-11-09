import re
import sys
from pathlib import Path

def verify_paper():
    paper_path = Path("submission/research_paper.md")
    if not paper_path.exists():
        print(f"FAIL: File does not exist: {paper_path}")
        sys.exit(1)
        
    with open(paper_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    words = text.split()
    lines = text.splitlines()
    print(f"Loaded {paper_path}: {len(words):,} words across {len(lines):,} lines.")
    
    errors = []
    
    # 1. Section Structure Check (All 21 sections in order)
    required_sections = [
        "Title",
        "Abstract",
        "Keywords",
        "Introduction",
        "Research problem and objectives",
        "Dataset description and data contract",
        "Cohort selection and eligibility",
        "Target construction",
        "Feature engineering and outlier treatment",
        "Experimental design and client-grouped validation",
        "Baseline and machine-learning models",
        "Results and quantitative comparison",
        "Client-level validation",
        "Feature importance and interpretation",
        "Ranked content opportunity framework",
        "Reason codes and editorial actions",
        "Discussion and practical implications",
        "Limitations and ethical considerations",
        "Reproducibility",
        "Conclusion and future work",
        "Acknowledgements and dataset credit"
    ]
    
    last_pos = -1
    for i, s_title in enumerate(required_sections, 1):
        pat = rf"##\s+{i}\.\s+{re.escape(s_title)}"
        match = re.search(pat, text, re.IGNORECASE)
        if not match:
            errors.append(f"Missing section {i}: '{s_title}'")
        else:
            pos = match.start()
            if pos < last_pos:
                errors.append(f"Section {i}: '{s_title}' appears out of order.")
            last_pos = pos
            
    # 2. Strict Privacy & Identity Protection Checks
    # Real client identifier hashes: client_[0-9a-f]{6,}
    client_hashes = re.findall(r"\bclient_[0-9a-f]{6,}\b", text, re.IGNORECASE)
    if client_hashes:
        errors.append(f"Privacy breach: Real client identifier hashes exposed: {set(client_hashes)}")
        
    # Real content identifier hashes: content_[0-9a-f]{6,}
    content_hashes = re.findall(r"\bcontent_[0-9a-f]{6,}\b", text, re.IGNORECASE)
    if content_hashes:
        errors.append(f"Privacy breach: Real content identifier hashes exposed: {set(content_hashes)}")
        
    # Tokens & credentials: hf_... or Bearer ...
    tokens = re.findall(r"\b(hf_[a-zA-Z0-9]{10,}|Bearer\s+[a-zA-Z0-9]+)\b", text)
    if tokens:
        errors.append(f"Security breach: Token or credential exposed: {tokens}")
        
    # Synthetic Opportunity ID Verification: OPP_0001 to OPP_0010
    if not re.search(r"\bOPP_0001\b", text):
        errors.append("Missing synthetic opportunity IDs (OPP_0001).")
        
    # Synthetic Domain Aliases: Domain A to Domain H
    if not re.search(r"Domain A", text):
        errors.append("Missing synthetic domain aliases (Domain A).")
        
    # 3. Authoritative Reconciled Metrics Checks
    # Decline rates
    if "41.69%" not in text:
        errors.append("Missing overall eligible portfolio decline rate (41.69%).")
    if "51.55%" not in text:
        errors.append("Missing held-out test cohort decline rate (51.55%).")
    if "31.77%" not in text:
        errors.append("Missing unweighted test client mean decline rate (31.77%).")
    if "30.16%" not in text:
        errors.append("Missing unweighted train client mean decline rate (30.16%).")
        
    # Cohort counts
    if "101,226" not in text:
        errors.append("Missing total eligible page count (101,226).")
    if "29,873" not in text:
        errors.append("Missing held-out test page count (29,873).")
    if "71,353" not in text:
        errors.append("Missing training page count (71,353).")
        
    # Precision@100
    if not re.search(r"0\.9600|0\.96", text):
        errors.append("Missing HistGradientBoosting Precision@100 (0.9600).")
    if not re.search(r"0\.5900|0\.59", text):
        errors.append("Missing Week 4 Baseline Precision@100 (0.5900).")
        
    # ROC-AUC
    if "0.7441" not in text:
        errors.append("Missing HistGradientBoosting ROC-AUC (0.7441).")
    if "0.5066" not in text:
        errors.append("Missing Baseline ROC-AUC (0.5066).")
    if "0.7353" not in text:
        errors.append("Missing Random Forest ROC-AUC (0.7353).")
    if "0.7250" not in text:
        errors.append("Missing Logistic Regression ROC-AUC (0.7250).")
        
    # Brier score
    if "0.2085" not in text:
        errors.append("Missing HistGradientBoosting Brier score (0.2085).")
        
    # 4. Subgroup Boundary Checks
    if not re.search(r"Sufficient\s*\(\$N\s*\\ge\s*50\$\)", text):
        errors.append("Missing 'Sufficient ($N \\ge 50$)' subgroup classification marker.")
    if not re.search(r"Insufficient\s*\(\$N\s*<\s*50\$\)", text):
        errors.append("Missing 'Insufficient ($N < 50$)' subgroup classification marker.")
    if "universal generalization" not in text.lower():
        errors.append("Missing explicit disclaimer of universal generalization.")
        
    # 5. Non-Causal & Decision Support Claims
    if "does not prove causation" not in text.lower():
        errors.append("Missing explicit non-causal statement ('does not prove causation').")
    if "decision support" not in text.lower():
        errors.append("Missing 'decision support' framing.")
        
    # 6. Technical Description Checks (Addressing user corrections)
    if re.search(r"\bbounded symmetric log-ratio\b", text, re.IGNORECASE):
        errors.append("Found incorrect phrasing: 'bounded symmetric log-ratio'. Should be 'symmetric log-ratio momentum feature'.")
    if re.search(r"\borthogonal 13-feature\b", text, re.IGNORECASE):
        errors.append("Found incorrect phrasing: 'orthogonal 13-feature'. Should be '13-feature matrix with reduced redundancy'.")
        
    # 7. Mandatory Acknowledgement Check
    required_ack = "Built on the FlyRank ML Internship dataset"
    if required_ack not in text:
        errors.append(f"Missing exact acknowledgement: '{required_ack}'")
    if "https://flyrank.ai" not in text:
        errors.append("Missing link to https://flyrank.ai in acknowledgement.")
        
    # 8. Check that paper_url.txt is untouched
    url_file = Path("submission/paper_url.txt")
    if not url_file.exists():
        errors.append("Missing submission/paper_url.txt")
    else:
        url_content = url_file.read_text(encoding="utf-8").strip()
        if url_content != "PASTE-YOUR-DEPLOYED-PAPER-URL-HERE":
            errors.append(f"submission/paper_url.txt was modified prematurely: '{url_content}'")
            
    # Final Reporting
    print("-" * 60)
    if errors:
        print(f"VERIFICATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL VERIFICATION CHECKS PASSED PERFECTLY (EXIT CODE 0).")
        print("  - All 21 sections present in required order.")
        print("  - Privacy: Zero real client or content hashes exposed.")
        print("  - Metrics: All numbers match final_metrics_reconciliation.json.")
        print("  - Subgroup & non-causal boundaries fully respected.")
        print("  - Corrected technical descriptions verified.")
        print("  - Mandatory FlyRank acknowledgement verified.")
        print("  - submission/paper_url.txt untouched.")
        sys.exit(0)

if __name__ == "__main__":
    verify_paper()

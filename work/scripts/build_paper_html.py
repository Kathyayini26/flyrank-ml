import re
import shutil
from pathlib import Path
import mistune

def generate_paper_html():
    paper_path = Path("submission/research_paper.md")
    docs_dir = Path("docs")
    img_dir = docs_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy all 5 charts to docs/img/
    charts_source = Path("work/outputs/charts")
    expected_charts = [
        "capstone_evaluation_panel.png",
        "roc_curves_comparison.png",
        "precision_at_k_comparison.png",
        "feature_importance_rf.png",
        "risk_score_distribution.png"
    ]
    if charts_source.exists():
        for chart_name in expected_charts:
            chart_file = charts_source / chart_name
            if chart_file.exists():
                shutil.copy2(chart_file, img_dir / chart_name)
                print(f"Copied chart: {chart_name} -> {img_dir / chart_name}")
            else:
                print(f"Warning: Chart file {chart_file} not found in {charts_source}")
            
    # 2. Read research paper markdown (strict read-only; never modify)
    with open(paper_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # 3. Convert markdown to HTML with table support
    markdown = mistune.create_markdown(plugins=['table'])
    raw_html = markdown(md_text)
    
    # 4. Define all 5 Figure HTML elements with relative paths and academic captions
    fig1_html = """
    <figure class="paper-figure">
      <img src="img/capstone_evaluation_panel.png" alt="Figure 1: Comprehensive Model Evaluation Panel" loading="lazy" />
      <figcaption><strong>Figure 1: Comprehensive Model Evaluation Panel.</strong> Multi-panel empirical evaluation across all benchmarked models on 29,873 held-out test pages. Top Left: ROC Curves showing HistGradientBoosting (AUC = 0.7441) vs. Week 4 Baseline (0.5066). Top Right: Precision-Recall Curves benchmarked against the 51.55% test base rate. Bottom Left: Precision@K across decision cutoffs K &in; {10, 20, 50, 100}. Bottom Right: Random Forest Gini Feature Importance.</figcaption>
    </figure>
    """
    
    fig2_html = """
    <figure class="paper-figure">
      <img src="img/roc_curves_comparison.png" alt="Figure 2: Receiver Operating Characteristic (ROC) Curves" loading="lazy" />
      <figcaption><strong>Figure 2: Receiver Operating Characteristic (ROC) Curves.</strong> True Positive Rate versus False Positive Rate across all decision thresholds for the Week 4 Rule Baseline (AUC = 0.5066), Logistic Regression (AUC = 0.7250), Random Forest (AUC = 0.7353), and HistGradientBoosting (AUC = 0.7441) evaluated on 29,873 held-out enterprise pages across 8 unseen clients.</figcaption>
    </figure>
    """
    
    fig3_html = """
    <figure class="paper-figure">
      <img src="img/precision_at_k_comparison.png" alt="Figure 3: Operational Precision@K Across Decision Cutoffs" loading="lazy" />
      <figcaption><strong>Figure 3: Operational Precision@K Across Decision Cutoffs.</strong> Precision achieved within the top K &in; {10, 20, 50, 100} ranked recommendations. All three machine learning models achieve &ge; 95% precision across top ranks, with HistGradientBoosting achieving 0.9600 Precision@100 versus 0.5900 for the rule baseline, compared against the held-out test cohort base rate of 51.55%.</figcaption>
    </figure>
    """
    
    fig4_html = """
    <figure class="paper-figure">
      <img src="img/feature_importance_rf.png" alt="Figure 4: Random Forest Gini Feature Importance Distribution" loading="lazy" />
      <figcaption><strong>Figure 4: Random Forest Gini Feature Importance Distribution.</strong> Relative predictive importance across the 13 engineered features. Month-over-month impression velocity (mom_log_ratio) represents the primary predictor at 38.28%, followed by longitudinal observation persistence (feat_observed_days, 15.12%) and snippet click-through efficiency (feat_ctr, 7.90%).</figcaption>
    </figure>
    """
    
    fig5_html = """
    <figure class="paper-figure">
      <img src="img/risk_score_distribution.png" alt="Figure 5: Model Risk Score (Predicted Decline Probability) Distribution" loading="lazy" />
      <figcaption><strong>Figure 5: Model Risk Score (Predicted Decline Probability) Distribution.</strong> Distribution of predicted decline probabilities across 29,873 held-out test pages, illustrating the separation between low-risk stable pages and high-priority decay candidates stratified across operational risk tiers.</figcaption>
    </figure>
    """
    
    # 5. Insert figures into designated sections cleanly
    # Section 12: Insert Figure 1 immediately after Section 12 heading
    raw_html = raw_html.replace(
        "<h2>12. Results and Quantitative Comparison</h2>",
        "<h2>12. Results and Quantitative Comparison</h2>" + fig1_html
    )
    
    # Section 12.1: Insert Figure 2 after Section 12.1 heading
    raw_html = raw_html.replace(
        "<h3>12.1 Evaluation Metrics</h3>",
        "<h3>12.1 Evaluation Metrics</h3>" + fig2_html
    )
    
    # Section 12.2: Insert Figure 3 after Section 12.2 heading
    raw_html = raw_html.replace(
        "<h3>12.2 Authoritative Reconciled Comparison</h3>",
        "<h3>12.2 Authoritative Reconciled Comparison</h3>" + fig3_html
    )
    
    # Section 14.1: Insert Figure 4 after Section 14.1 heading
    raw_html = raw_html.replace(
        "<h3>14.1 Gini Impurity Feature Contributions</h3>",
        "<h3>14.1 Gini Impurity Feature Contributions</h3>" + fig4_html
    )
    
    # Section 15.2: Insert Figure 5 after Section 15.2 heading
    raw_html = raw_html.replace(
        "<h3>15.2 Opportunity Tiering</h3>",
        "<h3>15.2 Opportunity Tiering</h3>" + fig5_html
    )
    
    # 6. Assemble complete responsive HTML page
    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Predicting Search Performance Decay — FlyRank AI/ML Research Paper</title>
  <meta name="description" content="A Grouped Machine Learning Framework for Content Opportunity Prioritisation. FlyRank AI/ML Internship Capstone Paper." />
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
  
  <!-- KaTeX for math typesetting -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" />
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {{
            delimiters: [
              {{left: '$$', right: '$$', display: true}},
              {{left: '$', right: '$', display: false}},
              {{left: '\\\\[', right: '\\\\]', display: true}},
              {{left: '\\\\(', right: '\\\\)', display: false}}
            ],
            throwOnError: false
          }});"></script>

  <style>
    :root {{
      --bg: #fdfdfc;
      --text: #1a1f2c;
      --text-muted: #5a6474;
      --primary: #0f2b5c;
      --primary-light: #e8f0fe;
      --accent: #0284c7;
      --border: #e2e8f0;
      --surface: #ffffff;
      --code-bg: #f1f5f9;
      --table-header: #0f2b5c;
      --radius: 8px;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: 'Source Serif 4', Georgia, serif;
      font-size: 17px;
      line-height: 1.7;
      -webkit-font-smoothing: antialiased;
    }}

    /* Top Navigation Banner */
    .top-banner {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(15, 43, 92, 0.96);
      backdrop-filter: blur(8px);
      color: #ffffff;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.15);
      font-family: 'Inter', sans-serif;
      font-size: 14px;
    }}
    .top-banner a {{
      color: #7dd3fc;
      text-decoration: none;
      font-weight: 500;
      transition: color 0.15s;
    }}
    .top-banner a:hover {{ color: #ffffff; text-decoration: underline; }}
    .banner-title {{ font-weight: 700; letter-spacing: 0.02em; }}
    .banner-links {{ display: flex; gap: 20px; align-items: center; }}

    /* Layout Container */
    .paper-container {{
      max-width: 860px;
      margin: 40px auto 100px;
      padding: 0 24px;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
      color: var(--primary);
      font-family: 'Inter', sans-serif;
      font-weight: 700;
      letter-spacing: -0.01em;
      margin-top: 1.8em;
      margin-bottom: 0.5em;
    }}
    h1 {{
      font-size: 2.2rem;
      line-height: 1.25;
      color: #0b1e42;
      border-bottom: 3px solid var(--primary);
      padding-bottom: 16px;
      margin-top: 0.5em;
    }}
    h2 {{
      font-size: 1.45rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-top: 2.2em;
    }}
    h3 {{ font-size: 1.18rem; color: #1e3a8a; }}
    h4 {{ font-size: 1.05rem; }}

    p {{
      margin: 0 0 1.25em;
      text-align: justify;
      hyphens: auto;
    }}

    /* Abstract Callout Box */
    h2#2-abstract + p,
    h2:nth-of-type(2) + p {{
      background: var(--primary-light);
      border-left: 4px solid var(--primary);
      padding: 20px 24px;
      border-radius: 0 var(--radius) var(--radius) 0;
      font-size: 1.02rem;
      line-height: 1.75;
      font-style: italic;
      color: #0c2340;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 28px 0;
      font-family: 'Inter', sans-serif;
      font-size: 14.5px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    th {{
      background: var(--primary);
      color: #ffffff;
      padding: 12px 14px;
      text-align: left;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    td {{
      padding: 11px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
    tr:hover td {{ background: #f1f5f9; }}

    /* Code Blocks */
    code {{
      font-family: 'Space Mono', monospace;
      font-size: 14px;
      background: var(--code-bg);
      color: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 18px 20px;
      border-radius: var(--radius);
      overflow-x: auto;
      font-family: 'Space Mono', monospace;
      font-size: 13.5px;
      line-height: 1.6;
      margin: 24px 0;
    }}
    pre code {{ background: transparent; color: inherit; padding: 0; }}

    /* Figures */
    .paper-figure {{
      margin: 36px 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }}
    .paper-figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
    }}
    figcaption {{
      font-family: 'Inter', sans-serif;
      font-size: 13.5px;
      line-height: 1.55;
      color: var(--text-muted);
      margin-top: 14px;
      padding: 0 4px;
    }}

    /* Lists */
    ul, ol {{
      margin: 0 0 1.5em;
      padding-left: 28px;
    }}
    li {{ margin-bottom: 8px; }}

    /* Blockquotes */
    blockquote {{
      margin: 24px 0;
      padding: 16px 20px;
      background: #f8fafc;
      border-left: 4px solid var(--accent);
      color: #334155;
      font-style: italic;
      border-radius: 0 var(--radius) var(--radius) 0;
    }}
    blockquote p:last-child {{ margin-bottom: 0; }}

    /* Links */
    a {{
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 3px;
    }}
    a:hover {{ color: var(--primary); }}

    /* Footer */
    footer {{
      border-top: 1px solid var(--border);
      margin-top: 60px;
      padding-top: 24px;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      color: var(--text-muted);
      text-align: center;
    }}

    @media (max-width: 768px) {{
      body {{ font-size: 16px; }}
      h1 {{ font-size: 1.75rem; }}
      .top-banner {{ flex-direction: column; gap: 8px; align-items: flex-start; }}
      table {{ font-size: 13px; display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header class="top-banner">
    <div class="banner-title">FlyRank AI/ML Research · Capstone Paper</div>
    <div class="banner-links">
      <a href="https://github.com/Kathyayini26/flyrank-ml" target="_blank" rel="noopener">GitHub Repository</a>
      <a href="https://flyrank.ai" target="_blank" rel="noopener">FlyRank.ai</a>
    </div>
  </header>

  <!-- Main Academic Paper Content -->
  <main class="paper-container">
    {raw_html}
  </main>

  <footer>
    <p>Built on the FlyRank ML Internship dataset &middot; <a href="https://flyrank.ai" target="_blank" rel="noopener">https://flyrank.ai</a> &middot; November 2026</p>
  </footer>

</body>
</html>
"""

    output_html_path = docs_dir / "index.html"
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_page)
        
    print(f"Academic paper web page generated successfully at: {output_html_path}")
    print(f"File size: {len(html_page):,} characters across {html_page.count(chr(10))} lines.")

if __name__ == "__main__":
    generate_paper_html()

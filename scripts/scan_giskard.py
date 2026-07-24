"""Giskard vulnerability scan for one extraction task.

The pipeline measures how well a prompt scores; this asks the other question —
how does it break? Giskard probes the model for hallucination, prompt injection,
and robustness to reworded input, then writes an HTML report to results/.

    python -m scripts.scan_giskard --task concept_relevant --model haiku

Needs OPENAI_API_KEY for Giskard's own probe-generating LLM (GISKARD_SCAN_MODEL
to override). Falls back to a few synthetic briefs when briefs.yml is absent.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A few generic briefs so the scan runs even without the confidential dataset.
_SYNTHETIC_BRIEFS = [
    {"brief_id": "syn1", "text": "A subscription app that helps freelancers track "
     "billable hours and send invoices in one tap."},
    {"brief_id": "syn2", "text": "A smart water bottle that reminds office workers "
     "to drink and syncs intake to a phone app."},
    {"brief_id": "syn3", "text": "An online marketplace connecting local farmers "
     "with restaurants for same-day produce delivery."},
]


def _load_scan_briefs(limit: int) -> list[str]:
    """Real briefs when available, otherwise synthetic ones."""
    try:
        from src.prompt_builder import load_briefs
        briefs = load_briefs()
        texts = []
        for b in briefs[:limit]:
            # Briefs are dicts of free-form fields; join their values into one blob.
            texts.append("\n".join(str(v) for v in b.values() if isinstance(v, str)))
        if texts:
            return texts
    except Exception:
        pass
    return [b["text"] for b in _SYNTHETIC_BRIEFS][:limit]


def _instruction_for(task: str) -> str:
    from src.prompt_builder import load_prompts
    templates = load_prompts()
    tmpl = templates.get(task)
    if tmpl is None:
        raise SystemExit(
            f"No template for task={task!r}. Available: "
            f"{', '.join(sorted(templates))}"
        )
    return tmpl.instruction.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="concept_relevant",
                    help="extraction task to probe (default: concept_relevant)")
    ap.add_argument("--model", default="haiku",
                    help="model_key under test from config.MODELS (default: haiku)")
    ap.add_argument("--n-briefs", type=int, default=5,
                    help="how many briefs to seed the scan with")
    args = ap.parse_args()

    try:
        import giskard
        import pandas as pd
    except ImportError:
        print("Giskard not installed. Run:  pip install -r requirements-qe.txt")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY — Giskard's scanner needs an LLM to generate probes.")
        return 0

    from src.llm_client import LLMClient
    from src import config as cfg

    if args.model not in cfg.MODELS:
        raise SystemExit(f"Unknown model {args.model!r}. Known: {', '.join(cfg.MODELS)}")

    giskard.llm.set_llm_model(os.environ.get("GISKARD_SCAN_MODEL", "gpt-4o-mini"))

    instruction = _instruction_for(args.task)
    client = LLMClient.from_env()

    def predict(df: "pd.DataFrame") -> list[str]:
        outs = []
        for brief_text in df["brief"]:
            prompt = f"{instruction}\n\nBRIEF:\n{brief_text}"
            r = client.call(model_key=args.model, prompt=prompt, brief_id="giskard",
                            task=args.task, config_id="giskard-scan")
            outs.append(r.parsed_output or r.raw_response or "")
        return outs

    giskard_model = giskard.Model(
        model=predict,
        model_type="text_generation",
        name=f"prompt-eval:{args.task}",
        description=(
            f"Extraction task '{args.task}': given a product brief, the model "
            f"returns one concise sentence. Instruction: {instruction[:200]}"
        ),
        feature_names=["brief"],
    )
    dataset = giskard.Dataset(
        pd.DataFrame({"brief": _load_scan_briefs(args.n_briefs)}),
        name=f"briefs-for-{args.task}",
    )

    print(f"Scanning {args.model} on task '{args.task}' with {args.n_briefs} briefs...")
    report = giskard.scan(giskard_model, dataset)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"giskard_scan_{args.task}_{args.model}.html"
    report.to_html(str(out))
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

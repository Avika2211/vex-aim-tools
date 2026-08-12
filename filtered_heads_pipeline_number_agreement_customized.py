# ============================================================================
# PRONOUN COREFERENCE — SPECIALIST DISCOVERY PIPELINE
# Dynamic Range Compression Metric Isolation
# ============================================================================
# Evaluates 384 attention heads (24 layers x 16 heads, Flan-T5-Large Encoder)
# to isolate asymmetric target specialists while filtering non-responsive
# origin-clustered noise using a dynamic range compression threshold.
# ============================================================================

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from google.colab import files
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CONDITION_COLORS = {
    "Them_Condition": "#1f77b4",  # Blue
    "It_Condition": "#ff7f0e",  # Orange
}


def prompt_for_dataset():
    """Prompts user for dataset file or reuses existing local copy."""
    target_name = "dataset.json"
    if os.path.exists(target_name):
        print(f"Dataset file '{target_name}' found in workspace.")
        choice = input("Overwrite existing dataset? (y/N): ").strip().lower()
        if choice in ["yes", "y"]:
            uploaded = files.upload()
            if target_name not in uploaded and len(uploaded) > 0:
                uploaded_file_name = list(uploaded.keys())[0]
                os.rename(uploaded_file_name, target_name)
    else:
        print(f"Please upload the dataset file ('{target_name}'):")
        uploaded = files.upload()
        if target_name not in uploaded and len(uploaded) > 0:
            uploaded_file_name = list(uploaded.keys())[0]
            os.rename(uploaded_file_name, target_name)

    return target_name


class ModelManager:
    """Handles initialization and batched inference for attention extraction."""

    def __init__(self, model_name="google/flan-t5-large"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.config = None

    def load_model(self):
        print(f"Loading model '{self.model_name}' onto device target [{self.device}]...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.config = self.model.config
        print(f"Model loaded successfully on {self.device.upper()}.")

    def get_attention_outputs_batch(self, sentences, is_decoder_layer=False):
        inputs = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if not is_decoder_layer:
                outputs = self.model.encoder(**inputs, output_attentions=True)
                attentions = outputs.attentions
            else:
                decoder_inputs = self.model._shift_right(inputs["input_ids"])
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_inputs,
                    output_attentions=True,
                )
                attentions = outputs.decoder_attentions

        stacked_attentions = torch.stack([att.detach().cpu() for att in attentions])
        return stacked_attentions, inputs.input_ids.cpu()


class PronounCoreferenceScorer:
    """Extracts attention scores, calculates statistics, and builds reporting artifacts."""

    def __init__(self, model_manager):
        self.model = model_manager
        self.num_layers = self.model.config.num_layers
        self.num_heads = self.model.config.num_heads

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _head_word(phrase):
        words = phrase.strip().split()
        if not words:
            return None
        return words[-1].strip(",.?!")

    def find_token_index(self, tokens, word):
        w = word.lower().strip().strip(",.?!")
        for i, t in enumerate(tokens):
            cleaned = t.replace(" ", "").replace(" ", "").lower()
            if cleaned == w:
                return i
        for i, t in enumerate(tokens):
            cleaned = t.replace(" ", "").replace(" ", "").lower()
            if cleaned and (w in cleaned or cleaned in w):
                return i
        return -1

    def precompute_dataset_indices(self, data):
        precomputed = []
        prompts_1 = [item["prompt_1"] for item in data]
        prompts_2 = [item["prompt_2"] for item in data]

        print("Pre-computing attention maps for conditions 'them' and 'it'...")
        att_1, input_ids_1 = self.model.get_attention_outputs_batch(prompts_1)
        att_2, input_ids_2 = self.model.get_attention_outputs_batch(prompts_2)

        for idx, item in enumerate(data):
            head_1 = self._head_word(item["target_1"])
            head_2 = self._head_word(item["target_2"])
            if head_1 is None or head_2 is None:
                continue

            tok_1 = self.model.tokenizer.convert_ids_to_tokens(input_ids_1[idx])
            tok_2 = self.model.tokenizer.convert_ids_to_tokens(input_ids_2[idx])

            pron_1_i = self.find_token_index(tok_1, "them")
            correct_1_i = self.find_token_index(tok_1, head_1)
            distractor_1_i = self.find_token_index(tok_1, head_2)

            pron_2_i = self.find_token_index(tok_2, "it")
            correct_2_i = self.find_token_index(tok_2, head_2)
            distractor_2_i = self.find_token_index(tok_2, head_1)

            if -1 in (pron_1_i, correct_1_i, distractor_1_i, pron_2_i, correct_2_i, distractor_2_i):
                continue

            precomputed.append({
                "idx": idx,
                "pid": idx,
                "pron_1_i": pron_1_i,
                "correct_1_i": correct_1_i,
                "distractor_1_i": distractor_1_i,
                "pron_2_i": pron_2_i,
                "correct_2_i": correct_2_i,
                "distractor_2_i": distractor_2_i,
            })
        return precomputed, att_1, att_2

    def extract_scores_from_cache(self, precomputed, att_1, att_2, layer, head):
        pronoun_to_target, target_to_pronoun = [], []
        for item in precomputed:
            idx = item["idx"]
            A1 = att_1[layer, idx, head].numpy()
            A2 = att_2[layer, idx, head].numpy()

            pronoun_to_target.append({
                "x": float(A1[item["pron_1_i"], item["correct_1_i"]]),
                "y": float(A1[item["pron_1_i"], item["distractor_1_i"]]),
                "id": item["pid"],
                "condition": "Them_Condition",
            })
            target_to_pronoun.append({
                "x": float(A1[item["correct_1_i"], item["pron_1_i"]]),
                "y": float(A1[item["distractor_1_i"], item["pron_1_i"]]),
                "id": item["pid"],
                "condition": "Them_Condition",
            })

            pronoun_to_target.append({
                "x": float(A2[item["pron_2_i"], item["correct_2_i"]]),
                "y": float(A2[item["pron_2_i"], item["distractor_2_i"]]),
                "id": item["pid"],
                "condition": "It_Condition",
            })
            target_to_pronoun.append({
                "x": float(A2[item["correct_2_i"], item["pron_2_i"]]),
                "y": float(A2[item["distractor_2_i"], item["pron_2_i"]]),
                "id": item["pid"],
                "condition": "It_Condition",
            })

        return pronoun_to_target, target_to_pronoun

    def compute_ratio_stats(self, points, min_activation=0.01):
        global_stats = self._calculate_metrics_chunk(points, min_activation)
        condition_breakdown = {}
        for cond_name in CONDITION_COLORS.keys():
            sub_points = [p for p in points if p["condition"] == cond_name]
            condition_breakdown[cond_name] = self._calculate_metrics_chunk(sub_points, min_activation)
        return {"global": global_stats, "conditions": condition_breakdown}

    def _calculate_metrics_chunk(self, points, min_activation):
        above, below, deviations = [], [], []
        xs = np.array([p["x"] for p in points])
        ys = np.array([p["y"] for p in points])

        r2_val = 0.0
        if len(ys) > 1:
            ss_res = np.sum((ys - xs) ** 2)
            y_mean = np.mean(ys)
            ss_tot = np.sum((ys - y_mean) ** 2)
            if ss_tot > 0:
                r2_val = float(1.0 - (ss_res / ss_tot))

        for p in points:
            x, y = p["x"], p["y"]
            deviations.append((x - y) ** 2)
            if x < min_activation and y < min_activation:
                continue
            ratio = (x + 1e-8) / (y + 1e-8)
            if x > y:
                above.append(ratio)
            elif x < y:
                below.append(ratio)

        msd = np.mean(deviations) if deviations else 0.0
        y_var = np.var(ys) if len(ys) > 0 else 0.0
        nmsd_val = float(msd / (y_var + 1e-8)) if deviations else 0.0

        stats = {
            "num_above": len(above),
            "num_below": len(below),
            "mean_above": float(np.mean(above)) if above else 0.0,
            "mean_below": float(np.mean(below)) if below else 0.0,
            "nmsd_from_line": nmsd_val,
            "r2": r2_val,
        }
        total = len(above) + len(below)
        stats["favor_correct_pct"] = (100 * len(above) / total) if total else 0
        return stats

    def is_non_responsive(self, points):
        if len(points) == 0:
            return True
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        return (max(xs) - min(xs) < 0.1) and (max(ys) - min(ys) < 0.1)

    def plot_dual_scatter(self, pt_pts, tp_pts, global_head_idx, pt_dead, tp_dead):
        os.makedirs("scatter_plots", exist_ok=True)
        png_out = f"scatter_plots/head_{global_head_idx}.png"
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))

        def render_single(ax_sub, pts, label_title):
            if len(pts) == 0:
                ax_sub.set_title(label_title, fontsize=8)
                return
            xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
            for cond, col in CONDITION_COLORS.items():
                c_pts = [p for p in pts if p["condition"] == cond]
                if c_pts:
                    ax_sub.scatter([], [], color=col, label=cond.replace("_Condition", ""))
                    for p in c_pts:
                        ax_sub.text(p["x"], p["y"], str(p["id"]), fontsize=6, fontweight="bold", color=col, ha="center", va="center")
            mn, mx = min(xs + ys), max(xs + ys)
            ax_sub.plot([mn, mx], [mn, mx], "--", color="black", alpha=0.5)
            pad = max((mx - mn) * 0.05, 1e-5)
            ax_sub.set_xlim(mn - pad, mx + pad)
            ax_sub.set_ylim(mn - pad, mx + pad)
            ax_sub.grid(True, linestyle=":", alpha=0.5)
            ax_sub.set_title(label_title, fontsize=8)

        render_single(ax[0], [] if pt_dead else pt_pts, f"Head {global_head_idx}: Pronoun → Target")
        render_single(ax[1], [] if tp_dead else tp_pts, f"Head {global_head_idx}: Target → Pronoun")

        plt.tight_layout()
        plt.savefig(png_out, dpi=130, bbox_inches="tight")
        plt.close()
        return png_out

    def generate_pdf_report(self, master_results, filename="pronoun_coreference_encoder_report.pdf"):
        print(f"\nGenerating summary PDF report ({filename})...")
        doc = SimpleDocTemplate(filename)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("Pronoun Coreference — Isolated Target Specialists Report", styles["Title"]),
            Spacer(1, 15),
        ]

        for r in master_results:
            title_text = f"<b>Attention Head {r['global_head_idx']} (Encoder Layer {r['layer']}, Local Head {r['local_head_idx']}) [Target Specialist]</b>"
            elements.append(Paragraph(title_text, styles["Heading1"]))
            elements.append(Spacer(1, 6))

            for label, dead, metrics in [
                ("Pronoun → Target", r["pt_dead"], r["pt_metrics"]),
                ("Target → Pronoun", r["tp_dead"], r["tp_metrics"]),
            ]:
                elements.append(Paragraph(f"<b>{label}:</b>", styles["Heading2"]))
                g = metrics["global"]
                g_msg = (
                    "[Non-Responsive]"
                    if dead
                    else f"NMSD: {g['nmsd_from_line']:.4f} | Favor-Correct Mean: {g['mean_above']:.2f} | Favor-Correct %: {g['favor_correct_pct']:.1f}% | R²: {g['r2']:.4f}"
                )
                elements.append(Paragraph(f"• <b>Global:</b> {g_msg}", styles["BodyText"]))

            if os.path.exists(r["plot_path"]):
                img = Image(r["plot_path"], width=480, height=220)
                img.hAlign = "CENTER"
                elements.append(img)
            elements.append(PageBreak())

        doc.build(elements)
        print(f"Report successfully saved to '{filename}'.")


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    dataset_file = prompt_for_dataset()
    manager = ModelManager()
    manager.load_model()
    scorer = PronounCoreferenceScorer(manager)

    dataset = scorer.load_dataset(dataset_file)
    MIN_ACTIVATION_VALUE = 0.01
    master_results_cache = []

    print("\n[Phase 1/2] Pre-computing global attention maps...")
    precomputed, cache_att_1, cache_att_2 = scorer.precompute_dataset_indices(dataset)

    print("\n[Phase 2/2] Evaluating 384 Encoder Attention Heads...")
    print("-" * 80)

    for global_idx in tqdm(range(384), desc="Evaluating Attention Heads"):
        local_layer = global_idx // 16
        local_head = global_idx % 16

        pt_pts, tp_pts = scorer.extract_scores_from_cache(precomputed, cache_att_1, cache_att_2, local_layer, local_head)
        pt_dead, tp_dead = scorer.is_non_responsive(pt_pts), scorer.is_non_responsive(tp_pts)

        if pt_dead and tp_dead:
            continue

        stats_pt = scorer.compute_ratio_stats(pt_pts, min_activation=MIN_ACTIVATION_VALUE)
        stats_tp = scorer.compute_ratio_stats(tp_pts, min_activation=MIN_ACTIVATION_VALUE)

        # Exclusive signature verification with dynamic 1/5th compression check
        def verify_exclusive_signature(stats_dict, points):
            if len(points) == 0:
                return False

            g_metrics = stats_dict["global"]
            c_breakdown = stats_dict["conditions"]

            them_stats = c_breakdown.get("Them_Condition", {})
            it_stats = c_breakdown.get("It_Condition", {})

            pct_gap = abs(them_stats["favor_correct_pct"] - it_stats["favor_correct_pct"])
            mean_gap = abs(them_stats["mean_above"] - it_stats["mean_above"])

            # Dynamic 1/5th Range Compression Check
            xs = np.array([p["x"] for p in points])
            ys = np.array([p["y"] for p in points])

            q3_x = np.percentile(xs, 75)
            q3_y = np.percentile(ys, 75)

            # Filter out heads where 75% of activations remain clustered near origin (<0.2)
            is_clustered_at_origin = (q3_x < 0.2) and (q3_y < 0.2)
            escapes_origin_cluster = not is_clustered_at_origin

            is_matched = (
                escapes_origin_cluster
                and (0.5 < g_metrics["nmsd_from_line"])
                and (g_metrics["r2"] < -0.05)
                and (14.0 <= g_metrics["favor_correct_pct"] <= 89.9)
                and (pct_gap > 12.0 or mean_gap > 1.0)
            )
            return is_matched

        pt_specialist = (not pt_dead) and verify_exclusive_signature(stats_pt, pt_pts)
        tp_specialist = (not tp_dead) and verify_exclusive_signature(stats_tp, tp_pts)

        is_jackpot = pt_specialist ^ tp_specialist
        is_relevant_node = pt_specialist or tp_specialist

        if is_jackpot:
            tqdm.write(f"Target Specialist Identified: Head {global_idx} (Layer {local_layer}, Local Head {local_head})")
            plot_path = scorer.plot_dual_scatter(pt_pts, tp_pts, global_idx, pt_dead, tp_dead)

            master_results_cache.append({
                "global_head_idx": global_idx,
                "layer": local_layer,
                "local_head_idx": local_head,
                "component": "Encoder",
                "is_relevant": is_relevant_node,
                "is_jackpot": is_jackpot,
                "plot_path": plot_path,
                "pt_dead": pt_dead,
                "pt_metrics": stats_pt,
                "tp_dead": tp_dead,
                "tp_metrics": stats_tp,
            })

    if master_results_cache:
        scorer.generate_pdf_report(master_results_cache)
    else:
        print("Pipeline execution complete. No heads matched the target specialist filtering criteria.")
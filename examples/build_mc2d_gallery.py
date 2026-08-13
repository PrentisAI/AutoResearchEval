"""Build a gallery of mc2d trajectories in the 26 STANDARD verifier-bound actions.

Same strategy as examples/mc2d_standard_actions.py (one episode), generalized to a
SET of materials with different SCF error→recovery strategies. For each chosen
restart episode we:

  1. reconstruct the real restart chain (adapters.aiida_walker)
  2. lift it onto the named SCF recovery tool space, then re-express in the 26
     standard actions (mc2d_standard_actions.to_standard) — reconstruction mode,
     actions/observations are the REAL recorded QE execution
  3. gate it: faithful reload + per-action verifiers + CHGNet physics → admissible
  4. visualize three ways into docs/images/mc2d/<slug>/:
       structure.png   — the real crystal structure (rendered from archive atoms)
       dag.png         — the raw AiiDA provenance DAG (dynamic topology extraction)
       trajectory.png  — the reconstructed 26-action agent trajectory
  5. write a per-episode page README.md (the "viz page")

Finally it writes docs/images/mc2d/README.md (gallery index) linking every material.

Run (needs `scicoder` env + read-only `mc2d` profile; CHGNet on CPU):
    /home/ubuntu/miniconda3/envs/scicoder/bin/python examples/build_mc2d_gallery.py
"""
from __future__ import annotations

import importlib
import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

# importing the standard-actions example triggers load_profile("mc2d") via its mc import
S = importlib.import_module("mc2d_standard_actions")
mc = S.mc
from aiida import orm  # noqa: E402
from adapters import aiida_walker as W  # noqa: E402
from reconstruct.tool_lift import is_phonon_chain, lift_restart_chain_named  # noqa: E402
from reconstruct import thought_completion  # noqa: E402
from reconstruct.base import stamp_provenance  # noqa: E402
from reconstruct.llm_openrouter import OpenRouterClient  # noqa: E402
from ir import Trajectory, Verification  # noqa: E402
from export import to_sft_react  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

for _fp in ["/tmp/noto_cjk.otf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

GALLERY = ROOT / "docs" / "images" / "mc2d"
RED, GREEN, BLUE, GREY, DICTC, OUTC = "#e8746b", "#5aa469", "#6d9dc5", "#cfd3d6", "#f2d49b", "#bcd6a7"
ACOL = {"run_dft": "#6d9dc5", "triage_failure": "#c79bd6", "custodian_correct": "#e0a85e",
        "check_convergence": "#7fb88f", "analyze_dft_output": "#5aa469"}

# candidate episodes (root pk → recovery flavor), chosen for material + strategy diversity.
# All start from a failed first attempt; the build keeps whichever pass admissibility.
CANDIDATES = [60615, 75248, 29926, 86259,           # known-good (R-3m, I4/mmm, Pmmn, P4/nmm)
              40686, 99820, 100989, 2205, 90888, 39648, 78234, 59215]
TARGET = 6


# --------------------------------------------------------------------------- #
# LLM thought reconstruction — simulate a computational-materials scientist's
# diagnostic reasoning (not template boilerplate), gated for honesty (§1.2, §11).
# --------------------------------------------------------------------------- #
SCIENTIST_SYSTEM = (
    "You are an experienced computational-materials scientist running plane-wave DFT "
    "(Quantum ESPRESSO) on 2D materials and recovering from SCF convergence failures. "
    "For each logged action, reconstruct the concise FIRST-PERSON reasoning you had "
    "*before* running it — the diagnosis and decision a real DFT practitioner would make. "
    "Ground it in the concrete numbers shown (how many SCF iterations ran, the recovery "
    "knob being changed and its value) and in the chemistry of THIS system (e.g. f-electron "
    "lanthanides / open-shell magnetic ions admit multiple SCF solutions and converge the "
    "charge density slowly; heavy polarizable chalcogenides & halides often need gentler "
    "linear mixing). You MAY put forward a likely cause TENTATIVELY ('likely…', 'this looks "
    "like…', 'consistent with…') as your working hypothesis for choosing the move. Hard rules: "
    "(1) never say the run converged/succeeded on a step whose status shows it did NOT; "
    "(2) the FIRST action has no previous run — do not call it a restart, recovery, or continuation; "
    "(3) use ONLY what was known before the action ran — never cite a later step's outcome as if "
    "already known; "
    "(4) justify the EXACT action and parameters shown — if the move restarts from the saved charge "
    "density WITHOUT changing mixing_beta, your reasoning must be about why a plain restart helps, NOT "
    "about lowering a mixing factor you are not lowering; only talk about reducing mixing_beta when the "
    "action's params actually reduce it. You may note a different knob as a *future* contingency ('if "
    "this restart fails too, I'll lower mixing next'), but never describe the current move as changing a "
    "knob it does not change. One concise Thought, first person, no preamble, no 'Thought:' label."
)


def _scientist_prompt(traj, i):
    step = traj.steps[i]
    past = [{"action": s.action.name, "params": s.action.params, "observation": s.observation.text}
            for s in traj.steps[:i]]
    future = [{"action": s.action.name} for s in traj.steps[i + 1:]]
    first = ("This is the FIRST action — no previous run exists; it is the initial SCF, not a restart.\n"
             if i == 0 else "")
    return (
        f"Goal: {traj.goal}\n"
        f"Past steps (what you already saw): {json.dumps(past, sort_keys=True, ensure_ascii=False)}\n"
        f"{first}"
        f"Current action: {step.action.name}({json.dumps(step.action.params, ensure_ascii=False)})\n"
        f"Current observation (the only facts recorded): {step.observation.text}\n"
        f"Later actions (names only, for flow — their results are NOT known to you yet): "
        f"{json.dumps(future, ensure_ascii=False)}\n"
        "Write the diagnostic reasoning that justifies the CURRENT action AND ITS EXACT PARAMETERS, "
        "grounded in the SCF iteration count and this material's chemistry, using only what was known "
        "before it ran. Do not claim to change a knob the current action does not change."
    )


# Unambiguous positive convergence CLAIMS — only these flag a failed step (negation-aware,
# so a scientist saying "did not converge" / "to help it converge" is NOT mis-flagged).
_POS_CONVERGED = ("now converged", "has converged", "have converged", "successfully converged",
                  "converged successfully", "scf converged", "it converged", "run converged",
                  "finally converged", "reached convergence", "is converged", "did converge")
# positive "this action IS a restart" claims — only valid as a violation on step 0
_FIRST_RESTART_CLAIMS = ("restarting from", "restart from the", "continuing the previous",
                         "continuing from the previous", "recovering from the", "from the previous run",
                         "resuming the", "picking up from", "continue the previous")


def scientist_thought_ok(traj):
    """Honest gate for the reconstructed scientist thoughts (§1.2): a real error→recovery
    chain whose thoughts don't (a) ASSERT convergence on a failed step, (b) treat the first
    action as a restart, or (c) leak a later outcome. Mechanism HYPOTHESES are allowed —
    a scientist reasons that way — but claiming the run converged when it didn't is not.
    Returns (ok, reason) so the caller can log/retry (thought completion is sampled)."""
    exits = [s.observation.exit_status for s in traj.steps]
    if not (exits[-1] == 0 and any(e not in (0, None) for e in exits[:-1])):
        return False, "not a real error→recovery chain"
    for i, s in enumerate(traj.steps):
        th = (s.thought or "").lower()
        if not th:
            return False, f"step{i}: empty thought"
        if s.observation.exit_status not in (0, None) and any(p in th for p in _POS_CONVERGED):
            return False, f"step{i}: claims convergence on a FAILED step"
        # step0 has no prior run: flag only a POSITIVE restart claim (negation-robust —
        # "this is a fresh run, no saved density to continue from" must NOT trip).
        if i == 0 and any(p in th for p in _FIRST_RESTART_CLAIMS):
            return False, "step0: treats the first action as a restart"
    return True, "ok"


def formula_math(formula: str) -> str:
    """'Sb8Te9' → r'$\\mathrm{Sb_8Te_9}$' (mathtext subscripts; CJK fonts lack U+2088…)."""
    if not formula:
        return "a 2D material"
    return r"$\mathrm{" + re.sub(r"(\d+)", r"_{\1}", formula) + r"}$"


def task_text(formula: str) -> str:
    return (f"TASK: 对 2D 材料 {formula_math(formula)} 跑平面波 DFT 自洽(SCF),"
            "反复 SCF 不收敛时通过重启/调整恢复,直到收敛并报告总能量")


# --------------------------------------------------------------------------- #
# small drawing helpers
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, txt, fc, ec="#444", fs=8.5, bold=False, r=0.08, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=z,
                 boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=z + 0.1,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, c="#555", ls="-", lw=1.4, style="-|>", z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, zorder=z,
                 mutation_scale=12, color=c, ls=ls, lw=lw, shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------- #
# dynamic provenance-DAG extraction (works for any restart chain)
# --------------------------------------------------------------------------- #
def _scf_iters(res):
    v = res.get("total_number_of_scf_iterations") or res.get("scf_iterations")
    if isinstance(v, (list, tuple)):
        return v[-1] if v else None
    return v


def extract_dag(chain):
    """[CalcJobNode] → list of per-attempt dicts + the shared input-structure pk/formula."""
    rows, struct_pk = [], None
    for i, c in enumerate(chain):
        inc = {t.link_label: t.node for t in c.base.links.get_incoming().all()}
        out = {t.link_label: t.node for t in c.base.links.get_outgoing().all()}
        pd = inc["parameters"].get_dict() if "parameters" in inc else {}
        res = out["output_parameters"].get_dict() if "output_parameters" in out else {}
        if i == 0 and "structure" in inc:
            struct_pk = inc["structure"].pk
        rows.append({
            "pk": c.pk, "exit": c.exit_status,
            "mixing_beta": (pd.get("ELECTRONS") or {}).get("mixing_beta"),
            "restart_mode": (pd.get("CONTROL") or {}).get("restart_mode"),
            "scf_it": _scf_iters(res), "energy": res.get("energy"),
            "param_pk": inc["parameters"].pk if "parameters" in inc else None,
            "remote_pk": out["remote_folder"].pk if "remote_folder" in out else None,
            "outstruct_pk": out["output_structure"].pk if "output_structure" in out else None,
        })
    return rows, struct_pk


def render_dag(rows, formula, path):
    n = len(rows)
    xs = [1.0 + i * 2.7 for i in range(n)]
    W_fig = max(11.0, 2.0 + n * 2.7)
    fig, ax = plt.subplots(figsize=(W_fig, 9.2))
    ax.axis("off")
    ax.set_xlim(-1.6, xs[-1] + 1.6)
    ax.set_ylim(-3.4, 6.0)
    cx = (xs[0] + xs[-1]) / 2
    ax.text(cx, 5.7, f"① {formula} — 原始 AiiDA 溯源 DAG(原料 · 真实 QE 执行记录)",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(cx, 5.15, task_text(formula), ha="center", fontsize=11, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff6e0", ec="#d9b34a"))
    yc = 0.6
    bw = min(max(6.0, n * 2.0), xs[-1] - xs[0] + 4.2)
    ax.add_patch(FancyBboxPatch((cx - bw / 2, 3.4), bw, 1.1, zorder=4,
                 boxstyle="round,pad=0.02,rounding_size=0.1", fc="#eaf2f8", ec=BLUE, lw=1.6, ls="--"))
    ax.text(cx, 4.23, "共享输入(每次尝试复用同一批节点 → 只有重启指针在变)",
            ha="center", fontsize=10, color="#2f6690", fontweight="bold", zorder=4.1)
    ax.text(cx, 3.73, f"StructureData {formula_math(formula)}  ·  KpointsData  ·  pseudopotentials  ·  Code  ·  vdw_table",
            ha="center", fontsize=8.8, color="#2f6690", zorder=4.1)
    for i, x in enumerate(rows and xs):
        r = rows[i]
        col = GREEN if r["exit"] == 0 else RED
        _arrow(ax, cx, 3.4, x, yc + 0.72, c=BLUE, ls=":", lw=0.7, style="-", z=0.5)
        mb = f"mixing_beta={r['mixing_beta']}" if r["mixing_beta"] is not None else ""
        _box(ax, x, 2.5, 2.3, 1.0,
             f"Dict parameters\npk {r['param_pk']}\nrestart_mode={r['restart_mode']}\n{mb}", DICTC, fs=7.2)
        _arrow(ax, x, 2.0, x, yc + 0.72, c="#9a7b32")
        _box(ax, x, yc, 2.3, 1.4,
             f"CalcJob (pw.x)\npk {r['pk']}\n尝试 {i}\nexit_status={r['exit']}\nSCF 迭代={r['scf_it']}",
             col, ec="#7a2b25" if r["exit"] else "#2f5e3a", fs=7.6, bold=True)
        _box(ax, x, -1.45, 1.95, 0.85, f"RemoteData\nremote_folder\npk {r['remote_pk']}", GREY, fs=7.2)
        _arrow(ax, x, yc - 0.72, x, -1.05, c="#666")
        etag = f"E={r['energy']:.3f} eV" if isinstance(r["energy"], (int, float)) and r["exit"] == 0 else "(无收敛能量)"
        _box(ax, x, -2.6, 2.05, 0.72, f"output_structure pk {r['outstruct_pk']}\n{etag}",
             OUTC if r["exit"] == 0 else "#e6e6e6", fs=6.8)
        _arrow(ax, x + 0.6, yc - 0.72, x + 0.6, -2.24, c="#7a9a5a", lw=0.9)
        if i < n - 1:
            _arrow(ax, x + 0.98, -1.45, xs[i + 1] - 1.18, yc - 0.25, c="#c0392b", lw=2.0)
            ax.text((x + xs[i + 1]) / 2, -0.6, "parent_calc_folder\n(restart 指针)",
                    ha="center", fontsize=6.6, color="#c0392b", fontweight="bold")
    ax.text(cx, -3.25, "红=SCF 未收敛(失败分支,刻意保留) · 绿=收敛 · "
            "红色脊柱 = remote_folder→parent_calc_folder 串起的真实重启链 = 字面意义的轨迹",
            ha="center", fontsize=8.8, color="#444")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def render_trajectory(ir, formula, path):
    steps = ir["steps"]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(15.8, max(6, 1.6 + n * 1.0)))
    ax.axis("off")
    ax.set_xlim(0, 15.8)
    ax.set_ylim(-0.6, n + 2.7)
    top = n + 2.0
    ax.text(7.9, top + 0.42, f"② {formula} — 重建后的 Agent 轨迹(产物 · 26 标准动作)",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(7.9, top - 0.2, task_text(formula), ha="center", fontsize=11, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff6e0", ec="#d9b34a"))
    for k, s in enumerate(steps):
        y = top - 1.15 - k * 0.95
        a = s["action"]
        name = a["name"]
        fb = s.get("is_failure_branch")
        col = ACOL.get(name, "#cccccc")
        ec = "#b03a2e" if fb else "#444"
        lw = 2.4 if fb else 1.3
        h = 0.82
        ax.add_patch(FancyBboxPatch((0.6, y - h / 2), 9.0, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=col, ec=ec, lw=lw, alpha=0.92))
        pr = json.dumps(a["params"], ensure_ascii=False)
        if len(pr) > 60:
            pr = pr[:58] + "…"
        ax.text(1.0, y + 0.16, f"step {s['index']}  ▶ {name}", fontsize=11, fontweight="bold", va="center")
        ax.text(1.0, y - 0.17, pr, fontsize=8.0, va="center", family="monospace", color="#222")
        if fb:
            ax.text(9.35, y, "失败\n分支", fontsize=7.6, ha="right", va="center", color="#fff",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#b03a2e", ec="none"))
        th = (s.get("thought") or "").strip()
        ob = (s["observation"]["text"] or "").replace("Observation: ", "").strip()
        if len(th) > 72:
            th = th[:70] + "…"
        if len(ob) > 68:
            ob = ob[:66] + "…"
        ax.text(9.95, y + 0.16, f"Thought: {th}", fontsize=7.9, va="center", color="#33438a")
        ax.text(9.95, y - 0.17, f"Obs: {ob}", fontsize=7.9, va="center", color="#555")
        rr = a.get("raw_ref")
        if rr:
            ax.text(0.62, y - h / 2 - 0.12, f"← 来自真实溯源 {rr}", fontsize=6.4, color="#999")
        if k < n - 1:
            _arrow(ax, 5.1, y - h / 2 - 0.02, 5.1, y - h / 2 - 0.28, c="#888", lw=1.4)
    ax.text(7.9, 0.1, "Thought = LLM 重建的科学家诊断推理(非模板),经 STaR/未来泄露门控。"
            "失败分支全保留;入库门控:faithful reload + per-action verifier + CHGNet 物理 → admissible ✓",
            ha="center", fontsize=8.8, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef3ee", ec="#9bb89b"))
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def render_verdi(leaf_pk, outdir):
    """Native, unannotated AiiDA provenance graph via `verdi node graph generate`
    (needs the graphviz `dot` binary + PyCifRW). Writes provenance_verdi.png; returns
    True on success. This is the most faithful view — every shared input/output node
    and link, exactly as archived (contrast the distilled dag.png)."""
    import os
    bindir = Path(sys.executable).parent           # env bin holds both `verdi` and `dot`
    verdi = bindir / "verdi"
    env = {**os.environ, "PATH": f"{bindir}:{os.environ.get('PATH', '')}"}  # so graphviz finds `dot`
    try:
        subprocess.run(
            [str(verdi), "-p", "mc2d", "node", "graph", "generate", str(leaf_pk),
             "--link-types", "data", "--ancestor-depth", "20", "--descendant-depth", "1",
             "--identifier", "pk", "--output-format", "png"],
            cwd=str(outdir), env=env, check=True, capture_output=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        print(f"    (verdi graph for pk {leaf_pk} failed: {type(e).__name__})")
        return False
    src = outdir / f"{leaf_pk}.dot.png"
    if src.exists():
        src.replace(outdir / "provenance_verdi.png")
        return True
    return False


def render_structure(struct_pk, formula, path):
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors
    from ase.visualize.plot import plot_atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    pmg = orm.load_node(struct_pk).get_pymatgen()
    sga = SpacegroupAnalyzer(pmg, symprec=1e-2)
    sg_sym, sg_num = sga.get_space_group_symbol(), sga.get_space_group_number()
    try:
        conv = sga.get_conventional_standard_structure()
    except Exception:  # noqa: BLE001
        conv = pmg
    atoms = AseAtomsAdaptor.get_atoms(conv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    plot_atoms(atoms.repeat((1, 2, 1)), axes[0], rotation="90x,0y,0z", radii=0.55, show_unit_cell=2)
    axes[0].set_title("侧视图", fontsize=12)
    axes[0].axis("off")
    plot_atoms(atoms.repeat((2, 2, 1)), axes[1], rotation="0x,0y,0z", radii=0.55, show_unit_cell=2)
    axes[1].set_title("俯视图", fontsize=12)
    axes[1].axis("off")
    species = sorted({str(s.symbol) for s in conv.species})
    handles = [mpatches.Patch(color=jmol_colors[atomic_numbers[s]], label=s) for s in species]
    fig.legend(handles=handles, loc="lower center", ncol=len(species), fontsize=12, frameon=False)
    fig.suptitle(f"{formula_math(formula)}  (空间群 {sg_sym}, #{sg_num}) — mc2d StructureData pk {struct_pk}  ·  真实结构",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"{sg_sym} (#{sg_num})"


# --------------------------------------------------------------------------- #
# per-episode build
# --------------------------------------------------------------------------- #
def build_episode(root_pk, llm):
    chain = W.restart_chain(orm.load_node(root_pk))
    traj = W.restart_chain_to_trajectory(chain, trajectory_id=f"aiida-restart-{chain[0].pk}-{chain[-1].pk}")
    if not traj.metadata.get("recovered") or is_phonon_chain(traj) or not mc.is_error_recovery(traj):
        return None
    formula = mc.root_formula(traj)
    lift_restart_chain_named(traj, formula=formula)
    std_steps, checks = S.to_standard(traj)
    faith = mc.faithfulness_ok(traj)
    std_ok = all(all(v) for v in checks.values())
    phys_ok, phys = mc.mlip_physics_ok(traj)
    phys = mc._native(phys)
    std = Trajectory(
        id=f"mc2d-standard-{traj.id}", goal=S.recovery_goal(traj, formula),
        system_prompt=S.SYSTEM_PROMPT, steps=std_steps,
        provenance=traj.provenance, metadata={**traj.metadata, "action_vocab": "registry-26"},
        verification=Verification(reexecuted=faith, reexecute_reproduced=faith,
                                  judge_votes=[std_ok, phys_ok], min_judges=2,
                                  physics={**phys, "standard_action_checks": {k: all(v) for k, v in checks.items()}}))

    # replace the template thoughts with REAL LLM-reconstructed scientist reasoning, then
    # STaR-gate for honesty (§1.2). Thought completion is SAMPLED, so a borderline thought
    # occasionally trips the gate — retry a few times before giving up (keeps the episode set
    # stable instead of dropping a good chain to sampling noise).
    kept = False
    for attempt in range(4):
        log = thought_completion.complete_thoughts(
            std, llm, overwrite=True, prompt_ref="scientist_thought/v1",
            system=SCIENTIST_SYSTEM, prompt_fn=_scientist_prompt)
        ok, reason = scientist_thought_ok(std)
        if ok:
            stamp_provenance(std, log)
            kept = True
            break
        print(f"  [{formula}] root {root_pk}: thought gate retry {attempt + 1}/4 — {reason}")
    if not kept:
        print(f"  [{formula}] root {root_pk}: thought gate FAILED after retries — skip")
        return None
    std.metadata["thoughts"] = {"source": "llm_reconstructed", "teacher": llm.model_id,
                                "gate": "scientist_thought_ok (STaR/future-leak)"}

    if not std.is_admissible():
        print(f"  [{formula}] root {root_pk}: not admissible (faith={faith} std={std_ok} phys={phys_ok}) — skip")
        return None

    slug = f"{formula}_{root_pk}"
    d = GALLERY / slug
    d.mkdir(parents=True, exist_ok=True)
    ir = json.loads(std.model_dump_json())
    (d / "trajectory_ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2))
    to_sft_react.export_jsonl([std], str(d / "trajectory_sft.jsonl"), require_admissible=True)

    rows, struct_pk = extract_dag(chain)
    render_dag(rows, formula, d / "dag.png")
    render_trajectory(ir, formula, d / "trajectory.png")
    sg = render_structure(struct_pk, formula, d / "structure.png") if struct_pk else "?"
    has_verdi = render_verdi(chain[-1].pk, d)

    seq = " → ".join(s["action"]["name"] for s in ir["steps"])
    exits = [r["exit"] for r in rows]
    meta = {"formula": formula, "slug": slug, "root_pk": root_pk, "leaf_pk": chain[-1].pk,
            "struct_pk": struct_pk, "spacegroup": sg, "n_attempts": len(chain),
            "n_steps": len(ir["steps"]), "exits": exits, "goal": ir["goal"],
            "max_force": phys.get("max_force"), "seq": seq, "has_verdi": has_verdi,
            "teacher": llm.model_id}
    _write_page(d, meta, ir)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))  # for no-LLM page refresh
    mf = meta["max_force"]
    print(f"  [{formula:9s}] root {root_pk}: ADMIT  steps={meta['n_steps']} exits={exits} "
          f"sg={sg} maxF={mf:.3f} → {slug}/")
    return meta


def _write_page(d, m, ir):
    verdi = ("\n## ①′ 原生 verdi 溯源图（最忠实,未标注,含所有共享输入/输出节点与链接）\n"
             f"`verdi -p mc2d node graph generate {m['leaf_pk']} --link-types data --ancestor-depth 20`\n"
             "![verdi](./provenance_verdi.png)\n") if m.get("has_verdi") else ""
    thoughts = _thoughts_md(ir)
    md = f"""# {m['formula']} — mc2d 轨迹可视化

**Task / 任务**：{ir['goal']}

- 来源 episode：`{m['slug']}`（root pk {m['root_pk']} → leaf pk {m['leaf_pk']}），空间群 {m['spacegroup']}
- 重启尝试数 {m['n_attempts']}（exit 链 {m['exits']}）→ 重建为 {m['n_steps']} 步标准动作
- 动作序列：`{m['seq']}`
- Thought：**LLM 重建的科学家诊断推理**(teacher `{m.get('teacher', '?')}`),经 STaR/未来泄露门控(`scientist_thought_ok`)
- 门控：faithful reload + per-action verifier + CHGNet 物理（maxF={m['max_force']:.3f} eV/Å）→ **admissible ✓**
- 数据：[`trajectory_ir.json`](./trajectory_ir.json) · [`trajectory_sft.jsonl`](./trajectory_sft.jsonl)

## 材料结构（真实结构,由 archive 原子渲染）
[![structure](./structure.png)](https://mc2d.materialscloud.org/)

> 来源：[Materials Cloud MC2D 数据库](https://mc2d.materialscloud.org/)（[Archive 2024.157](https://archive.materialscloud.org/record/2024.157)）。图由归档结构 pk {m['struct_pk']} 渲染。

## ① 原始 AiiDA 溯源 DAG（原料 · 蒸馏标注版）
![dag](./dag.png)
{verdi}
## ② 重建后的 Agent 轨迹（产物 · 26 标准动作）
![trajectory](./trajectory.png)

### 轨迹推理全文（LLM 重建的科学家诊断 · 图中为节省空间被截断,这里是全文）
{thoughts}
---
由 [`examples/build_mc2d_gallery.py`](../../../../examples/build_mc2d_gallery.py) 生成。
"""
    (d / "README.md").write_text(md)


def _thoughts_md(ir):
    out = []
    for s in ir["steps"]:
        fb = " 🔴失败分支" if s.get("is_failure_branch") else ""
        a = s["action"]
        params = json.dumps(a["params"], ensure_ascii=False)
        out.append(f"**step {s['index']} · `{a['name']}({params})`**{fb}\n\n> {s.get('thought') or ''}\n")
    return "\n".join(out)


def _write_gallery(metas):
    rows = "\n".join(
        f"| [{m['formula']}](./{m['slug']}/) | {m['spacegroup']} | {m['n_steps']} 步 "
        f"({m['n_attempts']} 次尝试) | `{m['seq'].split(' → ')[0]} → … → {m['seq'].split(' → ')[-1]}` | "
        f"[![structure](./{m['slug']}/structure.png)](./{m['slug']}/) |"
        for m in metas)
    md = f"""# mc2d 轨迹画廊(26 标准动作空间)

每条都是从只读 mc2d 数据库(Materials Cloud 2024.157)挖出的真实 SCF **错误→恢复**重启链,
经语义提升 + `to_standard()` 重建为 26 标准 verifier-bound 动作的轨迹,门控:
faithful reload + per-action verifier + CHGNet 物理 → admissible。点击材料名进入该条的可视化页面
(结构图 + 原始 DAG + 重建轨迹)。

| 材料 | 空间群 | 轨迹 | 动作(首…末) | 结构 |
|---|---|---|---|---|
{rows}

> 由 [`examples/build_mc2d_gallery.py`](../../../examples/build_mc2d_gallery.py) 生成;
> 单条版见 [`examples/mc2d_standard_actions.py`](../../../examples/mc2d_standard_actions.py)。
> 诚实边界(§11):动作/观测是**真实记录的 QE 执行**,这里不重跑 QE;`reexecuted`=忠实重载并与归档逐一对上。
>
> 每条的 ① DAG 是蒸馏标注版;最忠实的**原生**溯源图可由 verdi 直接生成,例如 Sb8Te9:
> `verdi -p mc2d node graph generate 48712 --link-types data --ancestor-depth 20 --identifier pk --output-format png`
"""
    (GALLERY / "README.md").write_text(md)


def main():
    print(f"building mc2d standard-action gallery → {GALLERY}")
    llm = OpenRouterClient(max_tokens=400)
    print(f"teacher (LLM thought reconstruction): {llm.model_id}\n")
    metas = []
    for pk in CANDIDATES:
        try:
            m = build_episode(pk, llm)
        except Exception as e:  # noqa: BLE001
            print(f"  root {pk}: error {type(e).__name__}: {e}")
            m = None
        if m:
            metas.append(m)
        if len(metas) >= TARGET:
            break
    # prune any stale episode folder from a prior run that isn't in the final set
    import shutil
    keep = {m["slug"] for m in metas}
    for sub in GALLERY.iterdir():
        if sub.is_dir() and sub.name not in keep:
            shutil.rmtree(sub)
            print(f"  pruned stale folder {sub.name}/")

    _write_gallery(metas)
    print(f"\n{len(metas)} episodes → gallery index docs/images/mc2d/README.md  (LLM calls: {llm.n_calls})")
    for m in metas:
        print(f"  - {m['formula']:9s} {m['slug']}")


if __name__ == "__main__":
    main()

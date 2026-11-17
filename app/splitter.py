"""病例切分核心：z3 约束求解 + 不可再缩减冲突集（MUS）提取。

规则（均为硬约束，不使用近似比例）：
1. 同一患者的所有病例必须同组；
2. 患者亲属关系沿“同一患者 + 亲属对”传递，传递相连的病例必须同组；
3. 每个病例属于训练集（组 1）或评估集（组 2）之一；
4. 两组各标签（0/1）的病例数必须精确等于配额；
5. 被锁定的病例必须进入指定组。

无解时返回不可再缩减的冲突条件集合：其中每条约束都是导致无解的必要条件
（删除任意一条后其余所列约束即可满足），并标注冲突来源属于配额、锁定还是亲属关系。

亲属约束严格按 **原始亲属对** 建模：每条亲属 CSV 记录对应一条约束，约束里
引用的两个病例分别属于该记录里的两位患者（若一位患者有多例，则引用其代表
病例，并由同患者约束把该患者的全部病例连带同组）。这样冲突报告中不会出现
“亲属对写的是 B-C、病例却写成 C-A”这类由并查集内部根节点造成的错位解释。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import z3

GROUP_NAMES = {1: "训练集", 2: "评估集"}


@dataclass(frozen=True)
class Case:
    case_id: str
    patient_id: str
    label: int


@dataclass(frozen=True)
class RelativePair:
    patient_a: str
    patient_b: str


@dataclass(frozen=True)
class KinshipEdge:
    """一条原始亲属对，以及用于约束求解的两位患者代表病例。"""

    patient_a: str
    patient_b: str
    case_a: str
    case_b: str


@dataclass(frozen=True)
class ConstraintInfo:
    cid: str
    kind: str  # kinship | lock | quota
    description: str
    detail: Dict[str, object] = field(default_factory=dict)


class SplitterError(Exception):
    """输入数据本身的问题（在进入求解器之前校验）。"""


# --------------------------------------------------------------------------- #
# 患者分组与并查集：组件统计仍用并查集；冲突解释则只使用原始亲属对。
# --------------------------------------------------------------------------- #
def _cases_by_patient(cases: Sequence[Case]) -> Dict[str, List[str]]:
    by_patient: Dict[str, List[str]] = {}
    for c in sorted(cases, key=lambda item: item.case_id):
        by_patient.setdefault(c.patient_id, []).append(c.case_id)
    return by_patient


def build_components(
    cases: Sequence[Case], relative_pairs: Sequence[RelativePair]
) -> Tuple[Dict[str, List[str]], List[KinshipEdge]]:
    """返回（连通分量, 原始亲属对及其代表病例）。

    连通分量用于统计与预览展示；``KinshipEdge`` 用于生成可解释的亲属约束，
    始终保持与原始 CSV 中的患者对一一对应（关系成环或患者多病例时也一样）。
    """
    parent: Dict[str, str] = {c.case_id: c.case_id for c in cases}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # 固定方向，保证结果确定性
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra

    by_patient = _cases_by_patient(cases)
    # 同一患者的病例天然同组
    for _patient_id, ids in sorted(by_patient.items()):
        anchor = ids[0]
        for other in ids[1:]:
            union(anchor, other)

    representative = {patient_id: ids[0] for patient_id, ids in by_patient.items()}
    edges: List[KinshipEdge] = []
    for pair in sorted(relative_pairs, key=lambda p: (p.patient_a, p.patient_b)):
        ca = representative.get(pair.patient_a)
        cb = representative.get(pair.patient_b)
        if ca is None or cb is None:
            # 调用方应已校验；这里防御性跳过
            continue
        edges.append(
            KinshipEdge(
                patient_a=pair.patient_a,
                patient_b=pair.patient_b,
                case_a=ca,
                case_b=cb,
            )
        )
        union(ca, cb)

    components: Dict[str, List[str]] = {}
    for c in cases:
        components.setdefault(find(c.case_id), []).append(c.case_id)
    for ids in components.values():
        ids.sort()
    return components, edges


# --------------------------------------------------------------------------- #
# MUS 缩减：所有 check 都区分 sat / unsat / unknown；unknown 时停止缩减。
# --------------------------------------------------------------------------- #
def _is_unsat(
    solver: z3.Solver, handler_by_id: Dict[str, z3.BoolRef], cids: Sequence[str]
) -> Optional[bool]:
    """返回 True=unsat、False=sat、None=求解器无法判定。"""
    result = solver.check(*[handler_by_id[h] for h in cids])
    if result == z3.unknown:
        return None
    return result == z3.unsat


def _delete_chunk(
    solver: z3.Solver,
    handler_by_id: Dict[str, z3.BoolRef],
    kept: List[str],
    chunk: List[str],
) -> Tuple[List[str], bool]:
    """尝试从 kept 中整体删除 chunk，返回（保留项, 是否可继续核验）。

    - 删除后仍 unsat：整块约束都不是必要的，直接丢弃；
    - 删除后变 sat：块内至少含一条必要约束，二分递归定位，保留必要项；
    - 求解器 unknown：停止缩减，由上层标记为“未完成核验”。
    """
    chunk_set = set(chunk)
    without = [c for c in kept if c not in chunk_set]
    status = _is_unsat(solver, handler_by_id, without)
    if status is None:
        return kept, False
    if status:  # 删除整块后仍然 unsat，整块都不必要
        return without, True
    if len(chunk) == 1:  # 该条是必要约束
        return kept, True
    mid = len(chunk) // 2
    kept, ok = _delete_chunk(solver, handler_by_id, kept, chunk[:mid])
    if not ok:
        return kept, False
    return _delete_chunk(solver, handler_by_id, kept, chunk[mid:])


def _shrink_to_mus(
    solver: z3.Solver,
    handler_by_id: Dict[str, z3.BoolRef],
    initial_core: set,
) -> Tuple[set, bool, Optional[str]]:
    """从 z3 给出的 unsat core 出发，得到不可再缩减的子集（MUS）。

    返回（冲突集, 缩减是否完成, 未完成原因）。缩减过程只使用
    sat/unsat 结论；任何一次 check 返回 unknown 都不会被声称为已验证。
    """
    mus = sorted(initial_core)
    if not mus:
        return set(), False, "求解器没有给出可核验的初始冲突核。"

    if len(mus) <= 24:
        for cid in list(mus):
            status = _is_unsat(
                solver, handler_by_id, [c for c in mus if c != cid]
            )
            if status is None:
                return set(mus), False, f"核验约束 {cid} 时求解器返回 unknown。"
            if status:
                mus.remove(cid)
    else:
        chunk_size = 32
        i = 0
        while i < len(mus):
            chunk = mus[i:i + chunk_size]
            new_mus, ok = _delete_chunk(solver, handler_by_id, mus, chunk)
            if not ok:
                return set(new_mus), False, "分块缩减过程中求解器返回 unknown。"
            if len(new_mus) == len(mus):
                i += len(chunk)  # 整块都必要，前进
            mus = new_mus
            # 块被部分/整体删除后，i 不变，新的元素滑入当前窗口

    # 用同一组假设做一次终检；unknown 一律视为核验未完成
    status = _is_unsat(solver, handler_by_id, mus)
    if status is None:
        return set(mus), False, "缩减后的冲突集终检返回 unknown。"
    if not status:
        return set(mus), False, "缩减后的冲突集在终检时不再是不可满足。"
    return set(mus), True, None


def _verify_conflict_set(
    cond_by_id: Dict[str, z3.BoolRef], conflict_ids: Sequence[str]
) -> Tuple[bool, Optional[str]]:
    """独立核验冲突集：整体 unsat，且删除任意一条后其余条件 sat。

    使用全新的 z3 Solver，只加入冲突集中各条件对应的**原始语义**表达式
    （不是 handler 假设），因此不依赖求解过程中可能残留的约束或内部状态。
    任何 unknown / 非预期结果都返回 False，调用方不得宣称“已验证”。
    """
    if not conflict_ids:
        return False, "冲突集为空，无法核验。"
    try:
        whole = z3.Solver()
        whole.add(*[cond_by_id[cid] for cid in conflict_ids])
        result = whole.check()
        if result == z3.unknown:
            return False, "核验冲突集整体时求解器返回 unknown。"
        if result != z3.unsat:
            return False, "所列条件整体并非不可满足。"

        for cid in conflict_ids:
            others = [c for c in conflict_ids if c != cid]
            probe = z3.Solver()
            if others:
                probe.add(*[cond_by_id[other] for other in others])
            result = probe.check()
            if result == z3.unknown:
                return False, f"核验移除 {cid} 后是否可行时求解器返回 unknown。"
            if result != z3.sat:
                return False, f"移除 {cid} 后其余所列条件仍不可满足。"
    except z3.Z3Exception as exc:  # 求解器异常同样视为核验失败
        return False, f"核验冲突集时求解器异常：{exc}"
    return True, None


def solve_split(
    cases: Sequence[Case],
    relative_pairs: Sequence[RelativePair],
    quotas: Dict[Tuple[int, int], int],
    locks: Dict[str, int],
) -> dict:
    """执行切分求解。

    返回::

        {"status": "sat", "assignments": {case_id: 1|2}, "stats": {...},
         "constraint_count": n, "component_count": m}
        {"status": "unsat", "conflicts": [ConstraintInfo...],
         "mus_verified": True|False, ...}
    """
    cases = sorted(cases, key=lambda c: c.case_id)
    if not cases:
        raise SplitterError("没有可切分的病例，请先上传病例 CSV。")

    components, kinship_edges = build_components(cases, relative_pairs)
    by_patient = _cases_by_patient(cases)
    case_index = {c.case_id: c for c in cases}

    unknown_locks = sorted(set(locks) - set(case_index))
    if unknown_locks:
        raise SplitterError(f"锁定了不存在的病例：{', '.join(unknown_locks)}")
    for cid, grp in locks.items():
        if grp not in (1, 2):
            raise SplitterError(f"病例 {cid} 的锁定分组必须是 1 或 2。")
    for (grp, lab), qty in quotas.items():
        if grp not in (1, 2) or lab not in (0, 1):
            raise SplitterError("配额键必须是 (组 1/2, 标签 0/1)。")
        if not isinstance(qty, int) or qty < 0:
            raise SplitterError(f"{GROUP_NAMES[grp]}标签{lab}的配额必须是非负整数。")

    solver = z3.Solver()
    # 每个病例一个布尔变量：True=评估集(组2)，False=训练集(组1)
    b = {c.case_id: z3.Bool(f"in_group2_{i}_{c.case_id}") for i, c in enumerate(cases)}

    infos: Dict[str, ConstraintInfo] = {}
    handler_by_id: Dict[str, z3.BoolRef] = {}
    cond_by_id: Dict[str, z3.BoolRef] = {}
    counter = 0

    def track(cond: z3.BoolRef, info: ConstraintInfo) -> None:
        nonlocal counter
        handler = z3.Bool(f"handler_{counter}")
        counter += 1
        solver.add(z3.Implies(handler, cond))
        infos[info.cid] = info
        handler_by_id[info.cid] = handler
        cond_by_id[info.cid] = cond

    # 1) 同患者约束：每位患者只有一条条件，引用该患者的全部病例。
    #    这样“一名患者有多例”时，删除该条件即完整解除该患者的分组绑定，
    #    与原始规则一一对应。
    for patient_id, ids in sorted(by_patient.items()):
        if len(ids) < 2:
            continue
        anchor = ids[0]
        cond = z3.And([b[other] == b[anchor] for other in ids[1:]])
        track(
            cond,
            ConstraintInfo(
                cid=f"kin:{counter}",
                kind="kinship",
                description=(
                    f"同一患者 {patient_id} 的病例 {'、'.join(ids)} 必须分在同一组"
                ),
                detail={
                    "reason": "same_patient",
                    "patient_id": patient_id,
                    "case_ids": ids,
                },
            ),
        )

    # 2) 原始亲属对约束：每条 CSV 记录一条，病例严格属于记录中的患者。
    for edge in kinship_edges:
        cases_a = by_patient[edge.patient_a]
        cases_b = by_patient[edge.patient_b]
        track(
            b[edge.case_a] == b[edge.case_b],
            ConstraintInfo(
                cid=f"kin:{counter}",
                kind="kinship",
                description=(
                    f"原始亲属对 {edge.patient_a}–{edge.patient_b}："
                    f"患者 {edge.patient_a} 的病例 {edge.case_a} 与"
                    f"患者 {edge.patient_b} 的病例 {edge.case_b} 必须分在同一组"
                ),
                detail={
                    "reason": "relative",
                    "patient_a": edge.patient_a,
                    "patient_b": edge.patient_b,
                    "case_a": edge.case_a,
                    "case_b": edge.case_b,
                    "patient_a_cases": list(cases_a),
                    "patient_b_cases": list(cases_b),
                },
            ),
        )

    # 3) 锁定约束
    for cid in sorted(locks):
        grp = locks[cid]
        track(
            b[cid] == (grp == 2),
            ConstraintInfo(
                cid=f"lock:{cid}",
                kind="lock",
                description=f"病例 {cid} 被锁定到{GROUP_NAMES[grp]}",
                detail={"case_id": cid, "group": grp},
            ),
        )

    # 4) 精确配额约束（两组 × 两标签，共四条）
    for grp in (1, 2):
        for lab in (0, 1):
            qty = quotas[(grp, lab)]
            same_label = [c for c in cases if c.label == lab]
            terms = [z3.If(b[c.case_id] == (grp == 2), 1, 0) for c in same_label]
            info = ConstraintInfo(
                cid=f"quota:g{grp}:l{lab}",
                kind="quota",
                description=(
                    f"{GROUP_NAMES[grp]}中标签 {lab} 的病例数量必须恰好为 {qty}"
                ),
                detail={"group": grp, "label": lab, "target": qty},
            )
            if terms:
                track(z3.Sum(terms) == qty, info)
            elif qty > 0:
                # 该标签根本没有病例，却要求正数配额：恒假约束，同样可被 MUS 追踪
                track(z3.BoolVal(False), info)

    assumptions = [handler_by_id[cid] for cid in sorted(infos)]
    result = solver.check(*assumptions)

    if result == z3.unknown:
        raise SplitterError(f"求解器无法判定（{solver.reason_unknown()}）。")

    if result == z3.sat:
        model = solver.model()
        assignments: Dict[str, int] = {}
        for c in cases:
            assignments[c.case_id] = (
                2 if z3.is_true(model.eval(b[c.case_id], model_completion=True)) else 1
            )
        # 独立复核配额与连通性，防止模型解读错误
        counts = {(g, l): 0 for g in (1, 2) for l in (0, 1)}
        for c in cases:
            counts[(assignments[c.case_id], c.label)] += 1
        if counts != dict(quotas):
            raise SplitterError(f"内部错误：模型配额不符 {counts} != {quotas}")
        if any(
            assignments[e.case_a] != assignments[e.case_b] for e in kinship_edges
        ):
            raise SplitterError("内部错误：模型违反亲属约束")
        for patient_id, ids in by_patient.items():
            if len({assignments[cid] for cid in ids}) != 1:
                raise SplitterError(f"内部错误：模型违反同患者约束（患者 {patient_id}）")
        for cid, grp in locks.items():
            if assignments[cid] != grp:
                raise SplitterError(f"内部错误：模型违反锁定 {cid}")
        return {
            "status": "sat",
            "assignments": assignments,
            "stats": {
                "case_count": len(cases),
                "patient_count": len({c.patient_id for c in cases}),
                "component_count": len(components),
                "counts": {f"g{g}_label{l}": counts[(g, l)] for g in (1, 2) for l in (0, 1)},
            },
            "constraint_count": len(infos),
        }

    # unsat：提取不可再缩减冲突集，并用独立求解器复核
    core_handler_objs = set(solver.unsat_core())
    initial_core = {
        cid for cid, h in handler_by_id.items() if h in core_handler_objs
    }
    if not initial_core:  # 理论上不会发生
        initial_core = set(infos)
    mus, shrink_complete, shrink_error = _shrink_to_mus(
        solver, handler_by_id, initial_core
    )

    verified, verify_error = (False, "缩减未完成，未进行独立终检。")
    if shrink_complete:
        verified, verify_error = _verify_conflict_set(cond_by_id, sorted(mus))
    mus_verified = bool(verified)
    verification_error = verify_error or shrink_error

    conflicts = [_info_to_json(infos[cid]) for cid in sorted(mus)]
    kinds = sorted({infos[cid].kind for cid in mus})
    kind_text = "、".join(_kind_name(k) for k in kinds)
    if mus_verified:
        summary = (
            f"无解：以下 {len(conflicts)} 条约束互相冲突"
            f"（已验证不可再缩减，删除其中任意一条后其余所列条件即可满足）。"
            f"冲突来源类型：{kind_text}。"
        )
    else:
        summary = (
            f"无解：以下 {len(conflicts)} 条为候选冲突条件，"
            f"但未能完成不可再缩减核验"
            f"（{verification_error}）。请勿将其视为已验证的冲突集。"
            f"冲突来源类型：{kind_text}。"
        )
    return {
        "status": "unsat",
        "conflicts": conflicts,
        "mus_verified": mus_verified,
        "verification_error": None if mus_verified else verification_error,
        "stats": {
            "case_count": len(cases),
            "patient_count": len({c.patient_id for c in cases}),
            "component_count": len(components),
        },
        "constraint_count": len(infos),
        "conflict_kinds": kinds,
        "summary": summary,
    }


def _kind_name(kind: str) -> str:
    return {"quota": "配额", "lock": "锁定", "kinship": "亲属/同患者关系"}.get(kind, kind)


def _info_to_json(info: ConstraintInfo) -> dict:
    return {
        "cid": info.cid,
        "kind": info.kind,
        "kind_name": _kind_name(info.kind),
        "description": info.description,
        "detail": info.detail,
    }

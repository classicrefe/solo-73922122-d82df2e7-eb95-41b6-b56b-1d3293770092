"""冲突解释回归测试：亲属条件必须对应原始患者对，且所列条件整体不可再缩减。

覆盖：
- 用户报告的场景：四名患者各一例标签 1，A–B、B–C 相连，A/C 锁定不同组，
  两组各需两例；冲突解释须引用原始患者对 A–B 与 B–C（不得出现错位病例）。
- 关系成环（A–B–C–A）。
- 一名患者有多例（亲属对引用该患者的代表病例，且同患者条件独立成条）。
- 独立复核：把冲突集的每条条件按 detail 还原为约束，整体 unsat，
  删除任意一条后其余条件 sat。
- 核验无法完成时 mus_verified=False，且不产生预览、不改动源数据与清单。
- 小规模/大规模冲突集（>24 条，走分块缩减路径）都能完成核验。
"""
import os
import sys

import pytest
import z3
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.splitter import Case, RelativePair, solve_split  # noqa: E402


def C(cid, pid, label):
    return Case(case_id=cid, patient_id=pid, label=label)


def _independent_check(cases, conflicts):
    """按冲突条目的 detail 重建条件并独立核验对象是否不可再缩减。"""
    b = {c.case_id: z3.Bool(f"chk_{c.case_id}") for c in cases}
    label_by_case = {c.case_id: c.label for c in cases}
    patient_by_case = {c.case_id: c.patient_id for c in cases}

    def condition(conflict):
        d = conflict["detail"]
        if conflict["kind"] == "kinship" and d["reason"] == "relative":
            return b[d["case_a"]] == b[d["case_b"]]
        if conflict["kind"] == "kinship" and d["reason"] == "same_patient":
            ids = d["case_ids"]
            return z3.And([b[x] == b[ids[0]] for x in ids[1:]])
        if conflict["kind"] == "lock":
            return b[d["case_id"]] == (d["group"] == 2)
        if conflict["kind"] == "quota":
            terms = [
                z3.If(b[cid] == (d["group"] == 2), 1, 0)
                for cid, lab in label_by_case.items()
                if lab == d["label"]
            ]
            return z3.Sum(terms) == d["target"] if terms else z3.BoolVal(False)
        raise AssertionError(f"未知冲突类型：{conflict}")

    conds = [condition(c) for c in conflicts]

    def is_unsat(selected):
        solver = z3.Solver()
        solver.add(*selected)
        return solver.check() == z3.unsat

    assert conds and is_unsat(conds)
    for i in range(len(conds)):
        assert not is_unsat([c for j, c in enumerate(conds) if j != i]), (
            f"删除第 {i + 1} 条后其余冲突条件仍不可满足：{conflicts[i]}"
        )
    # detail 里的病例必须确实属于对应患者
    for conflict in conflicts:
        d = conflict["detail"]
        if conflict["kind"] == "kinship" and d["reason"] == "relative":
            assert patient_by_case[d["case_a"]] == d["patient_a"]
            assert patient_by_case[d["case_b"]] == d["patient_b"]
    return True


DEMO_CASES = [C("A", "A", 1), C("B", "B", 1), C("C", "C", 1), C("D", "D", 1)]
DEMO_RELS = [RelativePair("A", "B"), RelativePair("B", "C")]
DEMO_QUOTAS = {(1, 0): 0, (1, 1): 2, (2, 0): 0, (2, 1): 2}
DEMO_LOCKS = {"A": 1, "C": 2}


def test_demo_conflict_reports_original_patient_pairs():
    r = solve_split(DEMO_CASES, DEMO_RELS, DEMO_QUOTAS, DEMO_LOCKS)
    assert r["status"] == "unsat"
    assert r["mus_verified"] is True

    kinships = [c for c in r["conflicts"] if c["kind"] == "kinship"]
    pairs = {(c["detail"]["patient_a"], c["detail"]["patient_b"]) for c in kinships}
    assert pairs == {("A", "B"), ("B", "C")}
    # 不得出现 A–C 之类原始 CSV 中不存在的患者对
    descriptions = " ".join(c["description"] for c in r["conflicts"])
    assert "A–B" in descriptions and "B–C" in descriptions
    _independent_check(DEMO_CASES, r["conflicts"])

    # 用户报告的关键事实：仅保留 B–C 后（去掉 A–B）切分可行
    fixed = solve_split(DEMO_CASES, [RelativePair("B", "C")], DEMO_QUOTAS, DEMO_LOCKS)
    assert fixed["status"] == "sat"
    assert fixed["assignments"]["A"] == 1
    assert fixed["assignments"]["C"] == 2
    assert fixed["assignments"]["B"] == fixed["assignments"]["C"]


def test_cycle_reports_each_original_edge():
    cases = [C("A", "A", 1), C("B", "B", 1), C("C", "C", 1)]
    rels = [
        RelativePair("A", "B"),
        RelativePair("B", "C"),
        RelativePair("C", "A"),
    ]
    quotas = {(1, 0): 0, (1, 1): 2, (2, 0): 0, (2, 1): 1}
    r = solve_split(cases, rels, quotas, {"B": 2})
    assert r["status"] == "unsat"
    assert r["mus_verified"] is True
    for conflict in r["conflicts"]:
        if conflict["kind"] == "kinship":
            pair = (conflict["detail"]["patient_a"], conflict["detail"]["patient_b"])
            assert pair in {("A", "B"), ("B", "C"), ("C", "A")}
    _independent_check(cases, r["conflicts"])


def test_multiple_cases_per_patient_reference_representative():
    # 患者 P1 有两例 X、Y；P1–P2 亲属；锁定 Y→组1、Z→组2
    cases = [C("X", "P1", 1), C("Y", "P1", 1), C("Z", "P2", 1)]
    rels = [RelativePair("P1", "P2")]
    quotas = {(1, 0): 0, (1, 1): 2, (2, 0): 0, (2, 1): 1}
    r = solve_split(cases, rels, quotas, {"Y": 1, "Z": 2})
    assert r["status"] == "unsat"
    assert r["mus_verified"] is True

    kinships = [c for c in r["conflicts"] if c["kind"] == "kinship"]
    relative = [c for c in kinships if c["detail"]["reason"] == "relative"]
    same_patient = [c for c in kinships if c["detail"]["reason"] == "same_patient"]
    assert len(relative) == 1 and len(same_patient) == 1
    rd = relative[0]["detail"]
    assert (rd["patient_a"], rd["patient_b"]) == ("P1", "P2")
    assert rd["case_a"] in ("X", "Y") and rd["case_b"] == "Z"
    assert rd["case_a"] == "X"  # 代表病例按病例号排序取首个
    assert set(same_patient[0]["detail"]["case_ids"]) == {"X", "Y"}
    _independent_check(cases, r["conflicts"])


def test_large_conflict_set_chunked_shrink_is_verified():
    # 30 个病例均锁定组1，组1配额 29、组2配额 1 -> 冲突集 >24 条，走分块缩减
    cases = [C(f"c{i:02d}", f"p{i:02d}", 1) for i in range(30)]
    quotas = {(1, 0): 0, (1, 1): 29, (2, 0): 0, (2, 1): 1}
    locks = {c.case_id: 1 for c in cases}
    r = solve_split(cases, [], quotas, locks)
    assert r["status"] == "unsat"
    assert r["mus_verified"] is True
    assert len(r["conflicts"]) > 24
    _independent_check(cases, r["conflicts"])


# --------------------------------------------------------------------------- #
# API：演示链路 + 核验失败语义
# --------------------------------------------------------------------------- #
@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    import importlib
    import app.main as main

    monkeypatch.setenv("SPLIT_DB_PATH", str(tmp_path / "test.db"))
    importlib.reload(main)
    yield main
    main.db.close()


@pytest.fixture()
def client(main_module):
    with TestClient(main_module.app) as c:
        yield c


def _upload(client, path, csv_text, name):
    r = client.post(
        path, files={"file": (name, csv_text.encode("utf-8"), "text/csv")}
    )
    assert r.status_code == 200, r.text
    return r.json()


def _load_demo(client):
    demo = client.get("/api/state").json()["demo"]
    _upload(client, "/api/upload/cases", demo["cases_csv"], "demo_cases.csv")
    _upload(client, "/api/upload/relatives", demo["relatives_csv"], "demo_relatives.csv")
    assert client.put("/api/quotas", json=demo["quotas"]).status_code == 200
    assert client.put("/api/locks", json={"locks": demo["locks"]}).status_code == 200
    return demo


def test_api_demo_conflict_and_corrected_rerun(client):
    demo = _load_demo(client)
    body = client.post("/api/solve").json()
    result = body["result"]
    assert result["status"] == "unsat"
    assert body["preview_token"] is None
    assert result["mus_verified"] is True
    pairs = {
        (c["detail"]["patient_a"], c["detail"]["patient_b"])
        for c in result["conflicts"]
        if c["kind"] == "kinship" and c["detail"].get("reason") == "relative"
    }
    assert pairs == {("A", "B"), ("B", "C")}
    # 冲突集只列配额/锁定/亲属三类来源，且不含任何虚构的 A–C 亲属对
    assert {c["kind"] for c in result["conflicts"]} <= {"quota", "lock", "kinship"}
    assert "A–C" not in " ".join(c["description"] for c in result["conflicts"])

    # 无解不产生清单，也不改动输入版本
    version_after = client.get("/api/state").json()["dataset_version"]
    assert client.get("/api/manifests").json()["manifests"] == []
    assert version_after == body["dataset_version"]

    # 更正：仅保留 B–C 后重新求解可行并可发布
    _upload(client, "/api/upload/relatives", demo["relatives_after_fix_csv"],
            "demo_relatives_fixed.csv")
    solved = client.post("/api/solve").json()
    assert solved["result"]["status"] == "sat"
    assert solved["result"]["assignments"] == demo["fix_result"]
    pub = client.post("/api/publish", json={"preview_token": solved["preview_token"]})
    assert pub.status_code == 200, pub.text
    manifest_id = pub.json()["manifest_id"]
    assert client.get(f"/api/manifests/{manifest_id}/download").status_code == 200

    # 再次更正输入（即使内容相同）= 新版本：旧清单永久失效，不能下载
    _upload(client, "/api/upload/relatives", demo["relatives_after_fix_csv"],
            "demo_relatives_fixed_again.csv")
    detail = client.get(f"/api/manifests/{manifest_id}").json()
    assert detail["expired"] is True and detail["invalidated"] is True
    assert client.get(f"/api/manifests/{manifest_id}/download").status_code == 409


def test_api_verification_failure_is_not_claimed_verified(
    client, main_module, monkeypatch
):
    _load_demo(client)
    before = client.get("/api/state").json()

    def _fail(_cond_by_id, _ids):
        return False, "模拟：核验时求解器返回 unknown。"

    monkeypatch.setattr(main_module, "solve_split", main_module.solve_split)
    import app.splitter as splitter

    monkeypatch.setattr(splitter, "_verify_conflict_set", _fail)
    body = client.post("/api/solve").json()
    result = body["result"]
    assert result["status"] == "unsat"
    assert result["mus_verified"] is False
    assert "未能完成不可再缩减核验" in result["summary"]
    assert "已验证不可再缩减" not in result["summary"]
    assert "unknown" in result["verification_error"]
    assert body["preview_token"] is None

    after = client.get("/api/state").json()
    # 源数据与清单均未被改动
    assert after["dataset_version"] == before["dataset_version"]
    assert after["input"] == before["input"]
    assert client.get("/api/manifests").json()["manifests"] == []

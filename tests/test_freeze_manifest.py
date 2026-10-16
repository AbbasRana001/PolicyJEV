import hashlib
import shutil
import sys

import pandas as pd
import pytest

from policyjev import freeze_manifest as fm
from policyjev.baselines import predict_test as pt
from policyjev.config import LABELS
from policyjev.evaluation import splits
from policyjev.jev import second_pass


def make_repo(root, with_distilbert: bool = True):
    """A tiny repository layout containing every file the manifest should cover."""
    files = {
        "configs/questions_v1.yaml": "version: v1\n",
        "configs/questions_v2.yaml": "version: v2\n",
        "artifacts/calibrators/logreg_word.json": "{}\n",
        "artifacts/calibrators/distilbert.json": "{}\n",
        "artifacts/calibrators/jev_v2.json": "{}\n",
        "artifacts/calibrators/jev_stack.json": "{}\n",
        "artifacts/metrics/validation_thresholds.csv": "system,label,threshold,f1_in_sample\n",
        "artifacts/models/logreg_word.joblib": "model bytes",
        "artifacts/policy/policy_thresholds.csv": "system,label\n",
    }
    if with_distilbert:
        files["artifacts/models/distilbert/model.safetensors"] = "weights"
        files["artifacts/models/distilbert/config.json"] = "{}"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return files


def test_sha256_matches_the_reference_digest(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert fm.sha256_of(path) == hashlib.sha256(b"abc").hexdigest()


def test_write_manifest_covers_every_defining_file_and_is_sorted(tmp_path):
    files = make_repo(tmp_path)

    count = fm.write_manifest(tmp_path, "v2")
    manifest = fm.read_manifest(fm.manifest_file(tmp_path))

    assert count == len(files) == len(manifest)
    assert list(manifest) == sorted(manifest)
    key = "artifacts/models/logreg_word.joblib"
    assert manifest[key] == (hashlib.sha256(b"model bytes").hexdigest(), len(b"model bytes"))
    assert "\r" not in fm.manifest_file(tmp_path).read_text(encoding="utf-8")


def test_write_manifest_refuses_missing_files_unless_allowed(tmp_path):
    make_repo(tmp_path, with_distilbert=False)

    with pytest.raises(fm.ManifestError) as info:
        fm.write_manifest(tmp_path, "v2")
    assert "distilbert" in str(info.value)

    count = fm.write_manifest(tmp_path, "v2", allow_missing=True)
    assert count == 9


def test_verify_files_accepts_matching_files_and_checks_only_the_entries_it_uses(tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    (tmp_path / "artifacts/models/logreg_word.joblib").write_text("tampered", encoding="utf-8")

    fm.verify_files([tmp_path / "configs/questions_v2.yaml"], root=tmp_path)
    fm.verify_files(
        [tmp_path / "artifacts/calibrators/jev_v2.json", tmp_path / "configs/questions_v1.yaml"], root=tmp_path
    )

    with pytest.raises(fm.ManifestError) as info:
        fm.verify_files([tmp_path / "artifacts/models/logreg_word.joblib"], root=tmp_path)
    assert "logreg_word.joblib" in str(info.value)
    assert "does not match" in str(info.value)


def test_verify_files_reports_unlisted_and_missing_files(tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    extra = tmp_path / "configs" / "questions_v9.yaml"
    extra.write_text("version: v9\n", encoding="utf-8")
    (tmp_path / "configs/questions_v1.yaml").unlink()

    with pytest.raises(fm.ManifestError) as info:
        fm.verify_files([extra, tmp_path / "configs/questions_v1.yaml"], root=tmp_path)
    message = str(info.value)
    assert "questions_v9.yaml: not listed" in message
    assert "questions_v1.yaml: file not found" in message


def test_missing_manifest_is_a_clear_error(tmp_path):
    make_repo(tmp_path)
    with pytest.raises(fm.ManifestError) as info:
        fm.verify_files([tmp_path / "configs/questions_v1.yaml"], root=tmp_path)
    assert "frozen_manifest.csv not found" in str(info.value)


def test_model_folder_verification_works_from_another_location(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    fm.write_manifest(repo, "v2")
    colab = tmp_path / "colab_copy"
    shutil.copytree(repo / "artifacts/models/distilbert", colab)

    fm.verify_model_folder("artifacts/models/distilbert", colab, fm.manifest_file(repo))

    (colab / "model.safetensors").write_text("other weights", encoding="utf-8")
    with pytest.raises(fm.ManifestError):
        fm.verify_model_folder("artifacts/models/distilbert", colab, fm.manifest_file(repo))

    with pytest.raises(fm.ManifestError):
        fm.verify_model_folder("artifacts/models/unknown", colab, fm.manifest_file(repo))


def test_check_all_detects_changes(tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    assert fm.check_all(tmp_path) == 11

    (tmp_path / "configs/questions_v2.yaml").write_text("version: changed\n", encoding="utf-8")
    with pytest.raises(fm.ManifestError):
        fm.check_all(tmp_path)


def test_predict_test_refuses_on_a_hash_mismatch_before_predicting(monkeypatch, tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    (tmp_path / "artifacts/models/logreg_word.joblib").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(fm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(splits, "tag_exists", lambda tag: True)
    frame = pd.DataFrame({"id": ["t1"], "comment_text": ["a"], **{label: 0 for label in LABELS}})
    monkeypatch.setattr(pt, "load_labeled_frame", lambda split: frame)
    monkeypatch.setattr(
        pt, "predict_logreg", lambda *a, **k: (_ for _ in ()).throw(AssertionError("predicted"))
    )
    monkeypatch.setattr(pt, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["predict_test", "--system", "logreg", "--model-dir", str(tmp_path / "artifacts/models")])

    with pytest.raises(fm.ManifestError):
        pt.main()


def test_predict_test_distilbert_verifies_a_relocated_model_folder(monkeypatch, tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    colab = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "artifacts/models/distilbert", colab)
    (colab / "model.safetensors").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(fm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(splits, "tag_exists", lambda tag: True)
    frame = pd.DataFrame({"id": ["t1"], "comment_text": ["a"], **{label: 0 for label in LABELS}})
    monkeypatch.setattr(pt, "load_labeled_frame", lambda split: frame)
    monkeypatch.setattr(sys, "argv", ["predict_test", "--system", "distilbert", "--model-dir", str(colab)])

    with pytest.raises(fm.ManifestError):
        pt.main()


def test_second_pass_refuses_a_changed_question_file(monkeypatch, tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    questions = tmp_path / "configs/questions_v2.yaml"
    questions.write_text("version: v2\nquestions:\n  toxic:\n    type: noul\n    instructions: changed\n", encoding="utf-8")
    monkeypatch.setattr(fm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(second_pass, "verify_files", fm.verify_files)
    tags = {"questions-alt-frozen", "protocol-frozen"}
    from policyjev.jev import runner

    monkeypatch.setattr(runner, "tag_exists", lambda tag: tag in tags)
    monkeypatch.setattr(splits, "tag_exists", lambda tag: tag in tags)
    monkeypatch.setattr(
        second_pass, "load_labeled_frame", lambda split: (_ for _ in ()).throw(AssertionError("read"))
    )
    monkeypatch.setattr(sys, "argv", ["second_pass", "--questions", str(questions)])

    with pytest.raises(fm.ManifestError):
        second_pass.main()


def test_test_bundle_refuses_a_changed_calibrator(monkeypatch, tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    (tmp_path / "artifacts/calibrators/jev_v2.json").write_text('{"changed": true}', encoding="utf-8")
    monkeypatch.setattr(fm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(splits, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(splits, "tag_exists", lambda tag: True)
    monkeypatch.setattr(
        splits, "calibrator_path", lambda system, version: tmp_path / "artifacts/calibrators" / f"{'jev_v2' if system == 'jev' else system}.json"
    )
    monkeypatch.setattr(
        splits, "load_labeled_frame", lambda split: (_ for _ in ()).throw(AssertionError("read"))
    )

    with pytest.raises(fm.ManifestError) as info:
        splits.load_split_scores_bundle("test", ["jev"])
    assert "jev_v2.json" in str(info.value)


@pytest.mark.parametrize("name", ["a.csv", "b.json", "c.yaml", "d.yml", "e.md", "f.txt", "G.CSV"])
def test_text_files_hash_the_same_with_crlf_and_lf(tmp_path, name):
    lf = tmp_path / "lf" / name
    crlf = tmp_path / "crlf" / name
    lf.parent.mkdir()
    crlf.parent.mkdir()
    lf.write_bytes(b"first line\nsecond line\n")
    crlf.write_bytes(b"first line\r\nsecond line\r\n")

    assert fm.sha256_of(lf) == fm.sha256_of(crlf)
    assert fm.digest_and_size(lf) == fm.digest_and_size(crlf)
    assert fm.digest_and_size(crlf)[1] == len(b"first line\nsecond line\n")


@pytest.mark.parametrize("name", ["model.safetensors", "logreg_word.joblib", "weights.bin"])
def test_binary_files_are_hashed_unchanged(tmp_path, name):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.write_bytes(b"\x00\x01line\nend")
    crlf.write_bytes(b"\x00\x01line\r\nend")
    lf_file = tmp_path / ("lf_" + name)
    crlf_file = tmp_path / ("crlf_" + name)
    lf_file.write_bytes(lf.read_bytes())
    crlf_file.write_bytes(crlf.read_bytes())

    assert fm.sha256_of(lf_file) == hashlib.sha256(b"\x00\x01line\nend").hexdigest()
    assert fm.sha256_of(crlf_file) == hashlib.sha256(b"\x00\x01line\r\nend").hexdigest()
    assert fm.sha256_of(lf_file) != fm.sha256_of(crlf_file)
    assert fm.digest_and_size(crlf_file)[1] == len(b"\x00\x01line\r\nend")


def test_manifest_verifies_a_text_file_after_a_line_ending_change(tmp_path):
    make_repo(tmp_path)
    fm.write_manifest(tmp_path, "v2")
    questions = tmp_path / "configs/questions_v2.yaml"
    lf_content = questions.read_bytes().replace(b"\r\n", b"\n")
    questions.write_bytes(lf_content.replace(b"\n", b"\r\n"))
    assert b"\r\n" in questions.read_bytes()

    fm.verify_files([questions], root=tmp_path)
    assert fm.check_all(tmp_path) == 11

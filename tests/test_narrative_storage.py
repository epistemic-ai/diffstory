"""Synthetic storage closure, identity, durability and size checks."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from test_analysis import snap
from test_narrative_plan import AFTER
from test_narrative_plan import BEFORE
from test_narrative_plan import authored_result
from test_narrative_runtime import changed_record
from test_narrative_runtime import digest
from test_narrative_runtime import fixture_v2
from test_narrative_runtime import identity
from test_narrative_runtime import invocation_for_results
from test_narrative_runtime import prepare_v2
from test_narrative_runtime import prepared_v2
from test_narrative_runtime import validate_v2
from test_narrative_runtime import wire_v2

from diffstory import narrative_runtime as runtime
from diffstory.analysis import compile_snapshot
from diffstory.narrative_contracts import ArtifactRecord
from diffstory.narrative_contracts import Contract
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import validate_detail_result
from diffstory.narrative_storage import StoredInvocation
from diffstory.narrative_storage import decode_invocation
from diffstory.narrative_storage import encode_invocation
from diffstory.narrative_storage import storage_digest


def artifacts_v2(count: int = 2) -> tuple:
    """Build expanded artifacts with the exact original successful metadata."""
    invocation, registry, results = fixture_v2(count)
    artifacts = tuple(
        runtime.StoredArtifact(
            original_run_digest=identity().digest,
            invocation=invocation,
            lookup=lookup,
            record=ArtifactRecord(
                semantic_key=invocation.artifact_key(lookup),
                dependencies=invocation.dependencies,
                original_request_digest=invocation.batch.request_digest,
                validator_version=lookup.validator_version,
                result=results[lookup.item_key],
            ),
            production_request=prepared_v2(
                invocation.batch.item_keys
            ).metadata,
        )
        for lookup in invocation.lookups
    )
    return artifacts, registry


def measure_authored_batch(directory: Path, count: int = 64) -> dict:
    """Measure all actual cache bytes for a source-validated authored batch."""
    ledger = build_evidence_ledger(
        compile_snapshot(
            snap(
                {f"unit_{i}.py": BEFORE for i in range(count)},
                {f"unit_{i}.py": AFTER for i in range(count)},
            )
        )
    )
    units = {u.id: u for u in ledger.units if u.change_key}
    results = tuple(authored_result(unit) for unit in units.values())
    invocation = invocation_for_results(results)
    registry = runtime.DependencyRegistry(dependencies=invocation.dependencies)
    index = build_ledger_index(ledger)

    def validate(_lookup: object, result: object) -> None:
        validate_detail_result(
            result, units[result.unit_key], ledger, index=index
        )

    send = Mock(
        return_value=wire_v2([r.model_dump(mode="json") for r in results])
    )
    callbacks = runtime.BatchCallbacks(prepare_v2, send, validate)
    cold = runtime.ArtifactRuntime(
        identity(), recovery_dir=directory, artifact_dir=directory
    )
    outcome = cold.execute_batch(invocation, registry, callbacks)
    warm = runtime.ArtifactRuntime(
        identity(), recovery_dir=directory, artifact_dir=directory
    )
    reused = warm.execute_batch(invocation, registry, callbacks)
    expanded = directory / "expanded-comparison"
    expanded.mkdir()
    for artifact in outcome.artifacts:
        (expanded / f"{artifact.lookup.key}.json").write_bytes(
            artifact.model_dump_json().encode("utf-8")
        )
    categories = {
        "invocations": list(
            (directory / "artifacts-v3/invocations").glob("*.json")
        ),
        "requests": list((directory / "artifacts-v3/requests").glob("*.json")),
        "leaves": list((directory / "artifacts-v3/items").glob("*/*.json")),
        "expanded_v2": list(expanded.glob("*.json")),
    }
    sizes = {
        name: {
            "count": len(paths),
            "bytes": sum(p.stat().st_size for p in paths),
        }
        for name, paths in categories.items()
    }
    total = sum(
        sizes[name]["bytes"] for name in ("invocations", "requests", "leaves")
    )
    return {
        "batch_items": len(results),
        "cold_calls": cold.usage.report()["calls"],
        "warm_calls": warm.usage.report()["calls"],
        "total_sends": send.call_count,
        "expanded_equality": outcome.artifacts == reused.artifacts,
        "semantic_keys_equal": tuple(
            a.record.semantic_key for a in outcome.artifacts
        )
        == tuple(a.record.semantic_key for a in reused.artifacts),
        "files": sizes,
        "total_cache_bytes": total,
        "all_cache_bytes": sum(
            p.stat().st_size for p in (directory / "artifacts-v3").rglob("*.*")
        ),
        "expanded_to_factored_ratio": sizes["expanded_v2"]["bytes"] / total,
    }


class InvocationCodecTests(unittest.TestCase):
    """Retain all exact fields while sharing only proven duplicate fields."""

    def test_lossless_roundtrip_and_strict_runtime_validation(self) -> None:
        """Decode through complete peer, stage, lookup and version checks."""
        original, _, _ = fixture_v2(32)
        encoded = encode_invocation(original)
        raw = encoded.model_dump_json()
        decoded = decode_invocation(
            StoredInvocation.model_validate_json(raw),
            runtime.ArtifactInvocation,
        )
        self.assertEqual(original, decoded)
        self.assertEqual(original.semantic_key, decoded.semantic_key)
        self.assertEqual(original.artifact_keys(), decoded.artifact_keys())
        self.assertEqual(raw.count('"static_dependencies"'), 1)
        self.assertNotIn(
            '"static_dependencies"', json.dumps(json.loads(raw)["lookups"])
        )
        for mutate in (
            lambda d: d.update(version="diffstory.invocation-storage.v0"),
            lambda d: d.update(extra="unknown"),
            lambda d: d["batch"].update(partition=True),
            lambda d: d["lookups"].reverse(),
            lambda d: d["lookups"][0].update(
                validator_version="unsafe version"
            ),
            lambda d: d["lookups"][0].update(result_kind="section"),
            lambda d: d["dynamic_dependencies"].pop(),
            lambda d: d["static_dependencies"].pop(),
            lambda d: d["dynamic_dependencies"].append(
                d["static_dependencies"][0]
            ),
        ):
            value = json.loads(raw)
            mutate(value)
            with (
                self.subTest(value=value.keys()),
                self.assertRaises((ValueError, runtime.ArtifactRuntimeError)),
            ):
                decode_invocation(
                    StoredInvocation.model_validate_json(json.dumps(value)),
                    runtime.ArtifactInvocation,
                )
        forged = original.model_copy(update={"dependencies": ()})
        with self.assertRaises((ValueError, runtime.ArtifactRuntimeError)):
            encode_invocation(forged)

    def test_full_content_address_distinguishes_same_semantic_key(
        self,
    ) -> None:
        """Every exact BatchSpec field remains present even outside semantic identity."""
        original, _, _ = fixture_v2(2)
        baseline = storage_digest(encode_invocation(original))
        changes = {
            "id": digest("other-batch"),
            "request_digest": digest("other-request"),
            "input_bound": original.batch.input_bound + 1,
            "output_reservation": original.batch.output_reservation - 1,
            "output_guard": original.batch.output_guard + 1,
            "estimated_response_bytes": original.batch.estimated_response_bytes
            + 1,
            "estimated_seconds": original.batch.estimated_seconds + 1,
            "partition": original.batch.partition + 1,
        }
        for name, value in changes.items():
            with self.subTest(field=name):
                other = changed_record(
                    original,
                    batch=changed_record(original.batch, **{name: value}),
                )
                encoded = encode_invocation(other)
                self.assertEqual(original.semantic_key, other.semantic_key)
                self.assertNotEqual(baseline, storage_digest(encoded))
                self.assertEqual(
                    other,
                    decode_invocation(encoded, runtime.ArtifactInvocation),
                )


class ArtifactStorageTests(unittest.TestCase):
    """Check shared reference closure without relying on trusted read caches."""

    def setUp(self) -> None:
        """Create only synthetic artifacts in a private temporary directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = runtime.ArtifactStore(self.root)
        self.artifacts, self.registry = artifacts_v2()
        self.artifact = self.artifacts[0]

    def commit(self) -> tuple[Path, Path, Path]:
        """Commit both peers and return the first leaf and its shared records."""
        for artifact in self.artifacts:
            self.store.commit(artifact)
        leaf = next(
            (self.store.directory / self.artifact.lookup.key).glob("*.json")
        )
        invocation = next(
            (self.store.shared_directory / "invocations").glob("*.json")
        )
        request = next(
            (self.store.shared_directory / "requests").glob("*.json")
        )
        return leaf, invocation, request

    def lookup(self, **kwargs: object) -> runtime.StoredArtifact | None:
        """Use the production cache validation and audit boundary."""
        return self.store.lookup(
            self.artifact.lookup, self.registry, validate_v2, **kwargs
        )

    def test_authored_64_item_batch_measures_every_cache_file(self) -> None:
        """Store one exact producer pair, preserve source-validated results, reuse without sends."""
        metrics = measure_authored_batch(self.root)
        self.assertEqual(metrics["batch_items"], 64)
        self.assertEqual(metrics["cold_calls"], 1)
        self.assertEqual(metrics["warm_calls"], 0)
        self.assertEqual(metrics["total_sends"], 1)
        self.assertTrue(metrics["expanded_equality"])
        self.assertTrue(metrics["semantic_keys_equal"])
        self.assertEqual(metrics["files"]["invocations"]["count"], 1)
        self.assertEqual(metrics["files"]["requests"]["count"], 1)
        self.assertEqual(metrics["files"]["leaves"]["count"], 64)
        self.assertEqual(
            metrics["total_cache_bytes"], metrics["all_cache_bytes"]
        )
        self.assertGreater(metrics["expanded_to_factored_ratio"], 5)

    def test_legacy_read_import_and_closed_new_import(self) -> None:
        """Read old files without rewriting; new imports need local reference closure."""
        raw = self.artifact.model_dump_json().encode()
        legacy = (
            self.store.legacy_directory
            / self.artifact.lookup.key
            / f"{digest(self.artifact.model_dump(mode='json'))}.json"
        )
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(raw)
        self.assertEqual(self.lookup(), self.artifact)
        self.assertFalse(self.store.shared_directory.exists())
        options = {
            "original_run": identity(),
            "invocation": self.artifact.invocation,
            "registry": self.registry,
            "validate": validate_v2,
        }
        self.assertTrue(self.store.import_typed(raw, **options))
        self.assertEqual(legacy.read_bytes(), raw)
        leaf, _, request = self.commit()
        new = leaf.read_bytes()
        self.assertEqual(self.store.read(raw), self.store.read(new))
        self.assertTrue(self.store.import_typed(new, **options))
        request.unlink()
        self.assertFalse(self.store.import_typed(new, **options))
        # A broken new reference does not hide an independently valid old record.
        self.assertEqual(self.lookup(), self.artifact)
        legacy.write_bytes(b"{}")
        self.assertIsNone(self.lookup())

    def test_missing_tampered_wrong_kind_and_substituted_records_miss(
        self,
    ) -> None:
        """Revalidate both shared records on every lookup, including repeated reads."""
        leaf, invocation, request = self.commit()
        originals = {p: p.read_bytes() for p in (leaf, invocation, request)}
        self.assertEqual(self.lookup(), self.artifact)
        for target, original in originals.items():
            for damage in (
                None,
                b"{",
                b"{}",
                originals[request if target != request else invocation],
            ):
                with self.subTest(
                    target=target.parent.name, damage=damage is None
                ):
                    if damage is None:
                        target.unlink()
                    else:
                        target.write_bytes(damage)
                    self.assertIsNone(self.lookup())
                    target.write_bytes(original)
                    self.assertEqual(self.lookup(), self.artifact)
        # Valid full content with the same semantic key still has a different address.
        other = changed_record(
            self.artifact.invocation,
            batch=changed_record(self.artifact.invocation.batch, partition=1),
        )
        invocation.write_text(encode_invocation(other).model_dump_json())
        self.assertIsNone(self.lookup())
        invocation.write_bytes(originals[invocation])
        value = json.loads(originals[request])
        value["metadata"]["safety_tokens"] += 1
        request.write_text(json.dumps(value))
        self.assertIsNone(self.lookup())

    def test_exact_byte_memo_reuses_shared_parses_and_current_audits(
        self,
    ) -> None:
        """Sibling hits read every record but share only structural parse work."""
        for artifact in self.artifacts:
            self.store.commit(artifact)
        validate = Mock(wraps=validate_v2)
        audit = Mock(return_value=True)
        first = self.store.lookup(
            self.artifacts[0].lookup,
            self.registry,
            validate,
            audit_candidate=audit,
        )
        second = self.store.lookup(
            self.artifacts[1].lookup,
            self.registry,
            validate,
            audit_candidate=audit,
        )
        repeated = self.store.lookup(
            self.artifacts[0].lookup,
            self.registry,
            validate,
            audit_candidate=audit,
        )

        self.assertEqual(first, self.artifacts[0])
        self.assertEqual(second, self.artifacts[1])
        self.assertEqual(repeated, self.artifacts[0])
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(audit.call_count, 3)
        counters = self.store.memo_counters
        self.assertEqual(counters["leaf_reads"], 3)
        self.assertEqual(counters["invocation_reads"], 3)
        self.assertEqual(counters["request_reads"], 3)
        self.assertEqual(counters["reads"], 9)
        self.assertEqual(counters["leaf_parses"], 2)
        self.assertEqual(counters["invocation_parses"], 1)
        self.assertEqual(counters["request_parses"], 1)
        self.assertEqual(counters["expansions"], 1)

    def test_failed_parse_is_not_retained(self) -> None:
        """Malformed exact bytes are parsed again until valid bytes return."""
        _leaf, invocation, _request = self.commit()
        self.assertEqual(self.lookup(), self.artifact)
        original = invocation.read_bytes()
        before = self.store.memo_counters
        invocation.write_bytes(b"{")
        self.assertIsNone(self.lookup())
        failed_once = self.store.memo_counters
        self.assertIsNone(self.lookup())
        failed_twice = self.store.memo_counters
        self.assertEqual(
            failed_twice["invocation_parses"],
            failed_once["invocation_parses"] + 1,
        )
        self.assertEqual(failed_twice["entries"], before["entries"])
        self.assertEqual(
            failed_twice["retained_bytes"], before["retained_bytes"]
        )
        invocation.write_bytes(original)
        self.assertEqual(self.lookup(), self.artifact)
        restored = self.store.memo_counters
        self.assertEqual(restored["invocation_parses"], 3)
        self.assertEqual(restored["expansions"], 1)

    def test_returned_models_and_audit_mutation_do_not_alias_memo(
        self,
    ) -> None:
        """Frozen-model bypasses stay detached from later reads of memo entries."""
        self.commit()

        def mutate(candidate: runtime.StoredArtifact) -> bool:
            object.__setattr__(candidate.invocation.batch, "input_bound", 0)
            object.__setattr__(
                candidate.invocation.lookups[0], "validator_version", "changed"
            )
            object.__setattr__(
                candidate.production_request, "safety_tokens", 0
            )
            object.__setattr__(
                candidate.record.result, "unit_key", digest("changed")
            )
            return True

        returned = self.lookup(audit_candidate=mutate)
        self.assertIsNotNone(returned)
        before = self.store.memo_counters
        assert returned is not None
        object.__setattr__(returned, "lookup", "changed")
        object.__setattr__(returned.record, "result", "changed")

        later = self.lookup(audit_candidate=lambda _candidate: True)
        after = self.store.memo_counters
        self.assertEqual(later, self.artifact)
        self.assertEqual(after["leaf_parses"], before["leaf_parses"])
        self.assertEqual(
            after["invocation_parses"], before["invocation_parses"]
        )
        self.assertEqual(after["request_parses"], before["request_parses"])
        self.assertEqual(after["expansions"], before["expansions"])
        self.assertEqual(after["reads"], before["reads"] + 3)

    def test_store_memo_is_bounded_and_independent(self) -> None:
        """Eviction and oversized records only add work; stores share no authority."""
        self.commit()
        sizing = runtime.ArtifactStore(self.root)
        sized_hit = sizing.lookup(
            self.artifact.lookup, self.registry, validate_v2
        )
        self.assertEqual(sized_hit, self.artifact)
        weights = [entry.retained_bytes for entry in sizing._memo.values()]  # noqa: SLF001
        self.assertEqual(len(weights), 3)
        self.assertGreater(sum(weights), max(weights))

        capacity = sum(weights) - min(weights)
        bounded = runtime.ArtifactStore(self.root, _memo_max_bytes=capacity)
        first = bounded.lookup(
            self.artifact.lookup, self.registry, validate_v2
        )
        second = bounded.lookup(
            self.artifacts[1].lookup, self.registry, validate_v2
        )
        repeated = bounded.lookup(
            self.artifact.lookup, self.registry, validate_v2
        )
        self.assertEqual(first, self.artifact)
        self.assertEqual(second, self.artifacts[1])
        self.assertEqual(repeated, self.artifact)
        bounded_counts = bounded.memo_counters
        self.assertGreater(bounded_counts["evictions"], 0)
        self.assertGreater(bounded_counts["parses"], 3)
        self.assertLessEqual(
            bounded_counts["retained_bytes"], bounded_counts["capacity_bytes"]
        )

        oversized = runtime.ArtifactStore(self.root, _memo_max_bytes=1)
        oversized_first = oversized.lookup(
            self.artifact.lookup, self.registry, validate_v2
        )
        oversized_second = oversized.lookup(
            self.artifact.lookup, self.registry, validate_v2
        )
        self.assertEqual(oversized_first, self.artifact)
        self.assertEqual(oversized_second, self.artifact)
        oversized_counts = oversized.memo_counters
        self.assertEqual(oversized_counts["entries"], 0)
        self.assertEqual(oversized_counts["retained_bytes"], 0)
        self.assertEqual(oversized_counts["oversized_records"], 6)
        self.assertEqual(oversized_counts["expansions"], 2)

        independent = runtime.ArtifactStore(self.root)
        audit = Mock(return_value=True)
        local_hit = independent.lookup(
            self.artifact.lookup,
            self.registry,
            validate_v2,
            audit_candidate=audit,
        )
        other_hit = sizing.lookup(
            self.artifact.lookup,
            self.registry,
            validate_v2,
            audit_candidate=audit,
        )
        self.assertEqual(local_hit, self.artifact)
        self.assertEqual(other_hit, self.artifact)
        self.assertIsNot(local_hit, other_hit)
        self.assertIsNot(local_hit.invocation, other_hit.invocation)
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(independent.memo_counters["leaf_parses"], 1)
        self.assertEqual(independent.memo_counters["invocation_parses"], 1)
        self.assertEqual(independent.memo_counters["request_parses"], 1)
        self.assertEqual(independent.memo_counters["expansions"], 1)

    def test_rehashed_missing_peer_and_leaf_binding_forgery_miss(self) -> None:
        """A valid storage checksum cannot remove an original peer or change item identity."""
        leaf, invocation, _request = self.commit()
        self.assertEqual(self.lookup(), self.artifact)
        original_leaf = leaf.read_bytes()
        encoded = json.loads(invocation.read_bytes())
        encoded["dynamic_dependencies"] = [
            d
            for d in encoded["dynamic_dependencies"]
            if not (
                d["kind"] == "peer"
                and d["key"] == self.artifacts[1].lookup.item_key
            )
        ]
        forged = StoredInvocation.model_validate_json(json.dumps(encoded))
        new_digest = storage_digest(forged)
        invocation.with_name(f"{new_digest}.json").write_text(
            forged.model_dump_json()
        )
        data = json.loads(original_leaf)
        data["invocation_digest"] = new_digest
        item = runtime._StoredItem.model_validate_json(json.dumps(data))  # noqa: SLF001
        new_leaf = leaf.with_name(f"{storage_digest(item)}.json")
        new_leaf.write_text(item.model_dump_json())
        leaf.unlink()
        self.assertIsNone(self.lookup())
        new_leaf.unlink()
        for field, value in (
            ("semantic_key", digest("forged")),
            ("item_key", self.artifacts[1].lookup.item_key),
        ):
            data = json.loads(original_leaf)
            data[field] = value
            item = runtime._StoredItem.model_validate_json(json.dumps(data))  # noqa: SLF001
            new_leaf = leaf.with_name(f"{storage_digest(item)}.json")
            new_leaf.write_text(item.model_dump_json())
            self.assertIsNone(self.lookup())
            new_leaf.unlink()

    def test_rehashed_exact_request_forgery_reaches_audit(self) -> None:
        """Producer auditing rejects coherent rehashed request changes outside miss handling."""
        leaf, invocation, request = self.commit()
        self.assertEqual(self.lookup(), self.artifact)
        changed = changed_record(
            self.artifact.invocation,
            batch=changed_record(self.artifact.invocation.batch, partition=1),
        )
        encoded = encode_invocation(changed)
        invocation_key = storage_digest(encoded)
        invocation.with_name(f"{invocation_key}.json").write_text(
            encoded.model_dump_json()
        )
        metadata = changed_record(
            self.artifact.production_request, safety_tokens=1
        )
        stored_request = runtime._StoredRequest(metadata=metadata)  # noqa: SLF001
        request_key = storage_digest(stored_request)
        request.with_name(f"{request_key}.json").write_text(
            stored_request.model_dump_json()
        )
        item = runtime._StoredItem.model_validate_json(leaf.read_bytes())  # noqa: SLF001
        item = changed_record(
            item, invocation_digest=invocation_key, request_digest=request_key
        )
        leaf.with_name(f"{storage_digest(item)}.json").write_text(
            item.model_dump_json()
        )
        leaf.unlink()

        def audit(candidate: runtime.StoredArtifact) -> bool:
            self.assertEqual(candidate.invocation, changed)
            self.assertEqual(candidate.production_request, metadata)
            msg = "request_mismatch"
            raise runtime.ArtifactRuntimeError(msg)

        with self.assertRaisesRegex(
            runtime.ArtifactRuntimeError, "request_mismatch"
        ):
            self.lookup(audit_candidate=audit)

    def test_same_semantics_exact_invocations_and_requests_remain_distinct(
        self,
    ) -> None:
        """Neither a semantic key nor a supplied batch ID can alias full metadata."""
        self.commit()
        invocation = changed_record(
            self.artifact.invocation,
            batch=changed_record(
                self.artifact.invocation.batch,
                input_bound=self.artifact.invocation.batch.input_bound + 1,
            ),
        )
        metadata = changed_record(
            self.artifact.production_request, safety_tokens=1
        )
        other = changed_record(
            self.artifact, invocation=invocation, production_request=metadata
        )
        self.assertEqual(
            other.invocation.semantic_key,
            self.artifact.invocation.semantic_key,
        )
        self.assertEqual(
            other.record.semantic_key, self.artifact.record.semantic_key
        )
        self.assertEqual(
            other.invocation.batch.id, self.artifact.invocation.batch.id
        )
        self.store.commit(other)
        self.assertEqual(
            len(
                list(
                    (self.store.shared_directory / "invocations").glob(
                        "*.json"
                    )
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                list((self.store.shared_directory / "requests").glob("*.json"))
            ),
            2,
        )
        leaves = (self.store.directory / self.artifact.lookup.key).glob(
            "*.json"
        )
        self.assertEqual(
            {self.store.read(p.read_bytes()) for p in leaves},
            {self.artifact, other},
        )
        self.assertEqual(
            self.lookup(audit_candidate=lambda a: a == other), other
        )

    def test_same_address_collision_fails_without_replacement(self) -> None:
        """Reject malformed and fully typed conflicting content at existing addresses."""
        paths = self.commit()
        for path in paths:
            original = path.read_bytes()
            valid = json.loads(original)
            if path.parent.name == "invocations":
                valid["batch"]["partition"] += 1
            elif path.parent.name == "requests":
                valid["metadata"]["safety_tokens"] += 1
            else:
                valid["original_run_digest"] = digest("other-run")
            for raw in (b"{", b"{}", json.dumps(valid).encode()):
                with self.subTest(
                    kind=path.parent.name, typed=raw.startswith(b'{"')
                ):
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(
                        runtime.ArtifactRuntimeError, "storage"
                    ):
                        self.store.commit(self.artifact)
                    self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(original)
        self.store.commit(self.artifact)
        self.assertEqual(self.lookup(), self.artifact)

    def test_interruption_between_shared_dependencies_and_leaf(self) -> None:
        """Dependency publication alone cannot expose a partially committed item."""
        publish = runtime._immutable_cache_record  # noqa: SLF001
        seen = []

        def interrupt(path: Path, record: Contract) -> None:
            seen.append(path)
            if path.parent.parent == self.store.directory:
                self.assertEqual(
                    len(
                        list(
                            (self.store.shared_directory / "invocations").glob(
                                "*.json"
                            )
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    len(
                        list(
                            (self.store.shared_directory / "requests").glob(
                                "*.json"
                            )
                        )
                    ),
                    1,
                )
                raise KeyboardInterrupt
            publish(path, record)

        with (
            patch.object(runtime, "_immutable_cache_record", interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.store.commit(self.artifact)
        self.assertEqual(len(seen), 3)
        self.assertIsNone(self.lookup())
        self.assertEqual(list(self.store.directory.glob("*/*.json")), [])
        self.store.commit(self.artifact)
        self.assertEqual(self.lookup(), self.artifact)
        self.assertEqual(list(self.store.shared_directory.rglob(".v3-*")), [])

    def test_concurrent_writers_verify_one_immutable_shared_pair(self) -> None:
        """Independent stores racing to publish keep every leaf and one exact pair."""
        barrier = threading.Barrier(8)

        def commit(position: int) -> None:
            barrier.wait(timeout=5)
            runtime.ArtifactStore(self.root).commit(
                self.artifacts[position % 2]
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(commit, range(8)))
        self.assertEqual(self.lookup(), self.artifact)
        self.assertEqual(len(list(self.store.directory.glob("*/*.json"))), 2)
        self.assertEqual(
            len(
                list(
                    (self.store.shared_directory / "invocations").glob(
                        "*.json"
                    )
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                list((self.store.shared_directory / "requests").glob("*.json"))
            ),
            1,
        )
        for path in self.store.shared_directory.rglob("*"):
            self.assertEqual(
                path.stat().st_mode & 0o777, 0o700 if path.is_dir() else 0o600
            )

    def test_dependency_fsync_failure_prevents_leaf_publication(self) -> None:
        """A failed durable dependency commit cannot publish a referencing leaf."""
        with (
            patch.object(os, "fsync", side_effect=OSError("synthetic")),
            self.assertRaisesRegex(runtime.ArtifactRuntimeError, "storage"),
        ):
            self.store.commit(self.artifact)
        self.assertIsNone(self.lookup())
        self.assertEqual(list(self.store.directory.glob("*/*.json")), [])

    def test_shared_records_are_fsynced_before_leaf_publication(self) -> None:
        """Observe file and directory durability before each dependent hard link."""
        native_link, native_fsync = os.link, os.fsync
        synced_files = set()
        published = {}

        def fsync(fd: int) -> None:
            native_fsync(fd)
            inode = os.fstat(fd).st_ino
            synced_files.add(inode)
            for path in published:
                if path.parent.stat().st_ino == inode:
                    published[path] = True

        def link(source: Path, target: Path) -> None:
            self.assertIn(source.stat().st_ino, synced_files)
            self.assertTrue(all(published.values()))
            native_link(source, target)
            published[target] = False

        with patch.object(os, "fsync", fsync), patch.object(os, "link", link):
            self.store.commit(self.artifact)
        self.assertEqual(len(published), 3)
        self.assertTrue(all(published.values()))
        self.assertEqual(self.lookup(), self.artifact)

    def test_concurrent_content_collision_rejects_loser(self) -> None:
        """A forced full-digest collision cannot replace the winning producer."""
        other = changed_record(
            self.artifact,
            invocation=changed_record(
                self.artifact.invocation,
                batch=changed_record(
                    self.artifact.invocation.batch, partition=1
                ),
            ),
        )
        barrier = threading.Barrier(2)

        def commit(artifact: runtime.StoredArtifact) -> object:
            barrier.wait(timeout=5)
            try:
                runtime.ArtifactStore(self.root).commit(artifact)
            except runtime.ArtifactRuntimeError as error:
                return error.reason
            return artifact

        with (
            patch.object(runtime, "storage_digest", return_value="c" * 64),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            outcomes = list(pool.map(commit, (self.artifact, other)))
            self.assertEqual(outcomes.count("storage"), 1)
            winner = next(value for value in outcomes if value != "storage")
            self.assertEqual(self.lookup(), winner)
        self.assertEqual(
            len(list(self.store.shared_directory.rglob("*.json"))), 3
        )

    def test_shared_cache_loss_does_not_renew_completed_allowance(
        self,
    ) -> None:
        """Completed recovery remains authoritative if optional shared content is lost."""
        invocation, registry, results = fixture_v2(2)
        send = Mock(
            return_value=wire_v2(
                [r.model_dump(mode="json") for r in results.values()]
            )
        )
        callbacks = runtime.BatchCallbacks(prepare_v2, send, validate_v2)
        first = runtime.ArtifactRuntime(
            identity(), recovery_dir=self.root, artifact_dir=self.root
        )
        first.execute_batch(invocation, registry, callbacks)
        previous = first.recovery.load(invocation.semantic_key)
        next(
            (self.store.shared_directory / "invocations").glob("*.json")
        ).unlink()
        resumed = runtime.ArtifactRuntime(
            identity(), recovery_dir=self.root, artifact_dir=self.root
        )
        with self.assertRaisesRegex(
            runtime.ArtifactRuntimeError, "results_unavailable"
        ):
            resumed.execute_batch(invocation, registry, callbacks)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(
            first.recovery.load(invocation.semantic_key), previous
        )

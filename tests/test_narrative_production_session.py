# ruff: noqa: SLF001  # Adversarial tests must inspect private session registrations.

"""Focused tests for shared production captures and incomplete stage bindings."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from contextlib import nullcontext
from types import MappingProxyType
from unittest.mock import patch

from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.narrative import Narrator
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_protocol import V2Options
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_runtime import RuntimeStageError
from diffstory.narrative_session import ReportSession
from diffstory.narrative_v2 import BatchedNarrator
from diffstory.narrative_v2 import StageInputs
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import planning_waves


class NarrativeProductionSessionTests(unittest.TestCase):
    """Use only the in-process fixture provider and synthetic reports."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one locally generated manifest for offline proof cases."""
        cls.report = golden_report(2)
        provider = FixtureProvider()
        with tempfile.TemporaryDirectory() as directory:
            cls.annotations = Narrator(
                provider,
                architecture="v2",
                recovery_dir=directory,
                concurrency=1,
            ).generate(cls.report)
        cls.configuration = copy.deepcopy(
            cls.annotations["generation"]["request_configuration"]
        )

    def setUp(self) -> None:
        """Give coordinator tests isolated local recovery storage."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _coordinator(
        self,
        report: dict,
        provider: FixtureProvider,
        *,
        options: V2Options | None = None,
    ) -> tuple[BatchedNarrator, ReportSession, V2Options]:
        """Build a session and coordinator from one exact local descriptor."""
        selected = options or V2Options()
        descriptor = Transport(provider, selected).descriptor
        session = ReportSession.capture(report, descriptor)
        coordinator = BatchedNarrator(
            provider,
            recovery_dir=self.temp.name,
            concurrency=1,
            options=selected,
        )
        return coordinator, session, selected

    def test_preview_and_production_inputs_share_one_capture_and_index(
        self,
    ) -> None:
        """Duplicate report trees or ledgers could diverge after preview approval."""
        cases = [
            (
                "python-shape",
                {
                    **copy.deepcopy(self.report),
                    "shape_probe": {"tuple": (True, 1), "list": [True, 1]},
                },
                {"tuple": (True, 1), "list": [True, 1]},
            )
        ]
        for name, report, expected_shape in cases:
            with self.subTest(name=name):
                provider = FixtureProvider()
                coordinator, session, _options = self._coordinator(
                    report, provider
                )
                legacy_wire = json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                with (
                    patch(
                        "diffstory.narrative_v2.build_evidence_ledger",
                        wraps=build_evidence_ledger,
                    ) as ledgers,
                    patch(
                        "diffstory.narrative_v2.build_ledger_index",
                        wraps=build_ledger_index,
                    ) as indexes,
                ):
                    preview = coordinator.preview(report, session=session)
                    self.assertEqual(ledgers.call_count, 1)
                    self.assertEqual(indexes.call_count, 1)

                capture = session._captured_input
                self.assertIsNotNone(capture)
                self.assertIs(coordinator.inputs._captured_input, capture)
                self.assertIs(coordinator.inputs.report, capture.report_view)
                self.assertIs(coordinator.inputs.index, capture.index)
                self.assertIs(coordinator.inputs.ledger, capture.ledger)
                self.assertIs(coordinator.inputs.run, capture.run)
                self.assertIs(coordinator.inputs.author, capture.author)
                self.assertEqual(
                    coordinator.inputs.author_digest, capture.author_digest
                )
                self.assertEqual(
                    capture.compiler_signature,
                    compiler_fact_signature(report),
                )
                self.assertIs(coordinator.approved_report, report)
                self.assertEqual(
                    preview["approval_digest"],
                    hashlib.sha256(legacy_wire).hexdigest(),
                )
                self.assertEqual(
                    capture.report_digest, content_digest(legacy_wire)
                )
                self.assertEqual(
                    session.captured_report()["shape_probe"], expected_shape
                )
                self.assertNotIn("_report_bytes", session.__slots__)
                self.assertNotIn("_report_snapshot", session.__slots__)
                self.assertFalse(hasattr(coordinator.inputs, "report_bytes"))
                self.assertEqual(provider.calls, [])

                with patch(
                    "diffstory.narrative_session._freeze_report",
                    side_effect=AssertionError("approval copied the report"),
                ):
                    coordinator.check_approval(report)

                report["meta"]["description"] = "caller mutation after capture"
                with self.assertRaisesRegex(
                    ArtifactRuntimeError, "dependencies_changed"
                ):
                    coordinator.check_approval(report)
                self.assertEqual(
                    session.captured_report()["meta"]["description"],
                    self.report["meta"]["description"],
                )
                session.close()

    def test_approval_mutations_are_rejected_before_the_next_send(
        self,
    ) -> None:
        """A report, provider, packing, prompt, schema, or version change could leak an unapproved request."""
        cases = [
            ("source", "report-source"),
            ("raw-scalar", "report-scalar-type"),
            ("model", "provider-model"),
            ("destination", "provider-destination"),
            ("capacity", "provider-capacity"),
            ("packing", "packing-bound"),
            ("prompt", "prompt-version"),
            ("schema", "schema-version"),
            ("validator", "validator-version"),
        ]
        for mutation, _name in cases:
            with self.subTest(mutation=mutation):
                report = copy.deepcopy(self.report)
                if mutation == "raw-scalar":
                    report["shape_guard"] = True
                provider = FixtureProvider()
                coordinator, session, options = self._coordinator(
                    report, provider
                )
                coordinator.preview(report, session=session)
                patcher = nullcontext()
                if mutation == "source":
                    report["source_evidence"]["fragments"][0]["text"] += "x"
                elif mutation == "raw-scalar":
                    report["shape_guard"] = 1
                elif mutation in {"model", "destination", "capacity"}:
                    attribute = {
                        "model": "model",
                        "destination": "destination",
                        "capacity": "context_tokens",
                    }[mutation]
                    setattr(provider, attribute, "changed")
                elif mutation == "packing":
                    values = options.model_dump(mode="python")
                    values["packing"]["framing_bytes"] += 1
                    coordinator.options = V2Options.model_validate(values)
                elif mutation == "prompt":
                    from diffstory.narrative_v2 import PROMPTS

                    patcher = patch.dict(
                        PROMPTS, {"detail": PROMPTS["detail"] + " changed"}
                    )
                elif mutation == "schema":
                    patcher = patch(
                        "diffstory.narrative_v2.wire_schema",
                        return_value={
                            "type": "object",
                            "properties": {"changed": {"type": "string"}},
                        },
                    )
                elif mutation == "validator":
                    patcher = patch(
                        "diffstory.narrative_v2.VALIDATOR_VERSION",
                        "narrator-v2-mutated",
                    )
                with (
                    patcher,
                    self.assertRaisesRegex(
                        ArtifactRuntimeError, "dependencies_changed"
                    ),
                ):
                    coordinator.generate(report)
                self.assertEqual(provider.calls, [])
                session.close()

    def test_approval_is_rechecked_after_each_fixture_reply(self) -> None:
        """A change after one reply could otherwise authorize the next batch."""
        report = copy.deepcopy(self.report)

        def mutate_after_first_reply(
            provider: FixtureProvider, _data: dict, response: dict
        ) -> dict:
            if len(provider.calls) == 1:
                report["meta"]["description"] = "changed during fixture run"
            return response

        provider = FixtureProvider(mutate_after_first_reply)
        coordinator, session, _options = self._coordinator(report, provider)
        coordinator.preview(report, session=session)
        with self.assertRaises((ArtifactRuntimeError, RuntimeStageError)):
            coordinator.generate(report)
        self.assertEqual(len(provider.calls), 1)
        session.close()

    def test_internal_report_ledger_index_run_and_source_rebindings_fail(
        self,
    ) -> None:
        """Rebound mutable coordinator attributes could detach checks from capture."""
        report = copy.deepcopy(self.report)
        provider = FixtureProvider()
        coordinator, session, _options = self._coordinator(report, provider)
        coordinator.preview(report, session=session)
        coordinator.inputs.sources = object()
        with self.assertRaises((ArtifactRuntimeError, ValueError)):
            coordinator.check_approval(report)
        coordinator.inputs.sources = None
        coordinator.inputs.bind_details(
            GenerationManifest.model_validate_json(
                canonical_bytes(self.annotations["generation"])
            ).details
        )
        cases = [
            ("report", {"rebound": True}),
            ("ledger", object()),
            ("index", object()),
            ("run", object()),
            ("author", object()),
            ("sources", object()),
        ]
        for field, replacement in cases:
            with self.subTest(field=field):
                original = getattr(coordinator.inputs, field)
                setattr(coordinator.inputs, field, replacement)
                with self.assertRaises((ArtifactRuntimeError, ValueError)):
                    coordinator.check_approval(report)
                setattr(coordinator.inputs, field, original)
                coordinator.check_approval(report)
        self.assertEqual(provider.calls, [])
        session.close()

    def test_raw_derived_model_mutations_fail_before_registration_or_send(
        self,
    ) -> None:
        """Frozen model bypasses could alter approved production authority."""
        cases = [
            ("run", "report_signature", "f" * 64),
            ("author", "title", "changed author title"),
            ("ledger", "buffers", ()),
            ("index", "units", MappingProxyType({})),
            ("options.packing", "framing_bytes", True),
        ]
        scopes = [
            ("input-check", "ValueError"),
            ("approval", "ArtifactRuntimeError"),
            ("stage-registration", "ValueError"),
            ("first-dispatch", "ArtifactRuntimeError"),
        ]
        for model_name, field, changed_value in cases:
            for scope, expected_error in scopes:
                with self.subTest(model=model_name, scope=scope):
                    report = copy.deepcopy(self.report)
                    provider = FixtureProvider()
                    coordinator, session, _options = self._coordinator(
                        report, provider
                    )
                    self.addCleanup(session.close)
                    coordinator.preview(report, session=session)
                    target = coordinator.inputs
                    for component in model_name.split("."):
                        target = getattr(target, component)
                    original_value = getattr(target, field)
                    object.__setattr__(target, field, changed_value)
                    try:
                        with self.assertRaisesRegex(
                            (ValueError, ArtifactRuntimeError),
                            "captured_|dependencies_changed|ledger_index_",
                        ) as raised:
                            if scope == "input-check":
                                coordinator.inputs.assert_capture_current()
                            elif scope == "approval":
                                coordinator.check_approval(report)
                            elif scope == "stage-registration":
                                stage = StageInputs(
                                    coordinator.inputs,
                                    "detail",
                                    coordinator.inputs.detail_items(),
                                )
                                session._bind_stage(stage)
                            else:
                                coordinator.generate(report)
                        self.assertEqual(
                            type(raised.exception).__name__, expected_error
                        )
                        self.assertEqual(provider.calls, [])
                    finally:
                        object.__setattr__(target, field, original_value)
                    coordinator.check_approval(report)
                    session.close()

    def test_run_signature_fact_signature_and_annotated_hash_stay_distinct(
        self,
    ) -> None:
        """A full run signature must not be replaced by compiler facts or output hash."""
        report = copy.deepcopy(self.report)
        generation = self.annotations["generation"]
        from diffstory.analysis import apply_annotations

        annotated_report = apply_annotations(report, self.annotations)
        cases = [
            (
                "pre-annotation-run",
                generation["run"]["report_signature"],
                content_digest(canonical_bytes(report)),
                True,
            ),
            (
                "compiler-facts",
                generation["compiler_fact_signature"],
                compiler_fact_signature(report),
                True,
            ),
            (
                "annotated-report",
                content_digest(canonical_bytes(annotated_report)),
                generation["run"]["report_signature"],
                False,
            ),
        ]
        for name, actual, expected, matches in cases:
            with self.subTest(name=name):
                if matches:
                    self.assertEqual(actual, expected)
                else:
                    self.assertNotEqual(actual, expected)

    def test_actual_stage_bindings_are_ordered_local_and_incomplete(  # noqa: PLR0915 # One focused matrix covers registration and its rejection boundaries.
        self,
    ) -> None:
        """A copied, rebound, partial, or closed stage must not gain proof authority."""
        report = copy.deepcopy(self.report)
        provider = FixtureProvider()
        coordinator, session, _options = self._coordinator(report, provider)
        coordinator.preview(report, session=session)
        annotations = coordinator.generate(report)
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(annotations["generation"])
        )
        registrations = tuple(session._stage_bindings.values())
        bindings = tuple(binding for _stage, binding in registrations)
        expected_names = [
            "detail",
            *("planning" for _wave in planning_waves(manifest.plan)),
            "section",
            "document",
        ]
        self.assertEqual(
            [binding.stage_name for binding in bindings], expected_names
        )
        for position, (stage, binding) in enumerate(registrations):
            self.assertIs(binding.stage, stage)
            self.assertEqual(binding.sequence_index, position)
            self.assertEqual(binding.item_keys, tuple(stage.items))
            self.assertEqual(
                binding.evidence_kind, "incomplete-stage-binding.v1"
            )
            self.assertIs(session._check_stage_binding(stage), binding)
            with self.assertRaises(ValueError):
                session.apply(binding)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                session._check_stage_binding(copy.copy(stage))
            view = stage.identity_view(
                "legacy", stage.transport.descriptor.fixture_json_layers
            )
            with self.assertRaises(ValueError):
                session._check_stage_binding(view)
            with self.assertRaises((AttributeError, TypeError)):
                binding.evidence_kind = "complete"  # type: ignore[misc]

        partial_data = copy.deepcopy(annotations["generation"])
        partial_data["details"] = partial_data["details"][:-1]
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            session.validate_import(partial_data)
        self.assertFalse(
            any(
                kind == "generation"
                for _handle, kind, _data in session._registrations.values()
            )
        )
        self.assertFalse(hasattr(session, "finish_production"))

        other = ReportSession.capture(report, self.configuration)
        with self.assertRaises(ValueError):
            other._check_stage_binding(bindings[0].stage)
        other.close()

        first_stage = bindings[0].stage
        rebuilt = type(first_stage)(
            first_stage.inputs,
            first_stage.stage,
            tuple(first_stage.items.values()),
        )
        with self.assertRaises(ValueError):
            session._check_stage_binding(rebuilt)
        with self.assertRaises((TypeError, ValueError)):
            session._check_stage_binding(copy.deepcopy(first_stage))
        with self.assertRaises(
            (AttributeError, pickle.PicklingError, TypeError, ValueError)
        ):
            serialized = pickle.dumps(first_stage)
            session._check_stage_binding(pickle.loads(serialized))  # noqa: S301 # Decode only a self-generated local fixture for an identity-rejection check.
        original_items = first_stage.items
        first_stage.items = MappingProxyType(
            dict(list(original_items.items())[:-1])
        )
        with self.assertRaisesRegex(ValueError, "stage_binding_changed"):
            session._check_stage_binding(first_stage)
        first_stage.items = original_items
        self.assertIs(session._check_stage_binding(first_stage), bindings[0])

        capture = session._captured_input
        session.close()
        self.assertEqual(session._stage_bindings, {})
        self.assertIsNone(session._captured_input)
        self.assertEqual(
            (
                coordinator.inputs.report,
                coordinator.inputs.ledger,
                coordinator.inputs.index,
                coordinator.inputs.sources,
                coordinator.inputs.transport,
                coordinator.inputs.options,
                coordinator.inputs._captured_input,
            ),
            (None, None, None, None, None, None, None),
        )
        self.assertIsNone(coordinator.approved_report)
        self.assertIsNone(coordinator.session)
        self.assertEqual(coordinator.stages, [])
        self.assertEqual(coordinator.reused, set())
        self.assertIsNone(coordinator.runtime)
        self.assertIsNone(coordinator.provider)
        self.assertIsNone(coordinator.options)
        with self.assertRaisesRegex(ValueError, "session_closed"):
            capture.canonical_report_bytes()
        with self.assertRaisesRegex(ValueError, "session_closed"):
            session._check_stage_binding(first_stage)

    def test_session_import_still_rejects_partial_proof_without_authorizing_data(
        self,
    ) -> None:
        """A data capture or incomplete manifest could otherwise look like proof."""
        report = copy.deepcopy(self.report)
        partial = copy.deepcopy(self.annotations["generation"])
        partial["details"] = partial["details"][:-1]
        descriptor = self.configuration
        with ReportSession.capture(report, descriptor) as session:
            with self.assertRaises((ValueError, ArtifactRuntimeError)):
                session.validate_import(partial)
            self.assertEqual(session._registrations, {})
            self.assertNotIn("finish_production", dir(session))


if __name__ == "__main__":
    unittest.main()
